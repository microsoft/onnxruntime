// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#include <unordered_map>

#ifdef ORT_MINIMAL_BUILD

ORT_API_STATUS_IMPL(OrtApis::CreateOpAttr,
                    _In_ const char*,
                    _In_ const void*,
                    _In_ int,
                    _In_ OrtOpAttrType,
                    _Out_ OrtOpAttr**) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "CreateOpAttr is not implemented for minimal build.");
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseOpAttr, _Frees_ptr_opt_ OrtOpAttr*) {
}

ORT_API_STATUS_IMPL(OrtApis::CreateOp,
                    _In_ const OrtKernelInfo*,
                    _In_ const char*,
                    _In_ const char*,
                    _In_ int,
                    _In_ const char**,
                    _In_ const ONNXTensorElementDataType*,
                    _In_ int,
                    _In_ const OrtOpAttr* const*,
                    _In_ int,
                    _Out_ OrtOp**) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "CreateOp is not implemented for minimal build.");
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeOp,
                    _In_ const OrtKernelContext*,
                    _In_ const OrtOp*,
                    _In_ const OrtValue* const*,
                    _In_ int,
                    _Inout_ OrtValue* const*,
                    _In_ int) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "InvokeOp is not implemented for minimal build.");
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseOp, _Frees_ptr_opt_ OrtOp*) {
}

ORT_API_STATUS_IMPL(OrtApis::GetExecutionProvider,
      _In_ const OrtKernelInfo*,
      _Outptr_ OrtExecutionProvider**) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "GetExecutionProvider is not implemented for minimal build.");
  API_IMPL_END
}
#else

namespace onnxruntime {
namespace standalone {

class NodeHolder {
 public:

  NodeHolder() = default;
  NodeHolder(NodeHolder&& node_holder) = default;
  NodeHolder& operator=(NodeHolder&& node_holder) = default;

  NodeHolder(onnxruntime::Node* const node) : node_(node){};

  NodeHolder(const NodeHolder& node_holder) {
    node_ = node_holder.node_;
    const_cast<NodeHolder&>(node_holder).node_ = {};
  }

  NodeHolder& operator = (const NodeHolder& node_holder) {
    node_ = node_holder.node_;
    const_cast<NodeHolder&>(node_holder).node_ = {};
    return *this;
  }

  bool operator != (const NodeHolder& node_holder) const {
    return node_ != node_holder.node_;
  }

  onnxruntime::Node* get() const {
    return node_;
  }

  ~NodeHolder() {
    if (node_) {
      std::cout << "releasing " << node_->OpType() << std::endl;
      for (auto* input_arg : node_->InputDefs()) {
        delete input_arg;
      }
      for (auto* output_arg : node_->OutputDefs()) {
        delete output_arg;
      }
      delete node_;
    }
  }
private:
  onnxruntime::Node* node_{};
};

using StandAloneNodes = std::unordered_map<const onnxruntime::OpKernel*, NodeHolder>;

StandAloneNodes& GetNodes() {
  static StandAloneNodes nodes;
  return nodes;
}

std::mutex& GetMutex() {
  static std::mutex mtx;
  return mtx;
}

void AddNode(const onnxruntime::OpKernel* kernel, const NodeHolder& node) {
  std::lock_guard<std::mutex> guard(GetMutex());
  StandAloneNodes nodes = GetNodes();
  auto iter = nodes.find(kernel);
  if (iter != nodes.end() && iter->second != node) {
    ORT_THROW("kernel mapped to two different nodes!");
  }
  nodes[kernel] = node;
}  

void DelNode(const onnxruntime::OpKernel* kernel) {
  std::lock_guard<std::mutex> guard(GetMutex());
  GetNodes().erase(kernel);
}

void UpdateNode(const onnxruntime::OpKernel* kernel,
                const OrtValue* const* input_values,
                int input_count,
                OrtValue* const* output_values,
                int output_count) {
  std::lock_guard<std::mutex> guard(GetMutex());
  StandAloneNodes nodes = GetNodes();
  auto iter = nodes.find(kernel);
  if (iter != nodes.end()) {
    Node* node = iter->second.get();
    auto& input_defs = node->MutableInputDefs();
    auto& output_defs = node->MutableOutputDefs();
    if (input_defs.size() == static_cast<size_t>(input_count) &&
        output_defs.size() == static_cast<size_t>(output_count)) {
      return;  // already done init
    }
    for (auto* input_arg : input_defs) {
      delete input_arg;
    }
    for (auto* output_arg : output_defs) {
      delete output_arg;
    }
    input_defs.clear();
    output_defs.clear();
    for (int i = 0; i < input_count; ++i) {
      input_defs.push_back(new onnxruntime::NodeArg(std::to_string(i), input_values[i]->Type()->GetTypeProto()));
    }
    for (int i = 0; i < output_count; ++i) {
      output_defs.push_back(new onnxruntime::NodeArg(std::to_string(i), output_values[i]->Type()->GetTypeProto()));
    }
  }
}

// For invoking kernels without a graph
class StandAloneKernelContext : public OpKernelContext {
 public:
  StandAloneKernelContext(const OrtValue* const* input_values,
                          int input_count,
                          OrtValue* const* output_values,
                          int output_count,
                          AllocatorPtr allocator,
                          onnxruntime::concurrency::ThreadPool* threadpool,
                          const logging::Logger& logger) : OpKernelContext(threadpool, logger),
                                                           input_values_(input_values),
                                                           input_count_(input_count),
                                                           output_values_(output_values),
                                                           output_count_(output_count),
                                                           allocator_(allocator) {}

  int NumVariadicInputs(size_t arg_num) const override {
    ORT_ENFORCE(arg_num < static_cast<size_t>(input_count_), "invalid arg_num.");
    auto ort_value = input_values_[arg_num];
    if (ort_value->IsTensor()) {
      return static_cast<int>(ort_value->Get<Tensor>().Shape().Size());
    } else if (ort_value->IsTensorSequence()) {
      return static_cast<int>(ort_value->Get<TensorSeq>().Size());
    } else if (ort_value->IsSparseTensor()) {
#ifdef DISABLE_SPARSE_TENSORS
      ORT_THROW("sparse tensor is not supported in this build.");
#else
      return static_cast<int>(ort_value->Get<SparseTensor>().Values().Shape().Size());
#endif
    } else {
      return 0;
    }
  }

  MLDataType InputType(int index) const override {
    if (index >= input_count_) {
      return nullptr;
    } else {
      return input_values_[index]->Type();
    }
  }

  MLDataType OutputType(int index) const override {
    if (index >= output_count_) {
      return nullptr;
    } else {
      return output_values_[index]->Type();
    }
  }

  bool TryGetInferredInputShape(int, TensorShape&) const override {
    return false;
  }

  bool TryGetInferredOutputShape(int, TensorShape&) const override {
    return false;
  }

  int InputCount() const override {
    return input_count_;
  }

  int ImplicitInputCount() const override {
    return 0;
  }

  int OutputCount() const override {
    return static_cast<int>(output_count_);
  }

  Status GetTempSpaceAllocator(AllocatorPtr* output) const override ORT_MUST_USE_RESULT {
    *output = allocator_;
    return Status::OK();
  }

  Fence_t InputFence(int index) const override {
    if (index >= input_count_) {
      return nullptr;
    } else {
      return input_values_[index]->Fence();
    }
  }

  Fence_t ImplicitInputFence(int) const override {
    return nullptr;
  }

  Fence_t OutputFence(int index) const override {
    if (index >= output_count_) {
      return nullptr;
    } else {
      return output_values_[index]->Fence();
    }
  }

  int GetDeviceId() const override {
    return 0;
  }

  void* GetComputeStream() const override {
    return nullptr;
  }

 protected:
  const OrtValue* GetInputMLValue(int index) const override {
    if (index >= input_count_) {
      return nullptr;
    } else {
      return input_values_[index];
    }
  }

  OrtValue* OutputMLValue(int index, const TensorShape& shape) override {
    if (index >= output_count_) {
      return nullptr;
    }
    OrtValue& ort_value = *output_values_[index];
    if (!ort_value.IsAllocated()) {
      if (ort_value.IsTensor()) {
        Tensor::InitOrtValue(ort_value.Type(), shape, allocator_, ort_value);
      } else if (ort_value.IsTensorSequence()) {
        auto ml_type = ort_value.Type();
        auto element_type = ml_type->AsSequenceTensorType()->GetElementType();
        auto p_sequence = std::make_unique<TensorSeq>(element_type);
        auto ml_tensor_sequence = DataTypeImpl::GetType<TensorSeq>();
        ort_value.Init(p_sequence.release(), ml_tensor_sequence, ml_tensor_sequence->GetDeleteFunc());
      } else if (ort_value.IsSparseTensor()) {
#ifdef DISABLE_SPARSE_TENSORS
        ORT_THROW("sparse tensor is not supported in this build.");
#else
        auto ml_type = ort_value.Type();
        auto element_type = ml_type->AsSparseTensorType()->GetElementType();
        SparseTensor::InitOrtValue(element_type, shape, allocator_, ort_value);
#endif
      }
    }
    return &ort_value;
  }

  OrtValue* GetOrCreateOutputMLValue(int index) override {
    if (index >= output_count_) {
      return nullptr;
    } else {
      return output_values_[index];
    }
  }

  const OrtValue* const* input_values_;
  const int input_count_;
  OrtValue* const* output_values_;
  const int output_count_;
  AllocatorPtr allocator_;
};  // StandAloneKernelContext

onnxruntime::Status CreateOpAttr(const char* name, const void* data, int len, OrtOpAttrType type, OrtOpAttr** op_attr) {
  auto attr = std::make_unique<ONNX_NAMESPACE::AttributeProto>();
  onnxruntime::Status status = onnxruntime::Status::OK();
  attr->set_name(std::string{name});
  const int* ints = reinterpret_cast<const int*>(data);
  const float* floats = reinterpret_cast<const float*>(data);
  auto str = reinterpret_cast<const char*>(data);
  auto strs = reinterpret_cast<const char* const*>(data);
  switch (type) {
    case OrtOpAttrType::ORT_OP_ATTR_INT:
      attr->set_i(ints[0]);
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
      break;
    case OrtOpAttrType::ORT_OP_ATTR_INTS:
      for (int j = 0; j < len; ++j) {
        attr->add_ints(ints[j]);
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
      break;
    case OrtOpAttrType::ORT_OP_ATTR_FLOAT:
      attr->set_f(floats[0]);
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
      break;
    case OrtOpAttrType::ORT_OP_ATTR_FLOATS:
      for (int j = 0; j < len; ++j) {
        attr->add_floats(floats[j]);
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);
      break;
    case OrtOpAttrType::ORT_OP_ATTR_STRING:
      attr->set_s(std::string{str});
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
      break;
    case OrtOpAttrType::ORT_OP_ATTR_STRINGS:
      for (int j = 0; j < len; ++j) {
        attr->add_strings(std::string{strs[j]});
      }
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS);
      break;
    default:
      status = Status(common::ONNXRUNTIME, common::FAIL, "Attribute type not supported yet.");
      break;
  }
  *op_attr = reinterpret_cast<OrtOpAttr*>(attr.release());
  return status;
}

onnxruntime::Status CreateOp(const OrtExecutionProvider* ort_ep,
                             const char* op_name,
                             const char* domain,
                             int version,
                             const char** type_constraint_names,
                             const ONNXTensorElementDataType* type_constraint_values,
                             int type_constraint_count,
                             const OrtOpAttr* const* attr_values,
                             int attr_count,
                             OrtOp** op) {
  *op = nullptr;
  auto ep = reinterpret_cast<const IExecutionProvider*>(ort_ep);
  auto kernel_registry = ep->GetKernelRegistry();
  const KernelCreateInfo* kernel_create_info{};
  std::unordered_map<std::string, MLDataType> type_constraint_map;
  for (int i = 0; i < type_constraint_count; ++i) {
    ONNX_NAMESPACE::TypeProto proto;
    proto.mutable_tensor_type()->set_elem_type(type_constraint_values[i]);
    type_constraint_map[type_constraint_names[i]] = DataTypeImpl::TypeFromProto(proto);
  }
  auto status = kernel_registry->TryFindKernel(op_name,
                                               domain,
                                               version,
                                               type_constraint_map,
                                               ep->Type(),
                                               &kernel_create_info);
  ORT_RETURN_IF_ERROR(status);
  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;
  auto node = std::make_unique<onnxruntime::Node>(std::string("standalone_") + op_name, op_name, "", input_args, output_args, nullptr, domain);
  for (int i = 0; i < attr_count; ++i) {
    auto attr_proto = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attr_values[i]);
    node->AddAttributeProto(*attr_proto);
  }

  auto kernel_def_builder = KernelDefBuilder::Create();
  kernel_def_builder->SetName(op_name);
  kernel_def_builder->SetDomain(domain);
  kernel_def_builder->SinceVersion(version);
  OpKernelInfo instant_kernel_info(*node.get(), *kernel_def_builder->Build(), *ep, {}, {}, {});
  std::unique_ptr<onnxruntime::OpKernel> op_kernel;
  FuncManager func_mgr;
  status = kernel_create_info->kernel_create_func(func_mgr, instant_kernel_info, op_kernel);
  ORT_RETURN_IF_ERROR(status);
  AddNode(op_kernel.get(), NodeHolder(node.release()));
  *op = reinterpret_cast<OrtOp*>(op_kernel.release());
  return status;
}

onnxruntime::Status InvokeOp(_In_ const OrtKernelContext* context,
                             _In_ const OrtOp* ort_op,
                             _In_ const OrtValue* const* input_values,
                             _In_ int input_count,
                             _Inout_ OrtValue* const* output_values,
                             _In_ int output_count) {
  auto ctx = reinterpret_cast<const OpKernelContext*>(context);
  AllocatorPtr allocator{};
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  UpdateNode(reinterpret_cast<const onnxruntime::OpKernel*>(ort_op), input_values, input_count, output_values, output_count);
  StandAloneKernelContext standalone_kernel_ctx(input_values,
                                                input_count,
                                                output_values,
                                                output_count,
                                                allocator,
                                                ctx->GetOperatorThreadPool(),
                                                ctx->Logger());
  auto kernel = reinterpret_cast<const OpKernel*>(ort_op);
  return kernel->Compute(&standalone_kernel_ctx);
}

}  // namespace standalone
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtApis::CreateOpAttr,
                    _In_ const char* name,
                    _In_ const void* data,
                    _In_ int len,
                    _In_ OrtOpAttrType type,
                    _Outptr_ OrtOpAttr** op_attr) {
  API_IMPL_BEGIN
  auto status = onnxruntime::standalone::CreateOpAttr(name, data, len, type, op_attr);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
  }
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseOpAttr, _Frees_ptr_opt_ OrtOpAttr* op_attr) {
  if (op_attr) {
    delete reinterpret_cast<ONNX_NAMESPACE::AttributeProto*>(op_attr);
  }
}

ORT_API_STATUS_IMPL(OrtApis::CreateOp,
                    _In_ const OrtExecutionProvider* ep,
                    _In_ const char* op_name,
                    _In_ const char* domain,
                    int version,
                    _In_opt_ const char** type_constraint_names,
                    _In_opt_ const ONNXTensorElementDataType* type_constraint_values,
                    int type_constraint_count,
                    _In_opt_ const OrtOpAttr* const* attr_values,
                    int attr_count,
                    _Outptr_ OrtOp** ort_op) {
  API_IMPL_BEGIN
  auto status = onnxruntime::standalone::CreateOp(ep,
                                                  op_name,
                                                  domain,
                                                  version,
                                                  type_constraint_names,
                                                  type_constraint_values,
                                                  type_constraint_count,
                                                  attr_values,
                                                  attr_count,
                                                  ort_op);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
  }
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::InvokeOp,
                    _In_ const OrtKernelContext* context,
                    _In_ const OrtOp* ort_op,
                    _In_ const OrtValue* const* input_values,
                    _In_ int input_count,
                    _Inout_ OrtValue* const* output_values,
                    _In_ int output_count) {
  API_IMPL_BEGIN
  auto status = onnxruntime::standalone::InvokeOp(context, ort_op, input_values, input_count, output_values, output_count);
  if (status.IsOK()) {
    return nullptr;
  } else {
    return CreateStatus(static_cast<OrtErrorCode>(status.Code()), status.ErrorMessage().c_str());
  }
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseOp, _Frees_ptr_opt_ OrtOp* op) {
  if (op) {
    onnxruntime::OpKernel* kernel = reinterpret_cast<onnxruntime::OpKernel*>(op);
    onnxruntime::standalone::DelNode(kernel);
    delete kernel;
  }
}

ORT_API_STATUS_IMPL(OrtApis::GetExecutionProvider,
      _In_ const OrtKernelInfo* info,
      _Outptr_ OrtExecutionProvider** ep) {
  API_IMPL_BEGIN
   auto kernel_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
   auto ort_ep = const_cast<onnxruntime::IExecutionProvider*>(kernel_info->GetExecutionProvider());
  *ep = reinterpret_cast<OrtExecutionProvider*>(ort_ep);
  return nullptr;
  API_IMPL_END
}

#endif
