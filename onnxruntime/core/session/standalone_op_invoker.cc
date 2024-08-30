// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/TensorSeq.h"
#include "core/session/ort_apis.h"
#include <unordered_map>

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/schema_registry.h"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
// disabling warning on calling of raw "delete" operator
#pragma warning(disable : 26400)
#endif

#if defined(ORT_MINIMAL_BUILD) && !defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

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
                    _In_opt_ const char**,
                    _In_opt_ const ONNXTensorElementDataType*,
                    _In_opt_ int,
                    _In_opt_ const OrtOpAttr* const*,
                    _In_opt_ int,
                    _In_ int,
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

ORT_API_STATUS_IMPL(OrtApis::CopyKernelInfo,
                    _In_ const OrtKernelInfo*,
                    _Outptr_ OrtKernelInfo**) {
  API_IMPL_BEGIN
  return CreateStatus(ORT_NOT_IMPLEMENTED, "CopyKernelInfo is not implemented for minimal build.");
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseKernelInfo, _Frees_ptr_opt_ OrtKernelInfo*) {
}

#else

namespace onnxruntime {
namespace standalone {

using NodePtr = std::unique_ptr<onnxruntime::Node>;
using ArgPtr = std::unique_ptr<onnxruntime::NodeArg>;
using ArgPtrs = onnxruntime::InlinedVector<ArgPtr>;

using NodeResource = std::pair<NodePtr, ArgPtrs>;
using NodeResourceMap = InlinedHashMap<const onnxruntime::OpKernel*, NodeResource>;

class NodeRepo {
 public:
  static NodeRepo& GetInstance() {
    static NodeRepo node_repo;
    return node_repo;
  }

  // create the kernel using the FuncManager NodeRepo owns for consistency
  onnxruntime::Status CreateKernel(const KernelCreateInfo& kernel_create_info,
                                   const OpKernelInfo& kernel_info,
                                   std::unique_ptr<OpKernel>& op_kernel) {
    std::lock_guard<std::mutex> guard(mutex_);
    return kernel_create_info.kernel_create_func(func_mgr_, kernel_info, op_kernel);
  }

  onnxruntime::Status AddNode(const onnxruntime::OpKernel* kernel, NodePtr&& node_ptr, ArgPtrs&& args) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto ret = resource_map_.try_emplace(kernel, NodeResource{std::move(node_ptr), std::move(args)});
    if (!ret.second) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "kernel already mapped to existing node");
    }

    return Status::OK();
  }

#if !defined(ORT_MINIMAL_BUILD)
  common::Status RegisterCustomOpNodeSchemas(KernelTypeStrResolver& kernel_type_str_resolver, Graph& graph) {
    std::lock_guard<std::mutex> guard(mutex_);

    for (auto cur = resource_map_.begin(), end = resource_map_.end(); cur != end; ++cur) {
      // Lookup the schema for the operator so we include it in the ORT format model and can match the kernel
      // in a minimal build.
      // The opset version will not necessarily match the model, so we need to call GetSchema directly to plug that in.
      // In theory this should never fail if the kernel lookup earlier was successful.
      const Node& node = *cur->second.first;
      auto* schema = graph.GetSchemaRegistry()->GetSchema(node.OpType(), node.SinceVersion(), node.Domain());

      ORT_RETURN_IF_NOT(schema, "Unable to find schema for node. Domain:'", node.Domain(),
                        "' op_type:", node.OpType());
      ORT_RETURN_IF_ERROR(kernel_type_str_resolver.RegisterOpSchema(*schema));
    }

    return Status::OK();
  }
#endif

  onnxruntime::Status ValidateInputOutputCounts(const onnxruntime::OpKernel* kernel,
                                                int input_count,
                                                int output_count) {
    size_t expect_input_count{};
    size_t expect_output_count{};
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto iter = resource_map_.find(kernel);
      if (iter == resource_map_.end()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "matching node is missing");
      }
      auto* node = iter->second.first.get();
      expect_input_count = node->InputDefs().size();
      expect_output_count = node->OutputDefs().size();
    }
    if (expect_input_count != static_cast<size_t>(input_count)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "invalid node input count: ", input_count,
                             ", expect: ", expect_input_count);
    }
    if (expect_output_count != static_cast<size_t>(output_count)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "invalid node output count", output_count,
                             ", expect: ", expect_output_count);
    }
    return Status::OK();
  }

  void RemoveNode(const onnxruntime::OpKernel* kernel) {
    std::lock_guard<std::mutex> guard(mutex_);
    resource_map_.erase(kernel);
  }

 private:
  explicit NodeRepo() = default;
  ~NodeRepo() = default;

  std::mutex mutex_;
  NodeResourceMap resource_map_;
  FuncManager func_mgr_;
};

#if !defined(ORT_MINIMAL_BUILD)
common::Status RegisterCustomOpNodeSchemas(KernelTypeStrResolver& kernel_type_str_resolver, Graph& graph) {
  return NodeRepo::GetInstance().RegisterCustomOpNodeSchemas(kernel_type_str_resolver, graph);
}
#endif

// For invoking kernels without a graph
class StandAloneKernelContext : public OpKernelContext {
 public:
  StandAloneKernelContext(const OrtValue* const* input_values,
                          int input_count,
                          OrtValue* const* output_values,
                          int output_count,
                          AllocatorPtr allocator,
                          onnxruntime::concurrency::ThreadPool* threadpool,
                          const logging::Logger& logger,
                          Stream* stream) : OpKernelContext(threadpool, logger, stream),
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

  Status GetTempSpaceAllocator(AllocatorPtr* output) const override {
    *output = allocator_;
    return Status::OK();
  }

  int GetDeviceId() const override {
    return 0;
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
  const int64_t* ints = reinterpret_cast<const int64_t*>(data);
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

onnxruntime::Status CreateOp(_In_ const OrtKernelInfo* info,
                             _In_z_ const char* op_name,
                             _In_z_ const char* domain,
                             int version,
                             _In_reads_(type_constraint_count) const char** type_constraint_names,
                             _In_reads_(type_constraint_count) const ONNXTensorElementDataType* type_constraint_values,
                             int type_constraint_count,
                             _In_reads_(attr_count) const OrtOpAttr* const* attr_values,
                             int attr_count,
                             int input_count,
                             int output_count,
                             OrtOp** op) {
  *op = nullptr;
  auto kernel_info = reinterpret_cast<const OpKernelInfo*>(info);
  auto ep = reinterpret_cast<const IExecutionProvider*>(kernel_info->GetExecutionProvider());
  auto kernel_registry = ep->GetKernelRegistry();
  const KernelCreateInfo* kernel_create_info{};
  InlinedHashMap<std::string, MLDataType> type_constraint_map;

  for (int i = 0; i < type_constraint_count; ++i) {
    ONNX_NAMESPACE::TypeProto proto;
    proto.mutable_tensor_type()->set_elem_type(type_constraint_values[i]);
    type_constraint_map[type_constraint_names[i]] = DataTypeImpl::TypeFromProto(proto);
  }

  ArgPtrs arg_ptrs;
  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;

  for (int i = 0; i < input_count; ++i) {
    arg_ptrs.push_back(std::make_unique<NodeArg>(std::to_string(i), nullptr));
    input_args.push_back(arg_ptrs.back().get());
  }

  for (int i = 0; i < output_count; ++i) {
    arg_ptrs.push_back(std::make_unique<NodeArg>(std::to_string(i), nullptr));
    output_args.push_back(arg_ptrs.back().get());
  }

  NodePtr node_ptr = std::make_unique<onnxruntime::Node>(std::string("standalone_") + op_name, op_name, "",
                                                         input_args, output_args, nullptr, domain);

  for (int i = 0; i < attr_count; ++i) {
    auto attr_proto = reinterpret_cast<const ONNX_NAMESPACE::AttributeProto*>(attr_values[i]);
    node_ptr->AddAttributeProto(*attr_proto);
  }

  node_ptr->SetSinceVersion(version);

  auto status = kernel_registry->TryFindKernel(*node_ptr, ep->Type(), type_constraint_map, &kernel_create_info);
  ORT_RETURN_IF_ERROR(status);

  auto& kernel_def = kernel_create_info->kernel_def;
  ORT_RETURN_IF_NOT(kernel_def, "Kernel definition was not found for node Domain:'",
                    node_ptr->Domain(), "' op_type:", node_ptr->OpType());

  static const std::unordered_map<int, OrtValue> kEmptyValueMap;
  static const OrtValueNameIdxMap kEmptyNameMap;

  OpKernelInfo tmp_kernel_info(*node_ptr.get(), *kernel_def, *ep, kEmptyValueMap, kEmptyNameMap,
                               kernel_info->GetDataTransferManager(),
                               kernel_info->GetAllocators(),
                               kernel_info->GetConfigOptions());

  std::unique_ptr<onnxruntime::OpKernel> op_kernel;

  auto& node_repo = NodeRepo::GetInstance();
  ORT_RETURN_IF_ERROR(node_repo.CreateKernel(*kernel_create_info, tmp_kernel_info, op_kernel));
  ORT_RETURN_IF_ERROR(node_repo.AddNode(op_kernel.get(), std::move(node_ptr), std::move(arg_ptrs)));

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
  auto kernel = reinterpret_cast<const OpKernel*>(ort_op);
  ORT_RETURN_IF_ERROR(NodeRepo::GetInstance().ValidateInputOutputCounts(kernel, input_count, output_count));
  StandAloneKernelContext standalone_kernel_ctx(input_values,
                                                input_count,
                                                output_values,
                                                output_count,
                                                allocator,
                                                ctx->GetOperatorThreadPool(),
                                                ctx->Logger(),
                                                ctx->GetComputeStream());
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
                    _In_ const OrtKernelInfo* info,
                    _In_z_ const char* op_name,
                    _In_z_ const char* domain,
                    int version,
                    _In_reads_(type_constraint_count) const char** type_constraint_names,
                    _In_reads_(type_constraint_count) const ONNXTensorElementDataType* type_constraint_values,
                    int type_constraint_count,
                    _In_reads_(attr_count) const OrtOpAttr* const* attr_values,
                    int attr_count,
                    int input_count,
                    int output_count,
                    _Outptr_ OrtOp** ort_op) {
  API_IMPL_BEGIN
  auto status = onnxruntime::standalone::CreateOp(info,
                                                  op_name,
                                                  domain,
                                                  version,
                                                  type_constraint_names,
                                                  type_constraint_values,
                                                  type_constraint_count,
                                                  attr_values,
                                                  attr_count,
                                                  input_count,
                                                  output_count,
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
    auto kernel = reinterpret_cast<onnxruntime::OpKernel*>(op);
    onnxruntime::standalone::NodeRepo::GetInstance().RemoveNode(kernel);
    delete kernel;
  }
}

ORT_API_STATUS_IMPL(OrtApis::CopyKernelInfo, _In_ const OrtKernelInfo* info, _Outptr_ OrtKernelInfo** info_copy) {
  API_IMPL_BEGIN
  auto kernel_info = reinterpret_cast<const onnxruntime::OpKernelInfo*>(info);
  auto tmp_info_holder = std::make_unique<onnxruntime::OpKernelInfo>(*kernel_info);
  *info_copy = reinterpret_cast<OrtKernelInfo*>(tmp_info_holder.release());
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseKernelInfo, _Frees_ptr_opt_ OrtKernelInfo* info_copy) {
  if (info_copy) {
    auto kernel_info = reinterpret_cast<onnxruntime::OpKernelInfo*>(info_copy);
    GSL_SUPPRESS(r.11)
    delete kernel_info;
  }
}

#endif
