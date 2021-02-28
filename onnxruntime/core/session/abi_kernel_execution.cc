// Licensed under the MIT License.

#include <iostream>
#include <core/graph/model.h>
#include <core/framework/mldata_type_utils.h>
#include "core/graph/onnx_protobuf.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/session/abi_kernel_execution.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/framework/customregistry.h"
#include "core/framework/data_types.h"
#include "core/framework/execution_providers.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/framework/tensorprotoutils.h"
#include "abi_session_options_impl.h"

using namespace onnxruntime;

/*
 * ORT API Implementations
 */
ORT_API_STATUS_IMPL(OrtApis::CreateKernelSession,
                    _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtKernelSession **session_,
                    int opset_version) {
  API_IMPL_BEGIN
    ORT_ENFORCE(session_, "OrtKernelSession pointer must not be null");
    std::unordered_map<std::string, int> domain_to_version{{"", opset_version}};


    KernelSessionImpl *session = new KernelSessionImpl();
    std::unique_ptr<Model> model = onnxruntime::make_unique<Model>("KernelExecutionModel",
                                       /*is_onnx_domain_only=*/true,
                                       /*model_metadata=*/ModelMetaData(),
                                       /*model_path=*/PathString(),
                                       /*local_registries=*/IOnnxRuntimeOpSchemaRegistryList(),
                                       /*domain_to_version=*/domain_to_version,
                                       /*model_functions=*/std::initializer_list<ONNX_NAMESPACE::FunctionProto>{},
                                       /*logger=*/session->logger_);
    session->model = std::move(model);


    // initialize the providers
    for(auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      provider->RegisterAllocator(session->allocator_mgr_);

      auto data_xfr = provider->GetDataTransfer();
      if (data_xfr) {
        auto st = session->data_transfer_mgr_.RegisterDataTransfer(std::move(data_xfr));
        if (!st.IsOK()) {
          return ToOrtStatus(st);
        }
      }
      session->provider_list.push_back(std::move(provider));
    }


    // Create the session state. We need only this because some CUDA ops static_cast
    // the OpKernelContext to OpKernelContextInternal. That could be decoupled better
    session->session_state_ = onnxruntime::make_unique<SessionState>(
                session->model->MainGraph(),
                ExecutionProviders(), // empty for now, doesn't seem to be used
                false,
                nullptr,
                nullptr,
                session->data_transfer_mgr_,
                session->logger_,
                session->profiler_,
                false);

    *session_ = reinterpret_cast<OrtKernelSession *>(session);

    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateExecutableKernelContext,
                    _In_ const char* name,
                    _In_ const char* op_type,
                    _Outptr_ OrtExecutableKernelContext **kernel_context_) {
  API_IMPL_BEGIN

    ORT_ENFORCE(kernel_context_, "OrtExecutableKernelContext pointer must be non-null.");

    ExecutableKernelContextImpl* kernel_context = new ExecutableKernelContextImpl();
    kernel_context->SetName(name);
    kernel_context->SetOpType(op_type);
    *kernel_context_ = reinterpret_cast<OrtExecutableKernelContext *>(kernel_context);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddInput,
                    _Inout_ OrtExecutableKernelContext* context_,
                    ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
    context->AddInput(type);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddOutput,
                    _Inout_ OrtExecutableKernelContext* context_,
                    ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
    context->AddOutput(type);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateExecutableKernel,
                    _Inout_ OrtKernelSession *session_,
                    _In_ OrtExecutableKernelContext *context_,
                    size_t provider_id,
                    _Outptr_ OrtExecutableKernel **kernel_) {
  API_IMPL_BEGIN

    ORT_ENFORCE(session_, "OrtKernelSession pointer must be non-null.");
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ORT_ENFORCE(kernel_, "OrtExecutableKernel pointer must be non-null.");

    KernelSessionImpl *session = reinterpret_cast<KernelSessionImpl *>(session_);
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);
    ORT_ENFORCE(provider_id < session->provider_list.size(),
                "provider_id (" + std::to_string(provider_id) + ")must be less than the provider list size (" + std::to_string(session->provider_list.size()) + ").");

    SingleKernelExecutionFrame* frame;
    Status status = context->CreateExecutionFrame(session, &frame, provider_id);
    if (!status.IsOK()){
      return ToOrtStatus(status);
    }
    *kernel_ = reinterpret_cast<OrtExecutableKernel*>(frame);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeString,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    _In_ const char* value) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;

    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    attribute_proto.set_s(value);

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeStrings,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    _In_ const char** values,
                    size_t num_values) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;
    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::STRINGS);

    for (size_t i = 0; i < num_values; i++) {
      attribute_proto.add_strings(values[i]);
    }

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeFloat,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    float value) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;

    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
    attribute_proto.set_f(value);

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeFloats,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    float* values,
                    size_t num_values) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;
    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::FLOATS);

    for (size_t i = 0; i < num_values; i++) {
      attribute_proto.add_floats(values[i]);
    }

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeInt,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    int64_t value) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;

    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::INT);
    attribute_proto.set_i(value);

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeInts,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    int64_t* values,
                    size_t num_values) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;
    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::INTS);

    for (size_t i = 0; i < num_values; i++) {
      attribute_proto.add_ints(values[i]);
    }

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernelContext_AddAttributeTensor,
                    _Inout_ OrtExecutableKernelContext* context_,
                    _In_ const char* name,
                    _In_ void* p_data,
                    size_t p_data_len,
                    _In_ const int64_t* shape,
                    size_t shape_len,
                    ONNXTensorElementDataType type) {
  API_IMPL_BEGIN
    ORT_ENFORCE(context_, "OrtExecutableKernelContext pointer must be non-null.");
    ExecutableKernelContextImpl *context = reinterpret_cast<ExecutableKernelContextImpl *>(context_);

    ONNX_NAMESPACE::AttributeProto attribute_proto;

    attribute_proto.set_name(name);
    attribute_proto.set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
    ONNX_NAMESPACE::TensorProto* t = attribute_proto.mutable_t();

    switch (type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::FLOAT);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT8);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::INT8);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::INT16);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::INT32);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT32);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::INT64);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::UINT64);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::BOOL);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        t->set_data_type(ONNX_NAMESPACE::TensorProto::DOUBLE);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      default: {
        std::ostringstream oss;
        oss << "type " << type << " is not supported in this function";
        std::string errmsg = oss.str();
        return ToOrtStatus(Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, errmsg));
      }
    }

    for(size_t i = 0; i < shape_len; i++) {
      t->add_dims(shape[i]);
    }

    for(size_t i = 0; i < p_data_len; i++) {
      switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
          t->add_float_data(static_cast<float*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
          t->add_int32_data(static_cast<uint8_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
          t->add_int32_data(static_cast<int8_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
          t->add_int32_data(static_cast<uint16_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
          t->add_int32_data(static_cast<bool*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
          t->add_int32_data(static_cast<int16_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
          t->add_int32_data(static_cast<int32_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
          t->add_int64_data(static_cast<int64_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
          t->add_uint64_data(static_cast<uint32_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
          t->add_uint64_data(static_cast<uint64_t*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
          t->add_double_data(static_cast<double*>(p_data)[i]);
          break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
        default: {
          std::ostringstream oss;
          oss << "type " << type << " is not supported in this function";
          std::string errmsg = oss.str();
          return ToOrtStatus(Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, errmsg));
        }
      }
    }

    return ToOrtStatus(context->AddAttribute(name, attribute_proto));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_SetInput,
                    _Inout_ OrtExecutableKernel *kernel_,
                    int index,
                    _In_ OrtValue *value) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* kernel = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    return ToOrtStatus(kernel->SetInput(*value, index));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_SetOutput,
                    OrtExecutableKernel *kernel_,
                    int index,
                    OrtValue *value) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* kernel = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    return ToOrtStatus(kernel->SetOutput(*value, index));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_Compute,
                    _Inout_ OrtExecutableKernel *kernel_) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* kernel = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    return ToOrtStatus(kernel->Compute());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_IsOutputOnCpu,
                    _Inout_ OrtExecutableKernel *kernel_, int index, _Out_ int* is_output_on_cpu) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* kernel = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    *is_output_on_cpu = kernel->IsOutputOnCpu(index);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_IsInputOnCpu,
                    _Inout_ OrtExecutableKernel *kernel_, int index, _Out_ int* is_input_on_cpu) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* kernel = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    *is_input_on_cpu = kernel->IsInputOnCpu(index);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseKernelSession, _Frees_ptr_opt_ OrtKernelSession* value) {
  delete reinterpret_cast<KernelSessionImpl*>(value);
}

ORT_API(void, OrtApis::ReleaseExecutableKernel, _Frees_ptr_opt_ OrtExecutableKernel* value) {
  delete reinterpret_cast<SingleKernelExecutionFrame*>(value);
}

ORT_API(void, OrtApis::ReleaseExecutableKernelContext, _Frees_ptr_opt_ OrtExecutableKernelContext * value) {
  delete reinterpret_cast<ExecutableKernelContextImpl*>(value);
}

SingleKernelExecutionFrame::Info::Info(std::unique_ptr<OpKernel> kernel,
                                       const logging::Logger& logger,
                                       const std::unique_ptr<IExecutionProvider>& provider,
                                       const std::unique_ptr<SessionState>& session_state)
    : kernel_(std::move(kernel)),
      logger_(&logger),
      session_state_(session_state),
      provider_(provider)
{
  ORT_ENFORCE(kernel_, "kernel cannot be null");
  ORT_ENFORCE(provider_, "provider cannot be null");

  allocator_ = provider_->GetAllocator(provider_->GetDeviceId(), OrtMemTypeDefault);

  auto &node = kernel_->Node();

  input_index_to_mlvalue_map_ = std::vector<int> (node.InputDefs().size(), -1);
  output_index_to_mlvalue_map_ = std::vector<int> (node.OutputDefs().size(), -1);

  if (node.ImplicitInputDefs().size()) {
    // not sure how to handle this correctly
    throw new NotImplementedException("Implicit inputs are not supported");
  }

  // initialize inputs and outputs with null values
  OrtValue null_value;
  node.ForEachWithIndex(node.InputDefs(),
                        [this](const NodeArg &arg, size_t index) {
                          this->AddInput(OrtValue(), index, arg.Name());
                          return Status::OK();
                        });

  node.ForEachWithIndex(node.OutputDefs(),
                        [this](const NodeArg &arg, size_t index) {
                          this->AddOutput(OrtValue(), index, arg.Name());
                          return Status::OK();
                        });
}

Status SingleKernelExecutionFrame::Info::AddOutput(OrtValue value, size_t index, const std::string& name) {
  int mlvalue_idx = value_name_idx_map_.Add(name);
  ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetOutputType(index);

  output_index_to_mlvalue_map_[index] = mlvalue_idx;
  fetches_.push_back(value);
  fetches_mlvalue_idxs_.push_back(mlvalue_idx);
  return Status::OK();
}

Status SingleKernelExecutionFrame::Info::AddInput(OrtValue value, size_t index, const std::string& name) {
  int mlvalue_idx = value_name_idx_map_.Add(name);

  input_index_to_mlvalue_map_[index] = mlvalue_idx;
  ort_value_idx_nodearg_map_[mlvalue_idx] = kernel_->Info().GetInputType(index);
  feeds_.push_back(value);
  feed_mlvalue_idxs_.push_back(mlvalue_idx);
  return Status::OK();
}

Status ExecutableKernelContextImpl::AddInput(ONNXTensorElementDataType type) {
  std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = onnxruntime::make_unique<ONNX_NAMESPACE::TypeProto>();

  Status status = SetupTensorType(type_proto, type);
  if (!status.IsOK()) {
    return status;
  }

  std::ostringstream oss;
  oss << name_ << "_Input_" << input_args_.size();
  std::string name = oss.str();

  std::unique_ptr<NodeArg> arg_ptr = onnxruntime::make_unique<NodeArg>(name, type_proto.get());
  input_args_.push_back(arg_ptr.get());
  types_.push_back(std::move(type_proto));
  args_.push_back(std::move(arg_ptr));
  return Status::OK();
}

Status ExecutableKernelContextImpl::AddOutput(ONNXTensorElementDataType type) {

  std::unique_ptr<ONNX_NAMESPACE::TypeProto> type_proto = onnxruntime::make_unique<ONNX_NAMESPACE::TypeProto>();

  Status status = SetupTensorType(type_proto, type);
  if (!status.IsOK()) {
    return status;
  }

  std::ostringstream oss;
  oss << name_ << "_Output_" << output_args_.size();
  std::string name = oss.str();

  std::unique_ptr<NodeArg> arg_ptr = onnxruntime::make_unique<NodeArg>(name, type_proto.get());

  output_args_.push_back(arg_ptr.get());
  types_.push_back(std::move(type_proto));
  args_.push_back(std::move(arg_ptr));
  return Status::OK();
}

Status ExecutableKernelContextImpl::CreateExecutionFrame(KernelSessionImpl* session, SingleKernelExecutionFrame** frame, size_t provider_id) {
  auto& graph = session->model->MainGraph();

  std::string description;

  Node& node = graph.AddNode(
      name_,
      op_type_,
      description,
      input_args_,
      output_args_,
      &attributes_);
  Status status = graph.Resolve();
  if (!status.IsOK()){
    return status;
  }
  if (node.Op() == nullptr) {
    std::string message("Unable to resolve node op. This may happen when the node has no outputs.");
    return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, message);
  }

  auto const& execution_provider = session->provider_list[provider_id];

  node.SetExecutionProviderType(execution_provider->Type());

  std::shared_ptr<KernelRegistry> registry = execution_provider->GetKernelRegistry();
  std::unique_ptr<OpKernel> op_kernel;
  status = registry->TryCreateKernel(node,
                                     *execution_provider,
                                     std::unordered_map<int, OrtValue>(),
                                     OrtValueNameIdxMap(),
                                     FuncManager(),
                                     session->data_transfer_mgr_,
                                     op_kernel);

  if (!status.IsOK()) {
    return status;
  }


  // create the context info
  std::unique_ptr<SingleKernelExecutionFrame::Info> info = onnxruntime::make_unique<SingleKernelExecutionFrame::Info>(
              std::move(op_kernel),
              session->logger_,
              execution_provider,
              session->session_state_);

  *frame = new SingleKernelExecutionFrame(std::move(info));
  return Status::OK();
}

Status ExecutableKernelContextImpl::SetupTensorType(const std::unique_ptr<ONNX_NAMESPACE::TypeProto>& type_proto, ONNXTensorElementDataType type) {
  ONNX_NAMESPACE::TypeProto::Tensor *tensor_type = type_proto->mutable_tensor_type();

  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT8);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT8);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT32);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT32);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::UINT64);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::STRING);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::BOOL);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::BFLOAT16);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::DOUBLE);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::COMPLEX64);
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
      tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto::COMPLEX128);
      break;
    default: {
      std::ostringstream oss;
      oss << "type " << type << " is not supported in this function";
      std::string errmsg = oss.str();
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED, errmsg);
    }
  }
  return Status::OK();
}

onnxruntime::common::Status SingleKernelExecutionFrame::Compute() {
    // We need to use OpKernelContextInternal here because some CUDA ops static_cast the context to it...
    OpKernelContextInternal context(*info_->session_state_, *this, *info_->kernel_, *info_->logger_, false);
    return info_->kernel_->Compute(&context);
}

// taken from OptimizerExecutionFrame
Status
SingleKernelExecutionFrame::CreateNodeOutputMLValueImpl(OrtValue &ort_value,
                                                        int ort_value_idx,
                                                        const TensorShape *shape,
                                                        size_t nnz) {
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(shape);
  ORT_UNUSED_PARAMETER(nnz);
  std::string name;
  Status status = info_->value_name_idx_map_.GetName(ort_value_idx, name);
  if (!status.IsOK()) {
    return status;
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::RUNTIME_EXCEPTION, "All outputs should already be allocated, but output "
          + name + " was not");
}
