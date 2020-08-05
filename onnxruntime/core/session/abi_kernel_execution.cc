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

// taken from OptimizerExecutionFrame
Status
SingleKernelExecutionFrame::CreateNodeOutputMLValueImpl(__attribute__((unused)) OrtValue &ort_value, int ort_value_idx,
                                                        __attribute__((unused)) const TensorShape *shape,
                                                        __attribute__((unused)) size_t nnz) {
  std::string name;
  Status status = info_->value_name_idx_map_.GetName(ort_value_idx, name);
  if (!status.IsOK()) {
    return status;
  }
  return Status(ONNXRUNTIME, RUNTIME_EXCEPTION, "All outputs should already be allocated, but output "
                                                + name + " was not");
}




/*
 * ORT API Implementations
 */
ORT_API_STATUS_IMPL(OrtApis::CreateKernelSession,
                    _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtKernelSession **session_) {
  API_IMPL_BEGIN
    ORT_ENFORCE(session_, "OrtKernelSession pointer must not be null");
    std::unique_ptr<Model> model = std::make_unique<Model>("KernelExecutionModel", true,
                                                           logging::LoggingManager::DefaultLogger());

    KernelSessionImpl *session = new KernelSessionImpl(std::move(model));

    // initialize the providers
    for(auto& factory : options->provider_factories) {
      auto provider = factory->CreateProvider();
      session->provider_list.push_back(std::move(provider));
    }

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
                "provider_id must be less than the provider list size (" + std::to_string(session->provider_list.size()) + ").");

    SingleKernelExecutionFrame* frame;
    Status status = context->CreateExecutionFrame(session, &frame, provider_id);
    if (!status.IsOK()){
      return ToOrtStatus(status);
    }
    *kernel_ = reinterpret_cast<OrtExecutableKernel*>(frame);
    return ToOrtStatus(Status::OK());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_SetInput,
                    _Inout_ OrtExecutableKernel *kernel_,
                    int index,
                    _In_ OrtValue *value) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* context = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    return ToOrtStatus(context->SetInput(*value, index));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_SetOutput,
                    OrtExecutableKernel *kernel_,
                    int index,
                    OrtValue *value) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame* context = reinterpret_cast<SingleKernelExecutionFrame*>(kernel_);
    return ToOrtStatus(context->SetOutput(*value, index));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::ExecutableKernel_Compute,
                    _Inout_ OrtExecutableKernel *context_) {
  API_IMPL_BEGIN
    SingleKernelExecutionFrame*context = reinterpret_cast<SingleKernelExecutionFrame*>(context_);
    return ToOrtStatus(context->Compute());
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

