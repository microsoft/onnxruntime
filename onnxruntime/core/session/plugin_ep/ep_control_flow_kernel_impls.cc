// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_control_flow_kernel_impls.h"

#include <algorithm>
#include <vector>

#include "core/framework/error_code_helper.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

//
// PluginEpControlFlowKernelImpl
//

PluginEpControlFlowKernelImpl::PluginEpControlFlowKernelImpl() : OrtKernelImpl{} {
  ort_version_supported = ORT_API_VERSION;

  // Indicate that this is a control flow OrtKernelImpl created by ORT.
  // Without RTTI, this gives ORT some way to check that static casting a OrtKernelImpl to
  // PluginEpControlFlowKernelImpl is valid.
  flags = OrtKernelImplFlags::kIsControlFlowKernelImpl;
}

//
// PluginEpIfKernelImpl
//

PluginEpIfKernelImpl::PluginEpIfKernelImpl(const OpKernelInfo& info) : kernel_(info) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpIfKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr,
                                                          OrtKernelContext* kernel_ctx) noexcept {
  API_IMPL_BEGIN
  auto* plugin_ep_kernel = static_cast<PluginEpIfKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
  API_IMPL_END
}

/*static*/
void ORT_API_CALL PluginEpIfKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpIfKernelImpl*>(this_ptr);
}

//
// PluginEpLoopKernelImpl
//

PluginEpLoopKernelImpl::PluginEpLoopKernelImpl(const OpKernelInfo& info, gsl::not_null<OrtLoopKernelHelper*> helper)
    : kernel_(info), helper_(helper) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  auto concat_output_func = [this](void* stream, std::vector<OrtValue>& per_iteration_outputs,
                                   void* output, size_t output_size_in_bytes) -> Status {
    std::vector<OrtValue*> value_ptrs;

    value_ptrs.reserve(per_iteration_outputs.size());
    std::transform(per_iteration_outputs.begin(), per_iteration_outputs.end(), std::back_inserter(value_ptrs),
                   [](OrtValue& value) -> OrtValue* { return &value; });

    return ToStatusAndRelease(helper_->ConcatOutput(helper_, stream, value_ptrs.data(), value_ptrs.size(),
                                                    output, output_size_in_bytes));
  };

  kernel_.SetConcatOutputFunc(concat_output_func);
}

PluginEpLoopKernelImpl::~PluginEpLoopKernelImpl() {
  helper_->Release(helper_);
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpLoopKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr,
                                                            OrtKernelContext* kernel_ctx) noexcept {
  API_IMPL_BEGIN
  auto* plugin_ep_kernel = static_cast<PluginEpLoopKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
  API_IMPL_END
}

/*static*/
void ORT_API_CALL PluginEpLoopKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpLoopKernelImpl*>(this_ptr);
}

//
// PluginEpScanKernelImpl
//

PluginEpScanKernelImpl::PluginEpScanKernelImpl(const OpKernelInfo& info, gsl::not_null<OrtScanKernelHelper*> helper)
    : kernel_(info), helper_(helper) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  // Bundle EP's function + state into a functor.
  auto transpose_func = [this](const gsl::span<const size_t>& permutation,
                               const Tensor& input, Tensor& output, Stream* stream) -> Status {
    auto empty_tensor_deleter = [](void* /*data*/) -> void { /* do not delete Tensor (not owned) */ };
    const OrtValue ort_value_input(const_cast<Tensor*>(&input), DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtValue ort_value_output(&output, DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtSyncStream* ort_stream = reinterpret_cast<OrtSyncStream*>(stream);

    return ToStatusAndRelease(helper_->Transpose(helper_, permutation.data(), permutation.size(),
                                                 &ort_value_input, ort_stream, &ort_value_output));
  };

  scan::detail::DeviceHelpers device_helpers{};
  device_helpers.transpose_func = transpose_func;

  kernel_.SetDeviceHelpers(device_helpers);
}

PluginEpScanKernelImpl::~PluginEpScanKernelImpl() {
  helper_->Release(helper_);
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpScanKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr,
                                                            OrtKernelContext* kernel_ctx) noexcept {
  API_IMPL_BEGIN
  auto* plugin_ep_kernel = static_cast<PluginEpScanKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
  API_IMPL_END
}

/*static*/
void ORT_API_CALL PluginEpScanKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernelImpl*>(this_ptr);
}
}  // namespace onnxruntime
