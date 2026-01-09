// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_control_flow_kernel_impls.h"

#include <algorithm>
#include <vector>

#include "core/framework/error_code_helper.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {

static OrtStatus* ORT_API_CALL GetControlFlowKernelImpl(OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept {
  // This OrtKernelImpl* IS the underlying control flow OrtKernelImpl instance.
  *out = this_ptr;
  return nullptr;
}

//
// PluginEpControlFlowKernelImpl
//

PluginEpControlFlowKernelImpl::PluginEpControlFlowKernelImpl() : OrtKernelImpl{} {
  ort_version_supported = ORT_API_VERSION;
  GetControlFlowKernel = GetControlFlowKernelImpl;
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
  auto plugin_ep_kernel = static_cast<PluginEpIfKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
void ORT_API_CALL PluginEpIfKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpIfKernelImpl*>(this_ptr);
}

//
// PluginEpLoopKernelImpl
//

PluginEpLoopKernelImpl::PluginEpLoopKernelImpl(const OpKernelInfo& info, OrtLoopConcatOutputFunc ort_concat_func,
                                               void* ort_concat_func_state) : kernel_(info) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  // Bundle EP's function + state into a functor.
  auto concat_output_func =
      [ep_concat_func = ort_concat_func,
       ep_concat_func_state = ort_concat_func_state](void* stream,
                                                     std::vector<OrtValue>& per_iteration_output,
                                                     void* output,
                                                     size_t output_size_in_bytes) -> Status {
    std::vector<OrtValue*> value_ptrs;

    value_ptrs.reserve(per_iteration_output.size());
    std::transform(per_iteration_output.begin(), per_iteration_output.end(), std::back_inserter(value_ptrs),
                   [](OrtValue& value) -> OrtValue* { return &value; });

    return ToStatusAndRelease(ep_concat_func(ep_concat_func_state, stream,
                                             value_ptrs.data(), value_ptrs.size(),
                                             output, output_size_in_bytes));
  };

  kernel_.SetConcatOutputFunc(concat_output_func);
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpLoopKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr,
                                                            OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpLoopKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
void ORT_API_CALL PluginEpLoopKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpLoopKernelImpl*>(this_ptr);
}

//
// PluginEpScanKernelImpl
//

PluginEpScanKernelImpl::PluginEpScanKernelImpl(const OpKernelInfo& info, OrtScanTransposeFunc ort_transpose_func,
                                               void* ort_transpose_func_state) : kernel_(info) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  // Bundle EP's function + state into a functor.
  auto transpose_func =
      [ep_func = ort_transpose_func,
       ep_func_state = ort_transpose_func_state](const gsl::span<const size_t>& permutations,
                                                 const Tensor& input, Tensor& output, Stream* stream) -> Status {
    auto empty_tensor_deleter = [](void* /*data*/) -> void { /* do not delete Tensor (not owned) */ };
    const OrtValue ort_value_input(const_cast<Tensor*>(&input), DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtValue ort_value_output(&output, DataTypeImpl::GetType<Tensor>(), empty_tensor_deleter);
    OrtSyncStream* ort_stream = reinterpret_cast<OrtSyncStream*>(stream);

    return ToStatusAndRelease(ep_func(ep_func_state, permutations.data(), permutations.size(),
                                      &ort_value_input, ort_stream, &ort_value_output));
  };

  scan::detail::DeviceHelpers device_helpers{};
  device_helpers.transpose_func = transpose_func;

  kernel_.SetDeviceHelpers(device_helpers);
}

/*static*/
OrtStatus* ORT_API_CALL PluginEpScanKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr,
                                                            OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernelImpl*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
void ORT_API_CALL PluginEpScanKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernelImpl*>(this_ptr);
}
}  // namespace onnxruntime
