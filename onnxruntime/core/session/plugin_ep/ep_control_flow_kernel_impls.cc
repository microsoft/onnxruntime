// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_control_flow_kernel_impls.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "core/framework/error_code_helper.h"
#include "core/providers/cpu/controlflow/utils.h"

namespace onnxruntime {

//
// PluginEpControlFlowKernelImpl
//

PluginEpControlFlowKernelImpl::PluginEpControlFlowKernelImpl() : OrtKernelImpl{} {
  ort_version_supported = ORT_API_VERSION;
  GetControlFlowKernel = [](OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept -> OrtStatus* {
    *out = this_ptr;
    return nullptr;
  };
}

//
// PluginEpIfKernelImpl
//

/*static*/
Status PluginEpIfKernelImpl::Create(const OpKernelInfo& info, /*out*/ std::unique_ptr<PluginEpIfKernelImpl>& out) {
  out = std::make_unique<PluginEpIfKernelImpl>(info, PrivateTag{});
  return Status::OK();
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

PluginEpIfKernelImpl::PluginEpIfKernelImpl(const OpKernelInfo& info, PrivateTag)
    : kernel_(info) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

//
// PluginEpLoopKernelImpl
//

/*static*/
Status PluginEpLoopKernelImpl::Create(const OpKernelInfo& info, const OrtLoopKernelConfig& config,
                                      /*out*/ std::unique_ptr<PluginEpLoopKernelImpl>& out) {
  auto concat_output_func =
      [ep_concat_func = config.concat_output_func,
       ep_concat_func_state = config.concat_output_func_state](void* stream,
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

  out = std::make_unique<PluginEpLoopKernelImpl>(info, concat_output_func, PrivateTag{});
  return Status::OK();
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

PluginEpLoopKernelImpl::PluginEpLoopKernelImpl(const OpKernelInfo& info, Loop::ConcatOutput concat_output_func,
                                               PrivateTag)
    : kernel_(info) {
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  kernel_.SetConcatOutputFunc(concat_output_func);
}

//
// PluginEpScanKernelImpl
//

/*static*/
template <>
Status PluginEpScanKernelImpl<8>::Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                                         /*out*/ std::unique_ptr<PluginEpScanKernelImpl<8>>& out) {
  if (config.zero_data_func == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Kernel configuration for Scan opset 8 requires a valid ",
                           "OrtScanZeroDataFunc function.");
  }

  if (config.transpose_func != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "A kernel configuration with a OrtScanTransposeFunc ",
                           "function is invalid for Scan opset 8");
  }

  auto zero_data_func =
      [ep_func = config.zero_data_func,
       ep_func_state = config.zero_data_func_state](void* data, size_t size_in_bytes) -> Status {
    return ToStatusAndRelease(ep_func(ep_func_state, data, size_in_bytes));
  };

  scan::detail::DeviceHelpers device_helpers{};
  device_helpers.set_data_to_zero_func = zero_data_func;

  out = std::make_unique<PluginEpScanKernelImpl<8>>(info, device_helpers, PrivateTag{});
  return Status::OK();
}

/*static*/
template <>
Status PluginEpScanKernelImpl<9>::Create(const OpKernelInfo& info, const OrtScanKernelConfig& config,
                                         /*out*/ std::unique_ptr<PluginEpScanKernelImpl<9>>& out) {
  if (config.transpose_func == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Kernel configuration for Scan opset >= 9 requires a valid ",
                           "OrtScanTransposeFunc function.");
  }

  if (config.zero_data_func != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "A kernel configuration with a OrtScanZeroDataFunc ",
                           "function is invalid for Scan opset >= 9");
  }

  auto transpose_func =
      [ep_func = config.transpose_func,
       ep_func_state = config.transpose_func_state](const gsl::span<const size_t>& permutations,
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

  out = std::make_unique<PluginEpScanKernelImpl<9>>(info, device_helpers, PrivateTag{});
  return Status::OK();
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernelImpl<8>::ComputeImpl(OrtKernelImpl* this_ptr,
                                                               OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernelImpl<8>*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
template <>
OrtStatus* ORT_API_CALL PluginEpScanKernelImpl<9>::ComputeImpl(OrtKernelImpl* this_ptr,
                                                               OrtKernelContext* kernel_ctx) noexcept {
  auto plugin_ep_kernel = static_cast<PluginEpScanKernelImpl<9>*>(this_ptr);
  ORT_API_RETURN_IF_STATUS_NOT_OK(plugin_ep_kernel->kernel_.Compute(reinterpret_cast<OpKernelContext*>(kernel_ctx)));

  return nullptr;
}

/*static*/
template <>
void ORT_API_CALL PluginEpScanKernelImpl<8>::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernelImpl<8>*>(this_ptr);
}

/*static*/
template <>
void ORT_API_CALL PluginEpScanKernelImpl<9>::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<PluginEpScanKernelImpl<9>*>(this_ptr);
}

template <>
PluginEpScanKernelImpl<8>::PluginEpScanKernelImpl(const OpKernelInfo& info,
                                                  const scan::detail::DeviceHelpers& device_helpers, PrivateTag)
    : kernel_(info) {
  Compute = PluginEpScanKernelImpl<8>::ComputeImpl;
  Release = PluginEpScanKernelImpl<8>::ReleaseImpl;
  kernel_.SetDeviceHelpers(device_helpers);
}

template <>
PluginEpScanKernelImpl<9>::PluginEpScanKernelImpl(const OpKernelInfo& info,
                                                  const scan::detail::DeviceHelpers& device_helpers, PrivateTag)
    : kernel_(info) {
  Compute = PluginEpScanKernelImpl<9>::ComputeImpl;
  Release = PluginEpScanKernelImpl<9>::ReleaseImpl;
  kernel_.SetDeviceHelpers(device_helpers);
}

}  // namespace onnxruntime
