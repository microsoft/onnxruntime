// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/session/plugin_ep/ep_factory_webgpu.h"

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

WebGpuEpFactory::~WebGpuEpFactory() {
  if (virtual_hw_device_ != nullptr) {
    OrtExecutionProviderApi::ReleaseHardwareDevice(virtual_hw_device_);
    virtual_hw_device_ = nullptr;
  }
}

OrtStatus* WebGpuEpFactory::GetSupportedDevices(EpFactoryInternal& ep_factory,
                                                const OrtHardwareDevice* const* devices,
                                                size_t num_devices,
                                                OrtEpDevice** ep_devices,
                                                size_t max_ep_devices,
                                                size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // TODO: any metadata or options to add?
      ORT_API_RETURN_IF_ERROR(OrtExecutionProviderApi::CreateEpDevice(&ep_factory,
                                                                      &device, nullptr, nullptr,
                                                                      &ep_devices[num_ep_devices++]));
    }
  }

  // If the environment allows virtual devices, register a virtual GPU EP device (vendor/device id 0) so
  // the WebGPU EP stays selectable for a device-free compile-only session on hosts where OS device
  // enumeration finds no GPU (e.g. a Win32k-lockdown sandbox). It is offered *in addition* to any real
  // GPU device, so the device-free path remains exercisable on a host that also has a real GPU. Since
  // allow_virtual_devices is opt-in, normal (real GPU) usage is unaffected.
  if (allow_virtual_devices_ && num_ep_devices < max_ep_devices) {
    OrtKeyValuePairs* hw_metadata = nullptr;
    OrtApis::CreateKeyValuePairs(&hw_metadata);
    OrtApis::AddKeyValuePair(hw_metadata, kOrtHardwareDevice_MetadataKey_IsVirtual, "1");
    OrtStatus* status = OrtExecutionProviderApi::CreateHardwareDevice(
        OrtHardwareDeviceType::OrtHardwareDeviceType_GPU, /*vendor_id=*/0, /*device_id=*/0,
        /*vendor_name=*/"Microsoft", hw_metadata, &virtual_hw_device_);
    OrtApis::ReleaseKeyValuePairs(hw_metadata);  // ORT makes a copy
    ORT_API_RETURN_IF_ERROR(status);

    ORT_API_RETURN_IF_ERROR(OrtExecutionProviderApi::CreateEpDevice(&ep_factory, virtual_hw_device_,
                                                                    nullptr, nullptr,
                                                                    &ep_devices[num_ep_devices++]));
  }

  return nullptr;
}

OrtStatus* WebGpuEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                                     const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                                     size_t num_devices,
                                                     const OrtSessionOptions* session_options,
                                                     const OrtLogger* session_logger,
                                                     std::unique_ptr<IExecutionProvider>* ep) noexcept {
  *ep = nullptr;

  if (num_devices != 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "WebGPU EP factory currently only supports one device at a time.");
  }

  // A virtual GPU device has no real GPU behind it, so it can only back a device-free compile-only session
  // (see the concept map in webgpu_context.cc). Reject the invalid combination up front with a clear message
  // instead of letting Dawn fail obscurely when it later tries to create a device.
  const bool compile_only =
      session_options->value.config_options.GetConfigOrDefault(kOrtSessionOptionCompileOnly, "0") == "1";
  const auto& device_metadata = devices[0]->metadata.Entries();
  const bool selected_virtual_device = device_metadata.count(kOrtHardwareDevice_MetadataKey_IsVirtual) != 0;
  if (selected_virtual_device && !compile_only) {
    return OrtApis::CreateStatus(
        ORT_INVALID_ARGUMENT,
        "WebGPU EP was selected on a virtual GPU device, which has no real GPU behind it and can only serve "
        "a compile-only session (session.compile_only=1). Select a real GPU device to run inference.");
  }

  auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(session_options->value.config_options);
  *ep = webgpu_ep_factory->CreateProvider();
  (*ep)->SetLogger(session_logger->ToInternal());

  return nullptr;
}

OrtStatus* WebGpuEpFactory::CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept {
  // Call the WebGPU provider's C API to create the data transfer
  // This is implemented in the WebGPU provider backend which has access to WebGPU headers
  *data_transfer = OrtWebGpuCreateDataTransfer();

  // API version mismatch is a fatal error - return error status if creation failed
  if (*data_transfer == nullptr) {
    return OrtApis::CreateStatus(ORT_RUNTIME_EXCEPTION,
                                 "Failed to create WebGPU data transfer - API version mismatch.");
  }

  return nullptr;
}

}  // namespace onnxruntime

#endif  // defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
