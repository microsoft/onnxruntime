// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && defined(BUILD_WEBGPU_EP_STATIC_LIB)
#include "core/session/plugin_ep/ep_factory_webgpu.h"

#include "core/framework/error_code_helper.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

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

  return nullptr;
}

OrtStatus* WebGpuEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
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

#endif  // defined(USE_WEBGPU) && defined(BUILD_WEBGPU_EP_STATIC_LIB)
