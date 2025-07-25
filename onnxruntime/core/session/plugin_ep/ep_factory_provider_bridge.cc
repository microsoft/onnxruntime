// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_factory_provider_bridge.h"

#include "core/providers/shared_library/provider_host_api.h"

namespace onnxruntime {
OrtStatus* ProviderBridgeEpFactory::GetSupportedDevices(EpFactoryInternal& ep_factory,
                                                        const OrtHardwareDevice* const* devices,
                                                        size_t num_devices,
                                                        OrtEpDevice** ep_devices,
                                                        size_t max_ep_devices,
                                                        size_t* num_ep_devices) noexcept {
  ORT_API_RETURN_IF_ERROR(ep_factory_.GetSupportedDevices(&ep_factory_, devices, num_devices, ep_devices,
                                                          max_ep_devices, num_ep_devices));

  // add the EpFactoryInternal layer back in so that we can redirect to CreateIExecutionProvider.
  for (size_t i = 0; i < *num_ep_devices; ++i) {
    auto* ep_device = ep_devices[i];
    if (ep_device) {
      ep_device->ep_factory = &ep_factory;
    }
  }

  return nullptr;
}

OrtStatus* ProviderBridgeEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                                             const OrtKeyValuePairs* const* ep_metadata_pairs,
                                                             size_t num_devices,
                                                             const OrtSessionOptions* session_options,
                                                             const OrtLogger* session_logger,
                                                             std::unique_ptr<IExecutionProvider>* ep) noexcept {
  // get the provider specific options
  auto ep_options = GetOptionsFromSessionOptions(session_options->value);
  auto& provider = provider_library_.Get();

  auto status = provider.CreateIExecutionProvider(devices, ep_metadata_pairs, num_devices,
                                                  ep_options, *session_options, *session_logger, *ep);

  return ToOrtStatus(status);
}
}  // namespace onnxruntime
