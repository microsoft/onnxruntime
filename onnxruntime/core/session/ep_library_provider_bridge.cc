// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_provider_bridge.h"

#include "core/framework/error_code_helper.h"
#include "core/framework/session_options.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_factory_internal.h"

namespace onnxruntime {
Status EpLibraryProviderBridge::Load() {
  // wrap the EpLibraryPlugin factories that were created by calling CreateEpFactories.
  // use GetDeviceInfoIfSupported from the factory.
  // call Provider::CreateIExecutionProvider in EpFactoryInternal::CreateIExecutionProvider.
  for (const auto& factory : ep_library_plugin_->GetFactories()) {
    const auto is_supported_fn = [&factory](const OrtHardwareDevice* device,
                                            OrtKeyValuePairs** ep_metadata,
                                            OrtKeyValuePairs** ep_options) -> bool {
      return factory->GetDeviceInfoIfSupported(factory, device, ep_metadata, ep_options);
    };

    const auto create_fn = [this, &factory](const OrtHardwareDevice* const* devices,
                                            const OrtKeyValuePairs* const* ep_metadata_pairs,
                                            size_t num_devices,
                                            const OrtSessionOptions* session_options,
                                            const OrtLogger* logger, std::unique_ptr<IExecutionProvider>* ep) {
      // get the provider options
      auto ep_options = GetOptionsFromSessionOptions(factory->GetName(factory), session_options->value);
      auto& provider = provider_library_->Get();

      auto status = provider.CreateIExecutionProvider(devices, ep_metadata_pairs, num_devices,
                                                      ep_options, *session_options, *logger, *ep);

      return ToOrtStatus(status);
    };

    auto internal_factory = std::make_unique<EpFactoryInternal>(factory->GetName(factory),
                                                                factory->GetVendor(factory),
                                                                is_supported_fn,
                                                                create_fn);

    factory_ptrs_.push_back(internal_factory.get());
    internal_factory_ptrs_.push_back(internal_factory.get());
    factories_.push_back(std::move(internal_factory));
  }

  return Status::OK();
}

Status EpLibraryProviderBridge::Unload() {
  provider_library_->Unload();
  return Status::OK();
}

}  // namespace onnxruntime
