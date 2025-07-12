// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_provider_bridge.h"

#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/session_options.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_factory_internal.h"

namespace onnxruntime {
Status EpLibraryProviderBridge::Load() {
  std::lock_guard<std::mutex> lock{mutex_};

  if (!factories_.empty()) {
    // already loaded
    return Status::OK();
  }

  // if we have been unloaded we can't just be reloaded.
  if (!ep_library_plugin_ || !provider_library_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "EpLibraryProviderBridge has been unloaded. "
                           "Please create a new instance using LoadPluginOrProviderBridge.");
  }

  // wrap the EpLibraryPlugin factories that were created via calling CreateEpFactories in the library.
  // use GetSupportedDevices from the library's factory.
  // to do this we need to capture `factory` and plug it in to is_supported_fn and create_fn.
  // we also need to update any returned OrtEpDevice instances to swap the wrapper EpFactoryInternal in so that we can
  // call Provider::CreateIExecutionProvider in EpFactoryInternal::CreateIExecutionProvider.
  for (const auto& factory : ep_library_plugin_->GetFactories()) {
    const auto is_supported_fn = [&factory](OrtEpFactory* ep_factory_internal,  // from factory_ptrs_
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* num_ep_devices) -> OrtStatus* {
      ORT_API_RETURN_IF_ERROR(factory->GetSupportedDevices(factory, devices, num_devices, ep_devices, max_ep_devices,
                                                           num_ep_devices));

      // add the EpFactoryInternal layer back in so that we can redirect to CreateIExecutionProvider.
      for (size_t i = 0; i < *num_ep_devices; ++i) {
        auto* ep_device = ep_devices[i];
        if (ep_device) {
          ep_device->ep_factory = ep_factory_internal;
        }
      }

      return nullptr;
    };

    const auto create_fn = [this, &factory](OrtEpFactory* /*ep_factory_internal from factory_ptrs_*/,
                                            const OrtHardwareDevice* const* devices,
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
                                                                factory->GetVendorId(factory),
                                                                is_supported_fn,
                                                                create_fn);
    factory_ptrs_.push_back(internal_factory.get());
    internal_factory_ptrs_.push_back(internal_factory.get());
    factories_.push_back(std::move(internal_factory));
  }

  return Status::OK();
}

Status EpLibraryProviderBridge::Unload() {
  std::lock_guard<std::mutex> lock{mutex_};

  internal_factory_ptrs_.clear();
  factory_ptrs_.clear();
  factories_.clear();

  // we loaded ep_library_plugin_ after provider_library_ in LoadPluginOrProviderBridge so do the reverse order here.
  ORT_RETURN_IF_ERROR(ep_library_plugin_->Unload());
  ep_library_plugin_ = nullptr;

  provider_library_->Unload();
  provider_library_ = nullptr;

  return Status::OK();
}

}  // namespace onnxruntime
