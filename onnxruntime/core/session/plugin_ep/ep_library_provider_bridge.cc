// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_provider_bridge.h"

#include "core/session/plugin_ep/ep_factory_provider_bridge.h"
#include "core/session/plugin_ep/ep_library_plugin.h"

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
    auto factory_impl = std::make_unique<ProviderBridgeEpFactory>(*factory, *provider_library_, library_path_);
    auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(factory_impl));

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
