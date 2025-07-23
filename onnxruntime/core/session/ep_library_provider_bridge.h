// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include <mutex>

#include "core/session/ep_library.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/provider_bridge_library.h"

namespace onnxruntime {

/// <summary>
/// EpLibraryProviderBridge wraps execution providers that use the provider bridge so they can return OrtEpFactory
/// instances.
///
/// It returns an EpFactoryInternal factory instance, which provides the ability to directly create an
/// IExecutionProvider instance for the wrapped execution provider.
/// </summary>
class EpLibraryProviderBridge : public EpLibrary {
 public:
  EpLibraryProviderBridge(std::unique_ptr<ProviderLibrary> provider_library,
                          std::unique_ptr<EpLibrary> ep_library_plugin)
      : provider_library_{std::move(provider_library)},
        ep_library_plugin_{std::move(ep_library_plugin)} {
  }

  const char* RegistrationName() const override {
    return ep_library_plugin_->RegistrationName();
  }

  const std::vector<OrtEpFactory*>& GetFactories() override {
    return factory_ptrs_;
  }

  // Provider bridge EPs are 'internal' as they can provide an IExecutionProvider instance directly.
  const std::vector<EpFactoryInternal*>& GetInternalFactories() {
    return internal_factory_ptrs_;
  }

  Status Load() override;
  Status Unload() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryProviderBridge);

 private:
  std::mutex mutex_;
  std::unique_ptr<ProviderLibrary> provider_library_;  // provider bridge EP library

  // EpLibraryPlugin that provides the CreateEpFactories and ReleaseEpFactory implementations.
  // we wrap the OrtEpFactory instances it contains to pass through function calls, and
  // implement EpFactoryInternal::CreateIExecutionProvider by calling Provider::CreateIExecutionProvider.
  std::unique_ptr<EpLibrary> ep_library_plugin_;

  std::vector<std::unique_ptr<EpFactoryInternal>> factories_;
  std::vector<OrtEpFactory*> factory_ptrs_;                // for convenience
  std::vector<EpFactoryInternal*> internal_factory_ptrs_;  // for convenience
};
}  // namespace onnxruntime
