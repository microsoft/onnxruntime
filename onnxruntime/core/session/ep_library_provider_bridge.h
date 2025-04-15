// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/ep_library.h"
#include "core/session/ep_factory_internal.h"
#include "core/session/provider_bridge_library.h"

namespace onnxruntime {

struct EpLibraryProviderBridge : EpLibrary {
  EpLibraryProviderBridge(const std::string& registration_name, const ORTCHAR_T* library_path)
      : registration_name_{registration_name},
        library_path_{library_path},
        provider_library_{library_path} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
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
  std::unique_ptr<EpFactoryInternal> CreateCudaEpFactory(Provider& provider);

  std::string registration_name_;
  std::filesystem::path library_path_;
  ProviderLibrary provider_library_;  // handles onnxruntime_providers_shared and the provider bridge EP library
  std::vector<std::unique_ptr<EpFactoryInternal>> factories_;
  std::vector<OrtEpFactory*> factory_ptrs_;                // for convenience
  std::vector<EpFactoryInternal*> internal_factory_ptrs_;  // for convenience
};
}  // namespace onnxruntime
