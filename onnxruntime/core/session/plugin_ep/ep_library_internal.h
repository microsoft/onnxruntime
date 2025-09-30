// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/session/plugin_ep/ep_library.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/provider_bridge_library.h"

namespace onnxruntime {

/// <summary>
/// EpLibraryInternal wraps statically included execution providers (i.e. 'internal') so they can return OrtEpFactory
/// instances in the same way as dynamically loaded libraries.
///
/// It returns an EpFactoryInternal factory instance, which provides the ability to directly create an
/// IExecutionProvider instance for the wrapped execution provider.
/// </summary>
class EpLibraryInternal : public EpLibrary {
 public:
  EpLibraryInternal(std::unique_ptr<EpFactoryInternal> factory)
      : factory_{std::move(factory)}, factory_ptrs_{factory_.get()} {
  }

  const char* RegistrationName() const override {
    return factory_->GetName();  // same as EP name for internally registered libraries
  }

  const std::vector<OrtEpFactory*>& GetFactories() override {
    return factory_ptrs_;
  }

  // there's only ever one currently
  EpFactoryInternal& GetInternalFactory() {
    return *factory_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryInternal);

  // create instances for all internal EPs included in this build.
  static std::vector<std::unique_ptr<EpLibraryInternal>> CreateInternalEps();

 private:
  static std::unique_ptr<EpLibraryInternal> CreateCpuEp();
#if defined(USE_DML)
  static std::unique_ptr<EpLibraryInternal> CreateDmlEp();
#endif
#if defined(USE_WEBGPU)
  static std::unique_ptr<EpLibraryInternal> CreateWebGpuEp();
#endif

  std::unique_ptr<EpFactoryInternal> factory_;  // all internal EPs register a single factory currently
  std::vector<OrtEpFactory*> factory_ptrs_;     // for convenience
};

}  // namespace onnxruntime
