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
      : factory_{std::move(factory)} {
  }

  const char* RegistrationName() const override {
    return factory_->GetName();  // same as EP name for internally registered libraries
  }

  // there's only ever one currently
  EpFactoryInternal& GetInternalFactory() {
    return *factory_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryInternal);

  // create instances for all internal EPs included in this build. allow_virtual_devices is forwarded to
  // the WebGPU EP factory (the only internal EP that can register a virtual device); see CreateWebGpuEp.
  static std::vector<std::unique_ptr<EpLibraryInternal>> CreateInternalEps(bool allow_virtual_devices);

 private:
  size_t GetFactoryCount() const override {
    return 1;
  }

  OrtEpFactory* GetFactory(size_t index) const override {
    ORT_ENFORCE(index == 0);
    return factory_.get();
  }

  static std::unique_ptr<EpLibraryInternal> CreateCpuEp();
#if defined(USE_DML)
  static std::unique_ptr<EpLibraryInternal> CreateDmlEp();
#endif
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  static std::unique_ptr<EpLibraryInternal> CreateWebGpuEp(bool allow_virtual_devices);
#endif

  std::unique_ptr<EpFactoryInternal> factory_;  // all internal EPs register a single factory currently
};

}  // namespace onnxruntime
