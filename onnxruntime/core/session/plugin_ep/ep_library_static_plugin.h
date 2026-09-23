// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "core/session/plugin_ep/ep_library.h"
#include "core/session/plugin_ep/ep_library_plugin_utils.h"

namespace onnxruntime {
/// <summary>
/// EpLibraryStaticPlugin supports a plugin execution provider that is statically linked into the ORT binary.
///
/// The provider implements the same CreateEpFactories and ReleaseEpFactory entry points that a dynamically loaded
/// plugin library exports, and uses the same public EP API, but the entry points are resolved at link time instead
/// of being looked up by name. There is no library to load or unload, so the provider's process-global state
/// outlives unregistration.
/// </summary>
class EpLibraryStaticPlugin : public EpLibrary {
 public:
  EpLibraryStaticPlugin(std::string registration_name,
                        CreateEpApiFactoriesFn create_fn,
                        ReleaseEpApiFactoryFn release_fn)
      : registration_name_{std::move(registration_name)},
        create_fn_{create_fn},
        release_fn_{release_fn} {
  }

  const char* RegistrationName() const override {
    return registration_name_.c_str();
  }

  // LibraryPath() intentionally uses the base class implementation, which returns nullptr. There is no library
  // file backing a statically linked provider.

  Status Load() override;

  Status Unload() override;

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpLibraryStaticPlugin);

 private:
  size_t GetFactoryCount() const override {
    return factories_.size();
  }

  OrtEpFactory* GetFactory(size_t index) const override {
    return factories_[index].get();
  }

  std::mutex mutex_;
  const std::string registration_name_;
  const CreateEpApiFactoriesFn create_fn_;
  const ReleaseEpApiFactoryFn release_fn_;
  std::vector<ep_library_plugin_utils::OrtEpFactoryUniquePtr> factories_{};
};
}  // namespace onnxruntime
