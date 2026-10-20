// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace ep_library_plugin_utils {

// Shared implementation for EpLibrary types that obtain OrtEpFactory instances from a plugin EP's
// CreateEpFactories and ReleaseEpFactory entry points. The entry points are resolved at load time for a
// dynamically loaded library and at link time for a statically linked one, but the lifetime rules are the same either
// way.

struct OrtEpFactoryDeleter {
  ReleaseEpApiFactoryFn release_fn{};

  void operator()(OrtEpFactory* factory) const noexcept;
};

using OrtEpFactoryUniquePtr = std::unique_ptr<OrtEpFactory, OrtEpFactoryDeleter>;

// Calls the CreateEpFactories entry point and appends the factories it produces to `factories`.
// `factories` is left unmodified and all returned factories are released if creation or validation fails.
Status CreateFactories(CreateEpApiFactoriesFn create_fn, ReleaseEpApiFactoryFn release_fn,
                       const std::string& registration_name,
                       std::vector<OrtEpFactoryUniquePtr>& factories);

}  // namespace ep_library_plugin_utils
}  // namespace onnxruntime
