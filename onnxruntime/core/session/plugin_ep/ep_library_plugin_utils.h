// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/common/status.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace ep_library_plugin_utils {

// Shared implementation for EpLibrary types that obtain OrtEpFactory instances from a plugin EP's
// CreateEpFactories and ReleaseEpFactory entry points. The entry points are resolved at load time for a
// dynamically loaded library and at link time for a statically linked one, but the calling convention and
// lifetime rules are the same either way.

/// <summary>
/// Calls the CreateEpFactories entry point and appends the factories it produces to `factories`.
/// `factories` is left unmodified if the entry point fails.
/// </summary>
Status CreateFactories(CreateEpApiFactoriesFn create_fn, const std::string& registration_name,
                       std::vector<OrtEpFactory*>& factories);

/// <summary>
/// Calls the ReleaseEpFactory entry point for every factory in `factories` and clears it.
/// Errors are logged rather than returned because the caller cannot meaningfully recover: the current
/// implementation assumes any failure is permanent and does not leave pieces around to re-attempt a release.
/// `library_description` identifies the library in log messages.
/// </summary>
void ReleaseFactories(ReleaseEpApiFactoryFn release_fn, std::vector<OrtEpFactory*>& factories,
                      std::string_view library_description);

}  // namespace ep_library_plugin_utils
}  // namespace onnxruntime
