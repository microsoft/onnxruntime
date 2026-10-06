// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/session/plugin_ep/ep_library.h"

namespace onnxruntime {
/// <summary>
/// Creates an EpLibrary for each plugin execution provider that is statically linked into this build.
/// The returned libraries have not been loaded yet.
/// </summary>
std::vector<std::unique_ptr<EpLibrary>> CreateStaticPluginEpLibraries();
}  // namespace onnxruntime
