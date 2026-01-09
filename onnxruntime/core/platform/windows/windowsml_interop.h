// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace windowsml {

/// @brief Get the EP Catalog API from the WindowsML plugin.
/// @return Pointer to OrtEpCatalogApi, or nullptr if WindowsML is not available.
/// @note This function is thread-safe and caches the result.
const OrtEpCatalogApi* GetEpCatalogApi();

/// @brief Get the WindowsML version string.
/// @return Version string, or nullptr if WindowsML is not available.
const char* GetVersionString();

/// @brief Check if WindowsML is available on this system.
/// @return true if WindowsML.dll was loaded successfully.
bool IsAvailable();

/// @brief Shutdown and unload the WindowsML plugin.
/// @note Called during ORT shutdown. After this call, GetEpCatalogApi() returns nullptr.
void Shutdown();

}  // namespace windowsml
}  // namespace onnxruntime
