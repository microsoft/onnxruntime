// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

namespace onnxruntime::telemetry_internal {

const std::string& GetAppSessionGuid();
const char* GetDefaultEncodedToken();
int32_t GetProcessorCount();
std::string GetCachePath(const std::string& directory);

}  // namespace onnxruntime::telemetry_internal
