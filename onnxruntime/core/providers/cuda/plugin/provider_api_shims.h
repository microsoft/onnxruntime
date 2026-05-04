// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Declarations for provider API shims used by the CUDA plugin EP build.
// In-tree builds get these via the SHARED_PROVIDER bridge (provider_api.h);
// the plugin build skips that bridge, so these thin wrappers provide direct
// implementations (defined in provider_api_shims.cc).

#pragma once

#include <cstdint>
#include <string>

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name);

namespace math {
uint16_t floatToHalf(float f);
float halfToFloat(uint16_t h);
}  // namespace math

}  // namespace onnxruntime
