// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
  return g_host->GetEnvironmentVar(var_name);
}

namespace math {

uint16_t floatToHalf(float f) {
  return g_host->math__floatToHalf(f);
}

float halfToFloat(uint16_t h) {
  return g_host->math__halfToFloat(h);
}

}  // namespace math

}  // namespace onnxruntime
