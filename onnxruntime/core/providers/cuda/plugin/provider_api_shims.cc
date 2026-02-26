// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider API shims used by migrated CUDA kernels.
//
// Two compilation paths are supported:
//   1. Legacy plugin path (!ORT_CUDA_PLUGIN_USE_ADAPTER): delegates to the
//      provider host bridge (`g_host`) for environment queries and math helpers.
//   2. Adapter path (ORT_CUDA_PLUGIN_USE_ADAPTER): uses standard library and
//      core headers directly, avoiding the SHARED_PROVIDER indirection.

#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
#include "core/providers/shared_library/provider_api.h"
#else
// Adapter path: no SHARED_PROVIDER bridge needed.
// provider_api.h is a no-op when ORT_CUDA_PLUGIN_USE_ADAPTER is defined,
// so we only include the specific headers we actually use.
#include <string>
#include <cstdlib>
#include "core/common/float16.h"
#endif

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
  return g_host->GetEnvironmentVar(var_name);
#else
  const char* val = std::getenv(var_name.c_str());
  return val ? std::string(val) : std::string();
#endif
}

namespace math {

uint16_t floatToHalf(float f) {
#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
  return g_host->math__floatToHalf(f);
#else
  return MLFloat16(f).val;
#endif
}

float halfToFloat(uint16_t h) {
#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER
  return g_host->math__halfToFloat(h);
#else
  return MLFloat16::FromBits(h).ToFloat();
#endif
}

}  // namespace math

}  // namespace onnxruntime
