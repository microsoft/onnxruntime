// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <cstring>

namespace onnxruntime {
namespace cuda {

constexpr const char* kGemmAutoTuneDiagnosticsEnvVar = "ORT_CUDA_GEMM_AUTO_TUNE_DIAGNOSTICS";

inline bool IsGemmAutoTuneDiagnosticsEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv(kGemmAutoTuneDiagnosticsEnvVar);
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}

}  // namespace cuda
}  // namespace onnxruntime
