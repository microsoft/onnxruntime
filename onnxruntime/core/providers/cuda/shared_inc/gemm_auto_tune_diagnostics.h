// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/env_var.h"

namespace onnxruntime {
namespace cuda {

constexpr const char* kGemmAutoTuneDiagnosticsEnvVar = "ORT_CUDA_GEMM_AUTO_TUNE_DIAGNOSTICS";

inline bool IsGemmAutoTuneDiagnosticsEnabled() {
  static const bool enabled = detail::GetEnvironmentVar(kGemmAutoTuneDiagnosticsEnvVar) == "1";
  return enabled;
}

}  // namespace cuda
}  // namespace onnxruntime
