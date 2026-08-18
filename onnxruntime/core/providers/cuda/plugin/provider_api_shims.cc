// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Provider API shims used by migrated CUDA kernels.
// Provides direct implementations of utility functions that in-tree kernels
// obtain via the SHARED_PROVIDER bridge (GetEnvironmentVar, floatToHalf,
// halfToFloat). Plugin builds skip SHARED_PROVIDER entirely, so these thin
// wrappers ensure the migrated kernel code compiles and links.

#include "provider_api_shims.h"

#include <string>
#include "core/common/float16.h"
#include "core/platform/env_var.h"  // detail::GetEnvironmentVar

namespace onnxruntime {

std::string GetEnvironmentVar(const std::string& var_name) {
  return detail::GetEnvironmentVar(var_name);
}

namespace math {

uint16_t floatToHalf(float f) {
  return MLFloat16(f).val;
}

float halfToFloat(uint16_t h) {
  return MLFloat16::FromBits(h).ToFloat();
}

}  // namespace math

}  // namespace onnxruntime
