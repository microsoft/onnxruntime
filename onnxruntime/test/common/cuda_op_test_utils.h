// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// CUDA architecture of the current device like 100 * major + 10 * minor.
// Please call this function after CUDA EP is enabled.
int GetCudaArchitecture();

inline bool HasCudaEnvironment(int min_cuda_architecture) {
  if (DefaultCudaExecutionProvider().get() == nullptr) {
    return false;
  }

  return GetCudaArchitecture() >= min_cuda_architecture;
}

inline bool NeedSkipIfCudaArchLowerThan(int min_cuda_architecture) {
  // only skip when CUDA ep is enabled.
  if (DefaultCudaExecutionProvider().get() != nullptr) {
    return !HasCudaEnvironment(min_cuda_architecture);
  }
  return false;
}

inline bool NeedSkipIfCudaArchGreaterEqualThan(int max_cuda_architecture) {
  // only skip when CUDA ep is enabled.
  if (DefaultCudaExecutionProvider().get() != nullptr) {
    return HasCudaEnvironment(max_cuda_architecture);
  }
  return false;
}
}  // namespace test
}  // namespace onnxruntime
