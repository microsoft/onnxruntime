// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/util/include/default_providers.h"
#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif

namespace onnxruntime {
namespace test {

inline bool HasCudaEnvironment(int min_cuda_architecture) {
  if (DefaultCudaExecutionProvider().get() == nullptr) {
    return false;
  }

  if (min_cuda_architecture == 0) {
    return true;
  }

  int cuda_architecture = 0;

#ifdef USE_CUDA
  int currentCudaDevice = 0;
  cudaGetDevice(&currentCudaDevice);
  cudaDeviceSynchronize();
  cudaDeviceProp prop;
  if (cudaSuccess != cudaGetDeviceProperties(&prop, currentCudaDevice)) {
    return false;
  }

  cuda_architecture = prop.major * 100 + prop.minor * 10;
#endif

  return cuda_architecture >= min_cuda_architecture;
}

inline bool NeedSkipIfCudaArchLowerThan(int min_cuda_architecture) {
  // only skip when CUDA ep is enabled.
  if (DefaultCudaExecutionProvider().get() != nullptr) {
    return !HasCudaEnvironment(min_cuda_architecture);
  }
  return false;
}
}  // namespace test
}  // namespace onnxruntime
