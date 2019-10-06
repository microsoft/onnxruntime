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

#ifdef USE_CUDA
  int currentCudaDevice = 0;
  cudaGetDevice(&currentCudaDevice);
  cudaDeviceSynchronize();
  cudaDeviceProp prop;
  if (cudaSuccess != cudaGetDeviceProperties(&prop, currentCudaDevice)) {
    return false;
  }

  int cuda_architecture = prop.major * 100 + prop.minor;
  if (cuda_architecture < min_cuda_architecture) {
    return false;
  }
#endif

  return true;
}

}  // namespace test
}  // namespace onnxruntime
