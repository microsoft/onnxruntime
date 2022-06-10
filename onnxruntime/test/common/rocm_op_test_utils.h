// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test/util/include/default_providers.h"
#ifdef USE_ROCM
#include "hip/hip_runtime_api.h"
#endif

namespace onnxruntime {
namespace test {

inline bool HasRocmEnvironment(int min_rocm_architecture) {
  if (DefaultRocmExecutionProvider().get() == nullptr) {
    return false;
  }

  if (min_rocm_architecture == 0) {
    return true;
  }

  int rocm_architecture = 0;

#ifdef USE_ROCM
  int currentRocmDevice = 0;
  if (hipSuccess != hipGetDevice(&currentRocmDevice)) {
    return false;
  }
  if (hipSuccess != hipDeviceSynchronize()) {
    return false;
  }
  hipDeviceProp_t prop;
  if (hipSuccess != hipGetDeviceProperties(&prop, currentRocmDevice)) {
    return false;
  }

  rocm_architecture = prop.major * 100 + prop.minor * 10;
#endif

  return rocm_architecture >= min_rocm_architecture;
}


} // namespace test
} // namespace onnxruntime
