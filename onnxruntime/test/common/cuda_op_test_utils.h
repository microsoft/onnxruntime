// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/util/math.h"
#include "test/util/include/default_providers.h"
#include "core/common/logging/logging.h"
#include "cuda_runtime_api.h"
#include <vector>

namespace onnxruntime {
namespace test {

inline std::vector<MLFloat16> ToFloat16(const std::vector<float>& data) {
  std::vector<MLFloat16> result;
  result.reserve(data.size());
  for (size_t i = 0; i < data.size(); i++) {
    result.push_back(MLFloat16(math::floatToHalf(data[i])));
  }
  return result;
}

inline int GetCudaArchitecture() {
  int currentCudaDevice;
  cudaDeviceProp prop;
  cudaGetDevice(&currentCudaDevice);
  cudaGetDeviceProperties(&prop, currentCudaDevice);
  VLOGS_DEFAULT(1) << "CUDA Device name=" << prop.name << " major=" << prop.major << " minor=" << prop.minor;
  return prop.major * 100 + prop.minor;
}

inline bool HasCudaEnvironment(int min_cuda_architecture = 0) {
  if (DefaultCudaExecutionProvider().get() == nullptr) {
    return false;
  }

  int cuda_architecture = GetCudaArchitecture();
  if (min_cuda_architecture > 0 && cuda_architecture < min_cuda_architecture) {
    VLOGS_DEFAULT(2) << "Skip test since CUDA device architecture " << cuda_architecture << " < required" << min_cuda_architecture;
    return false;
  }

  return true;
}

}  // namespace test
}  // namespace onnxruntime
