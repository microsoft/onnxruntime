// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {

// TensorRT EP Segmentation fault on A100: https://github.com/microsoft/onnxruntime/issues/19530
inline const std::unordered_set<std::string> ExcludeTrtOnA100() {
  // Note: GetCudaArchitecture need USE_CUDA to be defined. Currently, it is defined when TRT EP is enabled.
  // If we want to make TRT EP independent of CUDA EP, we need to change the implementation of GetCudaArchitecture.
  if (DefaultTensorrtExecutionProvider() != nullptr && GetCudaArchitecture() == 800) {
    return {kTensorrtExecutionProvider};
  }

  return {};
}

// Add TensorRT EP to an excluded provider list when running on A100
inline const std::unordered_set<std::string>& ExcludeTrtOnA100(std::unordered_set<std::string>& excluded_providers) {
  if (DefaultTensorrtExecutionProvider() != nullptr && GetCudaArchitecture() == 800) {
    excluded_providers.insert(kTensorrtExecutionProvider);
    return excluded_providers;
  }

  return excluded_providers;
}

}  // namespace test
}  // namespace onnxruntime
