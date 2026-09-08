// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <string>

namespace onnxruntime::cuda {

constexpr int64_t kGatherElementsMaxElementCount = std::numeric_limits<int32_t>::max();

constexpr bool IsGatherElementsElementCountSupported(int64_t element_count) {
  return element_count >= 0 && element_count <= kGatherElementsMaxElementCount;
}

inline std::string GatherElementsElementCountErrorMessage(int64_t element_count) {
  return "CUDA GatherElements output element count " + std::to_string(element_count) +
         " is outside the supported range [0, " + std::to_string(kGatherElementsMaxElementCount) + "].";
}

}  // namespace onnxruntime::cuda
