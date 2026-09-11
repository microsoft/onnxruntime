// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <limits>
#include <string>

namespace onnxruntime::cuda {

constexpr int64_t kQDQMaxElementCount = std::numeric_limits<int32_t>::max();

constexpr bool IsQDQElementCountSupported(int64_t element_count) {
  return element_count >= 0 && element_count <= kQDQMaxElementCount;
}

inline std::string QDQElementCountErrorMessage() {
  return "CUDA QuantizeLinear and DequantizeLinear support at most INT32_MAX elements.";
}

}  // namespace onnxruntime::cuda
