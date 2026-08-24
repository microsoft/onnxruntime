// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kMatMul4BitsM1ColsPerBlock = 8;
constexpr int kMatMul4BitsM1PipelineCtasPerSm = 8;

// Explicit prefetching helps when the launch has too few CTAs to hide memory
// latency, but can add overhead once the grid already supplies enough waves.
inline bool ShouldUseMatMul4BitsM1Pipeline(int n, int sm_count) {
  if (sm_count <= 0 || n % kMatMul4BitsM1ColsPerBlock != 0) {
    return false;
  }

  const int64_t grid_size = n / kMatMul4BitsM1ColsPerBlock;
  return grid_size < static_cast<int64_t>(sm_count) * kMatMul4BitsM1PipelineCtasPerSm;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
