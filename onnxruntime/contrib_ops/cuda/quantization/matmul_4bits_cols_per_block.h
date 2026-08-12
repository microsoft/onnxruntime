// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Host-only header for SelectColsPerBlock() — the occupancy-tuning function for
// the MatMulNBits M=1 GEMV kernel. Extracted so that GPU-free unit tests can
// include it without pulling CUDA device headers (cuda_bf16.h → CUB).

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kColsPerThreadBlock = 8;

// Host-only heuristic for cols-per-block selection. Used by unit tests and as a
// fallback when the CUDA occupancy API is unavailable. The CUDA-aware path
// (SelectColsPerBlockOccupancy in matmul_4bits_m1_impl.cuh) should be preferred
// when a kernel function pointer is available — it queries actual register/shared-
// memory limits per compute capability rather than guessing.
//
// Returns 8 when the default grid already meets a conservative fill target, or
// when sm_count is unavailable (0). Never returns less than 2.
// Deterministic for (n, sm_count).
inline int SelectColsPerBlock(int n, int sm_count) {
  if (sm_count <= 0) {
    return kColsPerThreadBlock;  // fail safe: use default 8
  }
  // Conservative fill target: 8 waves per SM. This is intentionally low — the
  // real decision is made by cudaOccupancyMaxActiveBlocksPerMultiprocessor in
  // the CUDA path. This heuristic exists only for GPU-free unit tests.
  constexpr int kFallbackWavesPerSm = 8;
  const int target = sm_count * kFallbackWavesPerSm;
  // Try 8, then 4, then 2. Pick the largest that fills the target.
  if ((n / 8) >= target) {
    return 8;
  }
  if (n % 4 == 0 && (n / 4) >= target) {
    return 4;
  }
  if (n % 2 == 0) {
    return 2;
  }
  // n is odd — can only use 1-col which we refuse; fall back to 8.
  return kColsPerThreadBlock;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
