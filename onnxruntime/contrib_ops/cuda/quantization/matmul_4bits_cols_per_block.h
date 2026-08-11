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

// Target CTAs per SM for the M=1 GEMV. When the grid from 8 cols/CTA is smaller
// than sm_count * kTargetCtasPerSm, we halve cols_per_block to double the grid,
// improving occupancy on narrow n without changing the per-column reduction.
constexpr int kTargetCtasPerSm = 12;

// Select the number of columns per CTA (8, 4, or 2) for the M=1 kernel so that
// the grid fills the device. The reduction within each CTA is per-column (one warp
// per column), so changing cols_per_block only changes how many columns share a CTA's
// __syncthreads() barrier for shared-memory loads — the arithmetic per output element
// is identical regardless of this choice.
//
// Returns 8 when the default grid already meets the occupancy target, or when
// sm_count is unavailable (0). Never returns less than 2 — a 1-column CTA would
// over-subscribe activation re-reads. Deterministic for (n, sm_count).
inline int SelectColsPerBlock(int n, int sm_count) {
  if (sm_count <= 0) {
    return kColsPerThreadBlock;  // fail safe: use default 8
  }
  const int target = sm_count * kTargetCtasPerSm;
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
