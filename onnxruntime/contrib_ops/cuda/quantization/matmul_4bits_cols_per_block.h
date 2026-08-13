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

// Candidate cols-per-block values, largest first. Order matters: ties are resolved in
// favour of the earlier (larger) candidate, which is the upstream launch geometry.
constexpr int kColsPerBlockCandidates[] = {8, 4, 2};
constexpr int kNumColsPerBlockCandidates = 3;

// Pure, host-testable form of the occupancy-driven launch choice. max_blocks_per_sm[i] is
// what cudaOccupancyMaxActiveBlocksPerMultiprocessor reported for kColsPerBlockCandidates[i]
// (<= 0 means the query failed for that candidate).
//
// The objective is lexicographic:
//   1. Maximise the number of SMs that receive at least one CTA, min(grid, sm_count).
//   2. Maximise resident warps, min(grid, max_blocks_per_sm * sm_count) * cols_per_block.
//   3. On a tie, keep the largest candidate, i.e. the upstream geometry.
//
// Criterion 1 exists because criterion 2 alone cannot ever prefer a smaller cols_per_block:
// resident warps per SM is essentially invariant in cols_per_block (halving the columns per
// CTA halves the warps per CTA and roughly doubles the CTAs that fit, and register, shared
// memory and warp-slot limits all scale the same way), so criterion 2 ties everywhere except
// where the hardware CTAs-per-SM cap bites, which penalises the smaller candidate. Criterion 1
// is the effect this selection exists for: with n = 256 on a 72-SM device, cols=8 launches 32
// CTAs and leaves 40 SMs idle, while cols=2 launches 128 and busies all of them. Total warps
// are n either way; spreading them over more SMs is what shortens a latency-bound GEMV.
//
// When the grid already covers every SM (criterion 1 tied) this returns 8, so wide-n shapes
// keep exactly the upstream launch configuration.
//
// Falls back to SelectColsPerBlock(n, sm_count) when every candidate is unusable.
// Deterministic for (n, sm_count, max_blocks_per_sm).
inline int ChooseColsPerBlockFromOccupancy(int n, int sm_count, const int* max_blocks_per_sm) {
  if (sm_count <= 0 || max_blocks_per_sm == nullptr) {
    return SelectColsPerBlock(n, sm_count);
  }

  int best_cols = 0;
  long long best_sms_busy = -1;
  long long best_active_warps = -1;
  for (int i = 0; i < kNumColsPerBlockCandidates; ++i) {
    const int cpb = kColsPerBlockCandidates[i];
    if (n % cpb != 0 || max_blocks_per_sm[i] <= 0) {
      continue;
    }
    const long long grid = n / cpb;
    const long long sms_busy = grid < sm_count ? grid : sm_count;
    const long long capacity = static_cast<long long>(max_blocks_per_sm[i]) * sm_count;
    const long long resident_blocks = grid < capacity ? grid : capacity;
    const long long active_warps = resident_blocks * cpb;
    if (sms_busy > best_sms_busy ||
        (sms_busy == best_sms_busy && active_warps > best_active_warps)) {
      best_sms_busy = sms_busy;
      best_active_warps = active_warps;
      best_cols = cpb;
    }
  }

  return best_cols > 0 ? best_cols : SelectColsPerBlock(n, sm_count);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
