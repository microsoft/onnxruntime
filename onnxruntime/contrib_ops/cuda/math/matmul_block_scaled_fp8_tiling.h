// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

constexpr int kFp8MmaOutputColumnsPerBlock = 16;

inline int Fp8MmaOutputBlocks(int n) {
  return (n + kFp8MmaOutputColumnsPerBlock - 1) / kFp8MmaOutputColumnsPerBlock;
}

// Chooses how many warps split the K reduction.
//
// KSplit 16 is 512 threads at 48 registers, so only two of its blocks fit one SM's 64 KB register
// file; KSplit 8 halves the block to 256 threads and fits five. A grid wider than 2 * sm_count
// output blocks therefore costs KSplit 16 a second residency round that KSplit 8 clears in one,
// and the crossover sits exactly on that boundary rather than on any particular N.
//
// Measured on H200 (132 SMs, 264 blocks) at boost clocks, forced plain KSplit 8 against the
// shipped choice, four launches per point: KSplit 8 loses below the boundary (0.90-0.93x at 262,
// 263 and 264 blocks) and wins above it continuously from 265 to 495 blocks (1.09-1.13x), where
// N >= 8192 already selected it. The boundary and the win hold for K = 2560, 5120 and 6144
// (windows 40, 80, 96) and for both M = 1 and M = 8.
inline int PickGenericFp8MmaKSplit(int n, int windows, int sm_count) {
  int k_split = Fp8MmaOutputBlocks(n) > 2 * sm_count ? 8 : 16;
  if (windows < k_split) {
    k_split = (windows >= 8) ? 8 : 4;
  }
  return k_split;
}

inline int PickFp8MmaKSplit(int n, int m, int windows, int sm_count,
                            int compute_capability_major, int compute_capability_minor) {
  int k_split = PickGenericFp8MmaKSplit(n, windows, sm_count);

  constexpr int kWideOutputMinBlocks = 1024;
  constexpr int kLongReductionMinBlocks = 320;
  constexpr int kWideOutputMinWindows = 80;
  constexpr int kLongReductionMinWindows = 128;
  const int output_blocks = Fp8MmaOutputBlocks(n);

  // The qualified 48-SM SM121 GPU benefits from KSplit32 in two measured low-M regimes:
  // wide outputs with substantial K and narrower outputs with very long reductions.
  // The wide regime remains beneficial through the measured N=248320 lm-head shape, so it
  // has no upper bound. Express these SM121 thresholds as output blocks so shapes with
  // identical launch geometry use the same override; leave the generic selector unchanged
  // to preserve behavior on other devices.
  if (compute_capability_major == 12 && compute_capability_minor == 1 &&
      sm_count == 48 && m <= 8 &&
      ((output_blocks >= kWideOutputMinBlocks && windows >= kWideOutputMinWindows) ||
       (output_blocks >= kLongReductionMinBlocks && windows >= kLongReductionMinWindows))) {
    k_split = 32;
  }

  return k_split;
}

// True when the tensor-core GEMV should launch the entry point that carries a residency hint.
//
// NOTE: PickGenericFp8MmaKSplit now selects KSplit 8 for every grid wider than 2 * sm_count
// output blocks, which is a superset of the window below, so this predicate no longer fires for
// any shape the selector produces. Plain KSplit 8 is the faster of the two in that window on
// H200 (8.672 us against 9.344 us pinned at N = 5120), so the hint is superseded rather than
// merely bypassed. It is kept for now so the selector change can be reverted in one line while
// other architectures are measured; remove it once they confirm.
//
// The mma grid is ceil(N / 16) blocks. A 16-warp block only fits twice per SM, so N just above
// 32 * sm_count spills into a second, nearly empty wave: on H200 N = 5120 launches 1.21 waves
// and ncu measures 66% active cycles. __launch_bounds__(threads, 3) makes those shapes a single
// wave, worth 1.21-1.35x. Outside that window it only costs registers, so:
//
//   * a grid at or below 2 blocks per SM is already one wave and must stay on the plain kernel;
//   * a grid above 3 blocks per SM stays multi-wave either way;
//   * pre-SM89 devices lack native FP8 tensor-core support and lose about 1% from the register
//     cap even inside the target grid window;
//   * 8-warp blocks must not carry the attribute at all -- declaring it replaces nvcc's implicit
//     bounds and costs 1.05-1.08x even when the register cap is unchanged, and KSplit 32 cannot
//     host 3 blocks per SM at all;
//   * only one row tile fits the 40-register cap that 3 blocks per SM imply. M = 16 (two tiles)
//     measures 0.74x and M = 32 (four tiles) 0.24x, both from spills.
inline bool Fp8MmaGemvPinsResidency(int n, int k_split, int m_tiles, int sm_count,
                                    int compute_capability_major, int compute_capability_minor) {
  if (compute_capability_major < 8 ||
      (compute_capability_major == 8 && compute_capability_minor < 9) ||
      k_split != 16 || m_tiles != 1) {
    return false;
  }
  const int col_blocks = Fp8MmaOutputBlocks(n);
  return col_blocks > 2 * sm_count && col_blocks <= 3 * sm_count;
}

}  // namespace onnxruntime::contrib::cuda
