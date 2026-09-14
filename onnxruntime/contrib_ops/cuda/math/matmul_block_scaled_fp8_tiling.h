// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

constexpr int kFp8MmaOutputColumnsPerBlock = 16;

inline int Fp8MmaOutputBlocks(int n) {
  return (n + kFp8MmaOutputColumnsPerBlock - 1) / kFp8MmaOutputColumnsPerBlock;
}

inline int PickGenericFp8MmaKSplit(int n, int m, int windows, int sm_count,
                                   int compute_capability_major, int compute_capability_minor) {
  int k_split = (n >= 8192) ? 8 : 16;
  const int output_blocks = Fp8MmaOutputBlocks(n);
  const bool qualified_h200 = compute_capability_major == 9 && compute_capability_minor == 0 &&
                              sm_count == 132 && m <= 8 && output_blocks > 2 * sm_count;
  const bool qualified_rtx4090 = compute_capability_major == 8 && compute_capability_minor == 9 &&
                                 sm_count == 128 && m <= 8 && windows >= 16 && windows <= 96 &&
                                 output_blocks > 3 * sm_count;
  const bool qualified_rtx5060ti = compute_capability_major == 12 && compute_capability_minor == 0 &&
                                   sm_count == 36 && m <= 8 && windows >= 40 && windows <= 96 &&
                                   output_blocks > 2 * sm_count;
  if (qualified_h200 || qualified_rtx4090 || qualified_rtx5060ti) {
    k_split = 8;
  }
  if (windows < k_split) {
    k_split = (windows >= 8) ? 8 : 4;
  }
  return k_split;
}

inline int PickFp8MmaKSplit(int n, int m, int windows, int sm_count,
                            int compute_capability_major, int compute_capability_minor) {
  int k_split = PickGenericFp8MmaKSplit(n, m, windows, sm_count,
                                        compute_capability_major, compute_capability_minor);

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
