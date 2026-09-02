// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
  int col_groups;
};

// Returns the tensor-core GEMV tiling selected for the given shape and device.
inline Fp4MmaConfig PickFp4MmaConfig(int m, int n, int k, int sm_count,
                                     int compute_capability_major, int compute_capability_minor) {
  constexpr int kSm121TargetSmCount = 48;
  constexpr int kSm121MaxM = 16;
  constexpr int kTargetGridWaves = 4;
  constexpr int kLongReductionGridWaves = 8;
  constexpr int kSm121MinWindows = 40;
  constexpr int kSm121NarrowGridMinWindows = 128;
  constexpr int kLongReductionWindows = 64;
  const int windows = k >> 7;
  const int col_tiles = (n + 15) / 16;
  const int wide_col_blocks = (col_tiles + 3) / 4;

  // The 48-SM SM121 GPU benefits from more K parallelism for the wide-grid and long-reduction
  // regimes below, unless N alone already provides eight waves of four-column blocks. Preserve the
  // generic selector for unqualified SM counts, shorter reductions, narrower grids, SM120, and the
  // wider M=32 row tiling, where this schedule can regress.
  if (compute_capability_major == 12 && compute_capability_minor == 1 &&
      sm_count == kSm121TargetSmCount && m <= kSm121MaxM && windows >= kSm121MinWindows &&
      (wide_col_blocks >= kTargetGridWaves * sm_count || windows >= kSm121NarrowGridMinWindows) &&
      wide_col_blocks < kLongReductionGridWaves * sm_count) {
    return {16, 1, 1};
  }

  if (wide_col_blocks >= kTargetGridWaves * sm_count &&
      (windows < kLongReductionWindows || wide_col_blocks >= kLongReductionGridWaves * sm_count)) {
    return {windows < 2 ? windows : 2, 4, 1};
  }
  if (col_tiles >= kTargetGridWaves * sm_count) {
    return {windows >= kLongReductionWindows ? 8 : (windows < 2 ? windows : 2), 1, 1};
  }

  int k_split = 1;
  while (k_split < 16 && k_split * 2 <= windows) {
    k_split <<= 1;
  }
  return {k_split, 1, 1};
}

// Returns the tiling for the column-grouped GEMV, where one warp owns two adjacent 16-column
// tiles and reuses a single activation fragment for both. That halves the activation L1 traffic
// per weight byte, which is what the ungrouped kernel is bound by at small M: ncu on H200 at
// N = 17408, K = 5120, M = 8 reports 65.9% L1/TEX against only 33.3% DRAM, at 0.82 waves per SM.
//
// Grouping halves the column grid, so it is only taken when the grouped grid still covers a
// couple of blocks per SM, and KSplit is then raised to land near 16 resident warps per SM.
// Measured on H200 (132 SMs, M = 8, half), kernel time from nsys:
//
//   N = 17408, K =  5120   28.38 -> 24.64 us (1.15x)  KSplit 4, ColGroups 2
//   N =  5120, K = 17408   28.67 -> 28.35 us          only 160 grouped blocks, not selected
//
// Only valid when M fits one mma row tile (M <= 8): a wider M multiplies the shared-memory
// reduction buffer past the 48 KB static limit.
inline Fp4MmaConfig PickFp4MmaGroupedConfig(int m, int n, int k, int sm_count,
                                            int compute_capability_major, int compute_capability_minor) {
  constexpr int kColGroups = 2;
  constexpr int kMinBlocksPerSm = 2;
  constexpr int kWarpBudgetPerSm = 24;
  const int windows = k >> 7;
  const int col_blocks = (n + 16 * kColGroups - 1) / (16 * kColGroups);
  if (col_blocks < kMinBlocksPerSm * sm_count) {
    return PickFp4MmaConfig(m, n, k, sm_count, compute_capability_major, compute_capability_minor);
  }

  int k_split = 1;
  while (k_split < 16 && k_split * 2 <= windows &&
         col_blocks * (k_split * 2) <= kWarpBudgetPerSm * sm_count) {
    k_split <<= 1;
  }
  return {k_split, 1, kColGroups};
}

}  // namespace onnxruntime::contrib::cuda
