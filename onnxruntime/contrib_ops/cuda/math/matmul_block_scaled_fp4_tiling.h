// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
  int col_groups;
};

// Returns the tensor-core GEMV tiling selected for the given shape and device size.
inline Fp4MmaConfig PickFp4MmaConfig(int n, int k, int sm_count) {
  constexpr int kTargetGridWaves = 4;
  constexpr int kLongReductionGridWaves = 8;
  constexpr int kLongReductionWindows = 64;
  const int windows = k >> 7;
  const int col_tiles = (n + 15) / 16;

  const int wide_col_blocks = (col_tiles + 3) / 4;
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
inline Fp4MmaConfig PickFp4MmaGroupedConfig(int n, int k, int sm_count) {
  constexpr int kColGroups = 2;
  constexpr int kMinBlocksPerSm = 2;
  constexpr int kWarpBudgetPerSm = 24;
  const int windows = k >> 7;
  const int col_blocks = (n + 16 * kColGroups - 1) / (16 * kColGroups);
  if (col_blocks < kMinBlocksPerSm * sm_count) {
    return PickFp4MmaConfig(n, k, sm_count);
  }

  int k_split = 1;
  while (k_split < 16 && k_split * 2 <= windows &&
         col_blocks * (k_split * 2) <= kWarpBudgetPerSm * sm_count) {
    k_split <<= 1;
  }
  return {k_split, 1, kColGroups};
}

}  // namespace onnxruntime::contrib::cuda
