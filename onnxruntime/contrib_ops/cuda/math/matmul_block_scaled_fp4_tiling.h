// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
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
    return {16, 1};
  }

  if (wide_col_blocks >= kTargetGridWaves * sm_count &&
      (windows < kLongReductionWindows || wide_col_blocks >= kLongReductionGridWaves * sm_count)) {
    return {windows < 2 ? windows : 2, 4};
  }
  if (col_tiles >= kTargetGridWaves * sm_count) {
    return {windows >= kLongReductionWindows ? 8 : (windows < 2 ? windows : 2), 1};
  }

  int k_split = 1;
  while (k_split < 16 && k_split * 2 <= windows) {
    k_split <<= 1;
  }
  return {k_split, 1};
}

}  // namespace onnxruntime::contrib::cuda
