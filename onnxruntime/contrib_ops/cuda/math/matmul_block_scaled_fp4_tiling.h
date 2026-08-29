// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
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
