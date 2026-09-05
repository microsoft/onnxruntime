// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

inline int PickFp8MmaKSplit(int n, int m, int windows, int sm_count, int compute_capability_major) {
  int k_split = (n >= 8192) ? 8 : 16;
  if (windows < k_split) {
    k_split = (windows >= 8) ? 8 : 4;
  }

  // Low-SM-count client Blackwell GPUs benefit from additional K parallelism for low-M decode.
  if (compute_capability_major == 12 && sm_count <= 64 && windows >= 16) {
    k_split = 16;
    if (m <= 8 && windows >= 32 && (n >= 4096 || m <= 2)) {
      k_split = 32;
    }
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
//   * 8-warp blocks (KSplit 8, taken from N >= 8192) must not carry the attribute at all --
//     declaring it replaces nvcc's implicit bounds and costs 1.05-1.08x even when the register
//     cap is unchanged, and KSplit 32 cannot host 3 blocks per SM at all;
//   * only one row tile fits the 40-register cap that 3 blocks per SM imply. M = 16 (two tiles)
//     measures 0.74x and M = 32 (four tiles) 0.24x, both from spills.
inline bool Fp8MmaGemvPinsResidency(int n, int k_split, int m_tiles, int sm_count) {
  if (k_split != 16 || m_tiles != 1) {
    return false;
  }
  const int col_blocks = (n + 15) / 16;
  return col_blocks > 2 * sm_count && col_blocks <= 3 * sm_count;
}

}  // namespace onnxruntime::contrib::cuda
