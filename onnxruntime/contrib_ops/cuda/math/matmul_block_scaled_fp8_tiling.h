// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime::contrib::cuda {

inline int PickGenericFp8MmaKSplit(int n, int windows) {
  int k_split = (n >= 8192) ? 8 : 16;
  if (windows < k_split) {
    k_split = (windows >= 8) ? 8 : 4;
  }
  return k_split;
}

inline int PickFp8MmaKSplit(int n, int m, int windows, int sm_count,
                            int compute_capability_major, int compute_capability_minor) {
  int k_split = PickGenericFp8MmaKSplit(n, windows);

  constexpr int kOutputColumnsPerBlock = 16;
  constexpr int kWideOutputMinBlocks = 1024;
  constexpr int kLongReductionMinBlocks = 320;
  constexpr int kWideOutputMinWindows = 80;
  constexpr int kLongReductionMinWindows = 128;
  const int output_blocks = (n + kOutputColumnsPerBlock - 1) / kOutputColumnsPerBlock;

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

}  // namespace onnxruntime::contrib::cuda
