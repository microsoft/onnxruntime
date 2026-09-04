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

  constexpr int kWideOutputMinN = 16384;
  constexpr int kWideOutputMinWindows = 80;
  constexpr int kLongReductionMinN = 5120;
  constexpr int kLongReductionMinWindows = 128;

  // The qualified 48-SM SM121 GPU benefits from KSplit32 in two measured low-M regimes:
  // wide outputs with substantial K and narrower outputs with very long reductions.
  if (compute_capability_major == 12 && compute_capability_minor == 1 &&
      sm_count == 48 && m <= 8 &&
      ((n >= kWideOutputMinN && windows >= kWideOutputMinWindows) ||
       (n >= kLongReductionMinN && windows >= kLongReductionMinWindows))) {
    k_split = 32;
  }

  return k_split;
}

}  // namespace onnxruntime::contrib::cuda
