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

}  // namespace onnxruntime::contrib::cuda
