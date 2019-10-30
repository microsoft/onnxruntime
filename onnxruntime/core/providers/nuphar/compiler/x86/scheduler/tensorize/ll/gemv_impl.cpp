// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

extern "C" int gemv_update(float* cc, float* aa, float* bb, int m, int l, int stride) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < l; ++j) {
      cc[i] += aa[j] * bb[i * stride + j];
    }
  }
  return 0;
}

extern "C" int gemv_reset(float* cc, int m) {
  for (int i = 0; i < m; ++i) {
    cc[i] = 0.0;
  }
  return 0;
}
