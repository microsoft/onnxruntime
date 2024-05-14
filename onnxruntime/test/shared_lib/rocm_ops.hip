// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "hip/hip_runtime.h"

__global__ void _Add(long long sz, float* Z, const float* X, const float* Y) {
  long long offset = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
  if (offset < sz) {
    Z[offset] = X[offset] + Y[offset];
  }
}

void rocm_add(int64_t sz, float* Z, const float* X, const float* Y, hipStream_t compute_stream) {
 _Add<<<256, 256, 0, compute_stream>>>(static_cast<long long>(sz), Z, X, Y);
}

