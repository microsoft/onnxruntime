// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

using namespace std;

__global__ void cuda_add_impl(int64_t N, float* O, const float* X, const float* Y) {
  auto offset = threadIdx.x;
  if (offset < N) {
    O[offset] = Y[offset] + X[offset];
  }
}

void cuda_add(int64_t N, float* O, const float* X, const float* Y) {
  cuda_add_impl<<<1, 256>>>(N, O, X, Y);
}

