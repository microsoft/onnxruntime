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

template<typename T>
__global__ void cuda_slice_impl(const T* X , int64_t from, int64_t to, T* Y) {
  auto offset = threadIdx.x;
  if (offset >= from && offset < to) {
    Y[offset - from] = X[offset];
  }
}

template<typename T>
void cuda_slice(const T* X, int64_t from, int64_t to, T* Y) {
    cuda_slice_impl<T><<<1, 256>>>(X, from, to, Y);
}

template void cuda_slice(const float*, int64_t, int64_t, float*);
template void cuda_slice(const double*, int64_t, int64_t, double*);
