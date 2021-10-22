// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

using namespace std;

template <typename T1, typename T2, typename T3>
__global__ void cuda_add_impl(int64_t N, T3* O, const T1* X, const T2* Y) {
  auto offset = threadIdx.x;
  if (offset < N) {
    O[offset] = Y[offset] + X[offset];
  }
}

template <typename T1, typename T2, typename T3>
void cuda_add(int64_t N, T3* O, const T1* X, const T2* Y, cudaStream_t compute_stream) {
  cuda_add_impl<<<1, 256, 0, compute_stream>>>(N, O, X, Y);
}

template <typename T>
__global__ void cuda_slice_impl(const T* X, int64_t from, int64_t to, T* Y) {
  auto offset = threadIdx.x;
  if (offset >= from && offset < to) {
    Y[offset - from] = X[offset];
  }
}

template <typename T>
void cuda_slice(const T* X, int64_t from, int64_t to, T* Y, cudaStream_t compute_stream) {
  cuda_slice_impl<T><<<1, 256, 0, compute_stream>>>(X, from, to, Y);
}

template void cuda_slice(const float*, int64_t, int64_t, float*, cudaStream_t compute_stream);
template void cuda_slice(const double*, int64_t, int64_t, double*, cudaStream_t compute_stream);

template void cuda_add(int64_t, float*, const float*, const float*, cudaStream_t compute_stream);
template void cuda_add(int64_t, float*, const float*, const double*, cudaStream_t compute_stream);
