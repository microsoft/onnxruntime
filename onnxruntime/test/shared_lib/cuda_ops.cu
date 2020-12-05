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
__global__ void cuda_mul_impl(const T* X , const T* Y, T* Z, int64_t size) {
  auto offset = threadIdx.x;
  if (offset < size) {
    Z[offset] = X[offset] * Y[offset];
  }
}

template<typename T>
void cuda_mul(const T* X , const T* Y, T* Z, int64_t size)
{
    cuda_mul_impl<T><<<1, 256>>>(X, Y, Z, size);
}

template void cuda_mul(const float*, const float*, float*, int64_t);
template void cuda_mul(const double*, const double*, double*, int64_t);
