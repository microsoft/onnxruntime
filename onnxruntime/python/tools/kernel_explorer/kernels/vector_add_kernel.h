// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "device_array.h"
#include "operator.h"

// aligned vector for vectorized load/store
template<typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) AlignedVector {
  T val[VecSize];
};

template <typename T, int VecSize>
__global__ void VectorAddKernel(const T* __restrict__ x,
                                  const T* __restrict__ y,
                                  T* __restrict__ z, int n) {
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  using LoadT = AlignedVector<T, VecSize>;

  if (VecSize * i + VecSize - 1 < n) {
    T x_vec[VecSize];
    LoadT* x_load = reinterpret_cast<LoadT*>(&x_vec);
    *x_load = *reinterpret_cast<const LoadT*>(&x[VecSize * i]);

    T y_vec[VecSize];
    LoadT* y_load = reinterpret_cast<LoadT*>(&y_vec);
    *y_load = *reinterpret_cast<const LoadT*>(&y[VecSize * i]);

    T z_vec[VecSize];

    #pragma unroll
    for (int j = 0; j < VecSize; j++) {
      z_vec[j] = x_vec[j] + y_vec[j];
    }

    *(reinterpret_cast<LoadT*>(&z[VecSize * i])) = *reinterpret_cast<LoadT*>(&z_vec[0]);
  }

  if (i == 0) {
    int tail_size = n % VecSize;
    for (int j = n - 1; j >= n - tail_size; j--) {
      z[j] = x[j] + y[j];
    }
  }
}

template <typename T, int ThreadsPerBlock, int VecSize>
void LaunchVectorAdd(const T* x, const T* y, T* z, int n) {
  hipLaunchKernelGGL((VectorAddKernel<T, VecSize>), 
                  dim3(ceil(float(n)/(float(ThreadsPerBlock)*VecSize))),
                  dim3(ThreadsPerBlock),
                  0, 0,
                  x, y, z, n);
}
