// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/util.h"
#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::aligned_vector;

namespace onnxruntime {

template <typename T, int VecSize>
__global__ void VectorAddKernel(const T* __restrict__ x,
                                const T* __restrict__ y,
                                T* __restrict__ z, int n) {
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  using LoadT = aligned_vector<T, VecSize>;

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
Status LaunchVectorAdd(hipStream_t stream, const T* x, const T* y, T* z, int n) {
  hipLaunchKernelGGL((VectorAddKernel<T, VecSize>),
                     dim3(CeilDiv(n, ThreadsPerBlock*VecSize)),
                     dim3(ThreadsPerBlock),
                     0, stream,
                     x, y, z, n);
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, hipGetErrorName(status));
  return Status::OK();
}

}  // namespace onnxruntime
