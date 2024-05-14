// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/tunable/util.h"
#elif USE_ROCM
#include <hip/hip_runtime.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/util.h"
#endif

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

#if USE_CUDA
using onnxruntime::cuda::aligned_vector;
using onnxruntime::cuda::CeilDiv;
using StreamT = cudaStream_t;
#elif USE_ROCM
using onnxruntime::rocm::aligned_vector;
using onnxruntime::rocm::CeilDiv;
using StreamT = hipStream_t;
#endif

namespace onnxruntime {

template <typename T, int VecSize>
__global__ void VectorAddKernel(const T* __restrict__ x,
                                const T* __restrict__ y,
                                T* __restrict__ z, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
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
Status LaunchVectorAdd(StreamT stream, const T* x, const T* y, T* z, int n) {
  VectorAddKernel<T, VecSize>
      <<<dim3(CeilDiv(n, ThreadsPerBlock * VecSize)), dim3(ThreadsPerBlock), 0, stream>>>(x, y, z, n);
#if USE_CUDA
  return CUDA_CALL(cudaGetLastError());
#elif USE_ROCM
  return HIP_CALL(hipGetLastError());
#endif
}

}  // namespace onnxruntime
