// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

#ifdef USE_ROCM
constexpr int kElementsPerThread = 2;
constexpr int kThreadsPerBlock = 512;
#else
constexpr int kElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kThreadsPerBlock = GridDim::maxThreadsPerBlock;
#endif

template <typename T, typename FuncT>
__global__ void ElementwiseKernel(T* output_data, const FuncT functor, uint32_t N) {
  uint32_t start = kElementsPerThread * kThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[kElementsPerThread];

  uint32_t id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      value[i] = functor(id);
      id += kThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < kElementsPerThread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += kThreadsPerBlock;
    }
  }
}

template <typename T, typename FuncT>
void LaunchElementwiseKernel(cudaStream_t stream, T* output_data, const FuncT& functor, size_t output_size) {
  if (output_size == 0) return;
  uint32_t N = static_cast<uint32_t>(output_size);
  uint32_t blocksPerGrid = CeilDiv(N, kThreadsPerBlock * kElementsPerThread);
  ElementwiseKernel<T, FuncT><<<blocksPerGrid, kThreadsPerBlock, 0, stream>>>(output_data, functor, N);
}

}  // namespace cuda
}  // namespace onnxruntime
