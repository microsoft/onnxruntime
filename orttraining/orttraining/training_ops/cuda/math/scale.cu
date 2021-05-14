// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/math/scale.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Scale(
    const T* input_data,
    const T scale_value,
    T* output_data,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T input_value[NumElementsPerThread];
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
        input_value[i] = input_data[id];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = input_value[i] * scale_value;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void Impl_Scale(
    cudaStream_t stream,
    const T* input_data,
    const float scale_value,
    T* output_data,
    size_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _Scale<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data,
      static_cast<T>(scale_value),
      output_data,
      N);
}

#define SPECIALIZE_SCALE_IMPL(T)        \
template void Impl_Scale<T>(            \
    cudaStream_t stream,                \
    const T* input_data,                \
    const float scale_value,            \
    T* output_data,                     \
    size_t count);

SPECIALIZE_SCALE_IMPL(half)
SPECIALIZE_SCALE_IMPL(float)
SPECIALIZE_SCALE_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime
