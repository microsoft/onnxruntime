// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(
    T* output_data,
    T val,
    CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T* output, T value, int64_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _Fill<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, value, N);
}

#define SPECIALIZED_FILL(T) \
  template void Fill<T>(cudaStream_t stream, T * output, T value, int64_t count);

SPECIALIZED_FILL(int8_t)
SPECIALIZED_FILL(bool)
SPECIALIZED_FILL(int16_t)
SPECIALIZED_FILL(int32_t)
SPECIALIZED_FILL(int64_t)
SPECIALIZED_FILL(float)
SPECIALIZED_FILL(double)
SPECIALIZED_FILL(__half)
SPECIALIZED_FILL(BFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
SPECIALIZED_FILL(Float8E4M3FN)
SPECIALIZED_FILL(Float8E5M2)
#endif

}  // namespace cuda
}  // namespace onnxruntime
