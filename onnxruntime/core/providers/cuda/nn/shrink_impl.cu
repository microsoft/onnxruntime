// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "shrink_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _ShrinkKernel(
    const T* input_data,
    const T bias,
    const T lambda,
    T* output_data,
    const CUDA_LONG N) {

  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  T x = input_data[id];
  if (x < -lambda) {
    output_data[id] = x - bias;    
  } else if (x > lambda) {
    output_data[id] = x + bias;
  } else {
    output_data[id] = (T)0;
  }

}

template <typename T>
void ShrinkImpl(
    const T* input_data,
    const T bias,
    const T lambda,
    T* output_data,
    size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _ShrinkKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      input_data, bias, lambda, output_data, (CUDA_LONG) N);
}

#define SPECIALIZED_IMPL(T) \
  template void ShrinkImpl<T>(const T* input_data, const T bias, const T lambda, T* output_data, size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
SPECIALIZED_IMPL(uint8_t)
SPECIALIZED_IMPL(int8_t)
SPECIALIZED_IMPL(uint16_t)
SPECIALIZED_IMPL(int16_t)
SPECIALIZED_IMPL(uint32_t)
SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(int64_t)

}  // namespace cuda
}  // namespace onnxruntime
