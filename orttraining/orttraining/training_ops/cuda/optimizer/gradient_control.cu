
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"
#include "gradient_control.h"

namespace onnxruntime {
namespace cuda {
template <typename T, typename T_GRAD>
__global__ void _InPlaceAccumulator(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  accumulated_gradient[id] = gradient_buffer[id] + T(gradient[id]);
}

template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _InPlaceAccumulator<T, T_GRAD><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      gradient_buffer,
      gradient,
      accumulated_gradient,
      N);
}

#define SPECIALIZED_IMPL_InPlaceAccumulator(T, T_GRAD) \
  template void InPlaceAccumulatorImpl(                \
      const T* gradient_buffer,                        \
      const T_GRAD* gradient,                          \
      T* accumulated_gradient,                         \
      size_t count);

SPECIALIZED_IMPL_InPlaceAccumulator(float, float)
SPECIALIZED_IMPL_InPlaceAccumulator(float, half)
SPECIALIZED_IMPL_InPlaceAccumulator(half, half)
SPECIALIZED_IMPL_InPlaceAccumulator(half, float)

}  // namespace cuda
}  // namespace onnxruntime