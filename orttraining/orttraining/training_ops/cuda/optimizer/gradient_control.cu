
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
    cudaStream_t stream,
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _InPlaceAccumulator<T, T_GRAD><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      gradient_buffer,
      gradient,
      accumulated_gradient,
      N);
}

#define SPECIALIZED_IMPL_InPlaceAccumulator(T, T_GRAD) \
  template void InPlaceAccumulatorImpl(                \
      cudaStream_t stream,                       \
      const T* gradient_buffer,                        \
      const T_GRAD* gradient,                          \
      T* accumulated_gradient,                         \
      size_t count);

SPECIALIZED_IMPL_InPlaceAccumulator(float, float)
SPECIALIZED_IMPL_InPlaceAccumulator(float, half)
SPECIALIZED_IMPL_InPlaceAccumulator(half, half)
SPECIALIZED_IMPL_InPlaceAccumulator(half, float)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
SPECIALIZED_IMPL_InPlaceAccumulator(float, nv_bfloat16)
SPECIALIZED_IMPL_InPlaceAccumulator(nv_bfloat16, nv_bfloat16)
SPECIALIZED_IMPL_InPlaceAccumulator(nv_bfloat16, float)
#endif

}  // namespace cuda
}  // namespace onnxruntime