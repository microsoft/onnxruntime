
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_control_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {

// This number is tuned among many data scales (listed in unit tests) that can fit one GPU card.
constexpr size_t large_data_scale_threshold = 4096 * 768;

}  // namespace

template <typename T, typename T_GRAD, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _InPlaceAccumulator(
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  // We see slightly perf degradation splitting read/write for this case.
  // So we handle it differently here.
  if (NumElementsPerThread == 1) {
    if (start < N)
      accumulated_gradient[start] = gradient_buffer[start] + T(gradient[start]);
    return;
  }

  T lvalue[NumElementsPerThread];
  T rvalue[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      lvalue[i] = gradient_buffer[id];
      rvalue[i] = T(gradient[id]);

      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      accumulated_gradient[id] = lvalue[i] + rvalue[i];

      id += NumThreadsPerBlock;
    }
  }
}

template <typename T, typename T_GRAD>
void InPlaceAccumulatorImpl(
    cudaStream_t stream,
    const T* gradient_buffer,
    const T_GRAD* gradient,
    T* accumulated_gradient,
    size_t count) {
  if (count == 0)
    return;

  const int num_threads_per_block = GridDim::maxThreadsPerBlock;
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  if (count < large_data_scale_threshold) {
    const int num_elements_per_thread = 1;
    const int blocksPerGrid = static_cast<int>(CeilDiv(count, num_threads_per_block * num_elements_per_thread));
    _InPlaceAccumulator<T, T_GRAD, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            gradient_buffer,
            gradient,
            accumulated_gradient,
            N);
  } else {
    const int num_elements_per_thread = GridDim::maxElementsPerThread;
    const int blocksPerGrid = static_cast<int>(CeilDiv(count, num_threads_per_block * num_elements_per_thread));
    _InPlaceAccumulator<T, T_GRAD, num_threads_per_block, num_elements_per_thread>
        <<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
            gradient_buffer,
            gradient,
            accumulated_gradient,
            N);
  }
}

#define SPECIALIZED_IMPL_INPLACEACCUMULATOR(T, T_GRAD)                                                        \
  template void InPlaceAccumulatorImpl(cudaStream_t stream, const T* gradient_buffer, const T_GRAD* gradient, \
                                       T* accumulated_gradient, size_t count);

SPECIALIZED_IMPL_INPLACEACCUMULATOR(float, float)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(float, half)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(half, half)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(half, float)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(float, BFloat16)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(BFloat16, BFloat16)
SPECIALIZED_IMPL_INPLACEACCUMULATOR(BFloat16, float)

}  // namespace cuda
}  // namespace onnxruntime
