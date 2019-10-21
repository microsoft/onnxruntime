// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "eye_like_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _EyeLikeKernel(
    const int64_t k,
    const fast_divmod fdm_x,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int x, y;
  fdm_x.divmod(id, x, y);
  T value = static_cast<T>(0);

  if (x + k == y)
    value = static_cast<T>(1);

  output_data[id] = value;
}

template <typename T>
void EyeLikeImpl(
    const int64_t k,
    const fast_divmod& fdm_x,
    T* output_data,
    size_t count) {
  constexpr int block_size = 256;
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / block_size));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);

  _EyeLikeKernel<<<blocksPerGrid, block_size, 0>>>(k, fdm_x, output_data, N);
}

#define SPECIALIZED_IMPL(T)                                          \
  template void EyeLikeImpl<T>(                                      \
    const int64_t k,                                                 \
    const fast_divmod& fdm_x,                                        \
    T* output_data,                                                  \
    size_t count);

SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime