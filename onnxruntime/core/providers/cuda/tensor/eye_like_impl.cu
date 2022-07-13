// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "eye_like_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _EyeLikeKernel(
    size_t offset,
    size_t stripe,
    T* output_data,
    CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  // offset is the first elements, stripe is width + 1.
  output_data[offset + id * stripe] = static_cast<T>(1);
}

template <typename T>
void EyeLikeImpl(
    cudaStream_t stream,
    size_t offset,
    size_t stripe,
    T* output_data,
    size_t diag_count) {
  constexpr int block_size = 256;
  int blocksPerGrid = (int)(ceil(static_cast<float>(diag_count) / block_size));
  CUDA_LONG N = static_cast<CUDA_LONG>(diag_count);

  _EyeLikeKernel<<<blocksPerGrid, block_size, 0, stream>>>(offset, stripe, output_data, N);
}

#define SPECIALIZED_IMPL(T)                                          \
  template void EyeLikeImpl<T>(                                      \
    cudaStream_t stream,                                       \
    size_t offset,                                                   \
    size_t stripe,                                                   \
    T* output_data,                                                  \
    size_t diag_count);

SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(uint64_t)
SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)

}  // namespace cuda
}  // namespace onnxruntime