// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include "core/providers/cuda/cuda_common.h"
#include "orttraining/training_ops/cuda/optimizer/clip_grad_norm/clip_grad_norm_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void ClipGradNorm(
    ChunkGroup<ClipGradNormGroupSize> chunks,
    const float* total_norm,
    const float epsilon,
    const float max_norm) {
  const int tensor_idx = chunks.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunks.tensor_sizes[tensor_idx];

  const int chunk_start_idx = chunks.block_index_to_chunk_start_index[blockIdx.x];
  // chunk_size is chunks.chunk_size if the loaded chunk is full. Otherwise (this
  // chunk is the last one in the source tensor), the actual size is determined
  // by the bound of the source tensor.
  const int chunk_size = min(tensor_size, chunk_start_idx + chunks.chunk_size) - chunk_start_idx;

  T* gradients_chunk_ptr = static_cast<T*>(chunks.tensor_ptrs[0][tensor_idx]) + chunk_start_idx;

#pragma unroll 4
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float clip_coefficient = max_norm / (*total_norm + epsilon);
    gradients_chunk_ptr[i] = static_cast<T>(gradients_chunk_ptr[i]) *
                             static_cast<T>(fminf(clip_coefficient, 1.0f));
  }
}

template <typename T>
void ClipGradNormFunctor<T>::operator()(
    cudaStream_t stream,
    ChunkGroup<ClipGradNormGroupSize> chunks,
    const float* total_norm,
    const float epsilon,
    const float max_norm) {
  const int num_blocks_per_grid = chunks.chunk_count;
  const int num_threads_per_block = ChunkGroup<ClipGradNormGroupSize>::thread_count_per_block;

  ClipGradNorm<T><<<num_blocks_per_grid, num_threads_per_block, 0, stream>>>(chunks, total_norm, epsilon, max_norm);
}

#define SPECIALIZE_CLIPGRADNORM_FUNCTOR(T)                                                   \
  template void ClipGradNormFunctor<T>::operator()(cudaStream_t stream,                      \
                                                   ChunkGroup<ClipGradNormGroupSize> chunks, \
                                                   const float* total_norm,                  \
                                                   const float epsilon,                      \
                                                   const float max_norm);                    \
                                                                                             \
  template __global__ void ClipGradNorm<T>(ChunkGroup<ClipGradNormGroupSize> chunks,         \
                                           const float* total_norm,                          \
                                           const float epsilon,                              \
                                           const float max_norm);

SPECIALIZE_CLIPGRADNORM_FUNCTOR(float);

#undef SPECIALIZE_CLIPGRADNORM_FUNCTOR

}  // namespace cuda
}  // namespace onnxruntime
