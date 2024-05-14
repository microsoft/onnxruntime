// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/optimizer/sgd/sgd_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T_WEIGHT, typename T_GRAD>
__global__ void SGDCompute(ChunkGroup<MTA_SGD_GROUP_SIZE> chunks, const float lr) {
  const int block_idx = blockIdx.x;
  T_WEIGHT* weight_chunk_ptr;
  T_GRAD* grad_chunk_ptr;
  int chunk_size;
  const int tensor_idx = chunks.block_index_to_tensor_group_index[block_idx];
  const int tensor_size = chunks.tensor_sizes[tensor_idx];
  T_WEIGHT* weight_tensor_ptr = static_cast<T_WEIGHT*>(chunks.tensor_ptrs[0][tensor_idx]);
  T_GRAD* grad_tensor_ptr = static_cast<T_GRAD*>(chunks.tensor_ptrs[1][tensor_idx]);
  const int chunk_start_idx = chunks.block_index_to_chunk_start_index[block_idx];
  // chunk_size is chunks.chunk_size if the loaded chunk is full. Otherwise (this
  // chunk is the last one in the source tensor), the actual size is determined
  // by the bound of the source tensor.
  chunk_size = min(tensor_size, chunk_start_idx + chunks.chunk_size) - chunk_start_idx;

  weight_chunk_ptr = weight_tensor_ptr + chunk_start_idx;
  grad_chunk_ptr = grad_tensor_ptr + chunk_start_idx;

#pragma unroll 4
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    float w = static_cast<float>(weight_chunk_ptr[i]);
    float g = static_cast<float>(grad_chunk_ptr[i]);
    w = w + -lr * g;
    // Update the new weight.
    weight_chunk_ptr[i] = static_cast<T_WEIGHT>(w);
  }
}

template <typename T_WEIGHT, typename T_GRAD>
void SGDMTAFunctor<T_WEIGHT, T_GRAD>::operator()(cudaStream_t stream,
                                                 ChunkGroup<MTA_SGD_GROUP_SIZE> chunks,
                                                 const float lr) {
  const int block_count = chunks.chunk_count;
  const int thread_count = ChunkGroup<MTA_SGD_GROUP_SIZE>::thread_count_per_block;
  SGDCompute<T_WEIGHT, T_GRAD><<<block_count, thread_count, 0, stream>>>(chunks, lr);
}

#define INSTANTIATE_SGD_FUNCTOR(T_WEIGHT, T_GRAD)                                                  \
  template void SGDMTAFunctor<T_WEIGHT, T_GRAD>::operator()(cudaStream_t stream,                   \
                                                            ChunkGroup<MTA_SGD_GROUP_SIZE> chunks, \
                                                            const float lr);                       \
  template __global__ void SGDCompute<T_WEIGHT, T_GRAD>(ChunkGroup<MTA_SGD_GROUP_SIZE> chunks,     \
                                                        const float lr);

INSTANTIATE_SGD_FUNCTOR(float, float)

#undef INSTANTIATE_SGD_FUNCTOR
}  // namespace cuda
}  // namespace onnxruntime
