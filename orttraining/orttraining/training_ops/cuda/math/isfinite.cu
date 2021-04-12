// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "orttraining/training_ops/cuda/math/isfinite.cuh"

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
__global__ void _IsFinite(const TSrc* input, bool* output, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = _IsFiniteScalar(input[id]);
}

template <typename TSrc>
void IsFinite(cudaStream_t stream, const TSrc* input, bool* output, size_t count) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _IsFinite<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input, output, N);
}

#define SPECIALIZE_ISFINITE_IMPL(T) \
template void IsFinite(cudaStream_t stream, const T* input, bool* output, size_t count);

SPECIALIZE_ISFINITE_IMPL(half)
SPECIALIZE_ISFINITE_IMPL(float)
SPECIALIZE_ISFINITE_IMPL(double)

template <typename TSrc>
__global__ void IsAllFiniteMultiTensorImpl(ChunkGroup<1> chunks, bool* output) {
  const int block_idx = blockIdx.x;
  const int tensor_idx = chunks.block_index_to_tensor_group_index[block_idx];
  const int tensor_size = chunks.tensor_sizes[tensor_idx];
  const TSrc* tensor_ptr = static_cast<TSrc*>(chunks.tensor_ptrs[0][tensor_idx]);
  const int chunk_start_idx = chunks.block_index_to_chunk_start_index[block_idx];
  // chunk_size is chunks.chunk_size if the loaded chunk is full. Otherwise (this
  // chunk is the last one in the source tensor), the actual size is determined
  // by the bound of the source tensor.
  const int chunk_size = min(tensor_size, chunk_start_idx + chunks.chunk_size) - chunk_start_idx;

  const TSrc* chunk_ptr = tensor_ptr + chunk_start_idx;
  bool result = true;
#pragma unroll(4)
  for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
    result &= _IsFiniteScalar(chunk_ptr[i]);
  }

  if (!result) {
    *output = false;
  }
}

template <typename T>
void IsAllFiniteFunctor<T>::operator()(cudaStream_t stream, ChunkGroup<1> chunks, bool* output) {
  const int block_count = chunks.chunk_count;
  const int thread_count = ChunkGroup<1>::thread_count_per_block;
  IsAllFiniteMultiTensorImpl<T><<<block_count, thread_count, 0, stream>>>(chunks, output);
}

#define INSTANTIATE_ISALLFINITE_FUNCTOR(T) \
  template void IsAllFiniteFunctor<T>::operator()(cudaStream_t stream, ChunkGroup<1> chunks, bool* output);

INSTANTIATE_ISALLFINITE_FUNCTOR(half)
INSTANTIATE_ISALLFINITE_FUNCTOR(float)
INSTANTIATE_ISALLFINITE_FUNCTOR(double)

}  // namespace cuda
}  // namespace onnxruntime