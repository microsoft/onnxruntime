//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// NVIDIA/apex is licensed under the
// BSD 3 - Clause "New" or "Revised" License
//

/* Modifications Copyright (c) Microsoft. */

#pragma once
#include <vector>

namespace onnxruntime {
namespace cuda {
// initial reference from:
// https://github.com/NVIDIA/apex/blob/5b71d3695bf39efcdcda9dff5be2f70314b8f091/csrc/multi_tensor_apply.cuh#L15
// further experiment to get the number below. The larger the better, but if too large, it won't fit into GPU stack.
constexpr int ACTUAL_TENSOR_GROUP_SIZE[8] = {1, 1, 2, 3, 4, 5, 6, 7};
constexpr int MAX_BLOCK_COUNTS[8] = {256, 320, 320, 320, 320, 288, 288, 256};
constexpr int MAX_TENSOR_GROUP_COUNTS[8] = {1, 96, 64, 32, 32, 32, 32, 32};
constexpr int MAX_BLOCK_THREAD_COUNTS[8] = {256, 512, 512, 512, 512, 512, 512, 512};

// TensorGroupSize is the number of parallel tensors. For element-wise
// operators such as Relu, it should be 1. For two-operand operators such as
// element-wise addition, it should be 2. The value 0 is reserved for implementing
// kernels to handle a single large tensor.
template <int TensorGroupSize>
struct ChunkGroup {
  // Number of chunks in this ChunkGroup.
  // It's the effective size of block_index_to_tensor_group_index and
  // block_index_to_chunk_start_index.
  // The i-th chunk starts at the block_index_to_chunk_start_index[i]-th
  // element in the block_index_to_tensor_group_index[i]-th tensor.
  int chunk_count = 0;
  // Max number of elements in each chunk in this ChunkGroup.
  // It's an upper bound because chunks locating in the end of tensors
  // are not always full. For example, if we split a 7-element vector into
  // two 4-element chunks, the second chunk may contain only 3 actual values.
  int chunk_size = 0;
  // blkIdx.x block processes chunks in block_index_to_tensor_group_index[blkIdx.x]-th
  // tensor group. Each chunk starts from block_index_to_chunk_start_index[blkIdx.x]-th
  // element until reaching the end of this chunk or the end of the whole tensor.
  //
  // Let i = block_index_to_tensor_group_index[blkIdx.x]
  //     n = tensor_sizes[i]
  //     b = block_index_to_chunk_start_index[blkIdx.x]
  //     e = min(b + chunk_size, n)
  // The valid index range for blockIdx.x is defined by the following equation.
  //     b <= valid index < e
  int block_index_to_tensor_group_index[MAX_BLOCK_COUNTS[TensorGroupSize]];
  int block_index_to_chunk_start_index[MAX_BLOCK_COUNTS[TensorGroupSize]];
  int tensor_sizes[MAX_TENSOR_GROUP_COUNTS[TensorGroupSize]];
  // The addresses of tensors where the chunks are extracted from.
  // 1. tensor_ptrs[0][i], ..., tensor_ptrs[TensorGroupSize-1][i] are
  //    the tensors' pointers in the i-th group.
  // 2. All tensors in the i-th group have the same size, tensor_sizes[i].
  void* tensor_ptrs[ACTUAL_TENSOR_GROUP_SIZE[TensorGroupSize]][MAX_TENSOR_GROUP_COUNTS[TensorGroupSize]];
  // Max number of GPU blocks to process the chunks in this chunk group.
  const static int max_block_count = MAX_BLOCK_COUNTS[TensorGroupSize];
  // Max number of tensor groups in this chunk group.
  const static int max_tensor_group_count = MAX_TENSOR_GROUP_COUNTS[TensorGroupSize];
  // The suggested number of threads to launch per GPU block.
  const static int thread_count_per_block = MAX_BLOCK_THREAD_COUNTS[TensorGroupSize];
};

template <int TensorGroupSize>
int compute_max_tensor_size_per_launch(int element_count_per_thread) {
  constexpr int block_count =
      ChunkGroup<TensorGroupSize>::max_block_count;
  constexpr int thread_count_per_block =
      ChunkGroup<TensorGroupSize>::thread_count_per_block;
  return block_count * thread_count_per_block * element_count_per_thread;
}

template <int TensorGroupSize, typename TMultiTensorFunctor, typename... TFunctorParams>
void launch_multi_tensor_functor(
    cudaStream_t stream,
    const int chunk_size,
    std::vector<int>& tensor_sizes,
    std::vector<std::vector<void*>>& grouped_tensor_pointers,
    TMultiTensorFunctor multipleTensorKernel,
    TFunctorParams&&... kernelParams) {
  ORT_ENFORCE(tensor_sizes.size() > 0);
  ORT_ENFORCE(tensor_sizes.size() < static_cast<size_t>(std::numeric_limits<int>::max()));
  ORT_ENFORCE(grouped_tensor_pointers.size() > 0);
  ORT_ENFORCE(grouped_tensor_pointers.size() < static_cast<size_t>(std::numeric_limits<int>::max()));
  ORT_ENFORCE(chunk_size > 0);
  // Number of groups, for example, the number of updated weight tensors in Lamb optimizer.
  const int group_count = static_cast<int>(grouped_tensor_pointers.size());
  // Tensor count per group.
  const int group_size = static_cast<int>(grouped_tensor_pointers[0].size());
  int tensor_group_index = 0;
  int block_index = 0;

  // Check if 32-bit integer is enough.
  ORT_ENFORCE(tensor_sizes.size() < static_cast<size_t>(std::numeric_limits<int>::max()));
  ORT_ENFORCE(grouped_tensor_pointers.size() == tensor_sizes.size());
  ORT_ENFORCE(group_size == ACTUAL_TENSOR_GROUP_SIZE[TensorGroupSize]);
  for (int i = 0; i < group_count; ++i) {
    ORT_ENFORCE(grouped_tensor_pointers[i].size() == static_cast<size_t>(group_size));
  }

  // Handle multiple tensors per CUDA kernel call.
  ChunkGroup<TensorGroupSize> chunk_group;
  for (int i = 0; i < group_count; ++i) {
    // Add pointers to one group of tensors into chunk_group.
    for (int j = 0; j < group_size; ++j) {
      chunk_group.tensor_ptrs[j][tensor_group_index] = grouped_tensor_pointers[i][j];
    }

    // Assuming that all tensors' shapes are the same, we just record w's size.
    chunk_group.tensor_sizes[tensor_group_index] = tensor_sizes[i];
    chunk_group.chunk_size = chunk_size;

    const int chunk_count = (tensor_sizes[i] + chunk_size - 1) / chunk_size;

    // Process all chunks in this tensor group.
    for (int chunk_index = 0; chunk_index < chunk_count; ++chunk_index) {
      chunk_group.block_index_to_tensor_group_index[block_index] = tensor_group_index;
      chunk_group.block_index_to_chunk_start_index[block_index] = chunk_index * chunk_size;
      // After ++block_index, block_index becomes the count of chunks in chunk_group.
      ++block_index;
      chunk_group.chunk_count = block_index;

      if (block_index == chunk_group.max_block_count) {
        multipleTensorKernel(stream, chunk_group, std::forward<TFunctorParams>(kernelParams)...);
        block_index = 0;
      }
    }

    // After ++tensor_group_index, tensor_group_index becomes the count of tensor group in chunk_group.
    ++tensor_group_index;
    if (tensor_group_index == chunk_group.max_tensor_group_count) {
      multipleTensorKernel(stream, chunk_group, std::forward<TFunctorParams>(kernelParams)...);
      block_index = 0;
      tensor_group_index = 0;
    }
  }

  // This round of processing tensor group is finished.
  // All the groups remain in chunk group should be processed right now.
  if (block_index != 0) {
    multipleTensorKernel(stream, chunk_group, std::forward<TFunctorParams>(kernelParams)...);
    block_index = 0;
    tensor_group_index = 0;
  }
}

}  // namespace cuda
}  // namespace onnxruntime