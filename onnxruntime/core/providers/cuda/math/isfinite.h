// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cuda/cudnn_common.h"

constexpr int PARALLEL_LOADS = 4;
constexpr int WARP_THREAD_COUNT = 32;
constexpr int MAX_BLOCK_COUNT = 288;
constexpr int MAX_TENSOR_COUNT = 128;
constexpr int MAX_BLOCK_THREAD_COUNT = 512;

namespace onnxruntime {
namespace cuda {

template <typename TSrc>
class IsFiniteOp final : public CudaKernel {
 public:
  IsFiniteOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename TSrc>
void IsFinite(const TSrc* input, bool* output, size_t N);

template <typename TSrc>
class IsAllFiniteOp final : public CudaKernel {
 public:
  IsAllFiniteOp(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
struct ChunkGroup {
  // Number of chunks in this ChunkGroup.
  // It's the effective size of block_index_to_tensor_index and
  // block_index_to_chunk_start_index.
  // The i-th chunk starts at the block_index_to_chunk_start_index[i]-th
  // element in the block_index_to_tensor_index[i]-th tensor.
  int chunk_count;
  // Max number of elements in each chunk in this ChunkGroup.
  // It's an upper bound because chunks locating in the end of tensors
  // are not always full. For example, if we split a 7-element vector into
  // two 4-element chunks, the second chunk may contain only 3 actual values.
  int chunk_size;
  // blkIdx.x block processes block_index_to_tensor_index[blkIdx.x]-th tensor's
  // elements starting from block_index_to_chunk_start_index[blkIdx.x] until
  // reaching the end of this chunk or the end of the whole tensor.
  //
  // Let i = block_index_to_tensor_index[blkIdx.x]
  //     n = tensor_sizes[i]
  //     b = block_index_to_chunk_start_index[blkIdx.x]
  //     e = min(b + chunk_size, n)
  // The valid index range for blockIdx.x is defined by the following equation.
  //     b <= valid index < e
  int block_index_to_tensor_index[MAX_BLOCK_COUNT];
  int block_index_to_chunk_start_index[MAX_BLOCK_COUNT];
  int tensor_sizes[MAX_TENSOR_COUNT];
  // The addresses of tensors where the chunks are extracted from.
  const T* tensor_ptrs[MAX_TENSOR_COUNT];
};

template <typename T>
void IsAllFinite(const ChunkGroup<T> chunks, bool* output);

}  // namespace cuda
}  // namespace onnxruntime