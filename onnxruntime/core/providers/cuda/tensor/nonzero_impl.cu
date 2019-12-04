// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nonzero_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include <cub/cub.cuh>

namespace onnxruntime {
namespace cuda {

static const int NONZERO_THREADS_PER_BLOCK = GridDim::maxThreadsPerBlock;

int NonZeroCalcBlockCount(int64_t x_size) {
  return CeilDiv(x_size, NONZERO_THREADS_PER_BLOCK);
}

cudaError_t NonZeroCalcPrefixSumTempStorageBytes(
    int* prefix_counts, int number_of_blocks, size_t& temp_storage_bytes) {
  temp_storage_bytes = 0;
  return cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, prefix_counts, prefix_counts, number_of_blocks);
}

cudaError_t NonZeroInclusivePrefixSum(
    void* d_temp_storage, size_t temp_storage_bytes, int* prefix_counts, int number_of_blocks) {
  return cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, prefix_counts, prefix_counts, number_of_blocks);
}

template <typename InputT, int THREADS_PER_BLOCK>
__global__ void NonZeroCountEachBlockKernel(const InputT* x, int64_t x_size, int* count_in_blocks) {
  typedef cub::BlockReduce<int, THREADS_PER_BLOCK, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceT;
  __shared__ typename BlockReduceT::TempStorage temp_storage;

  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const cub::CastOp<bool> cast_to_bool;
  int nz = 0;
  if (index < x_size && cast_to_bool(x[index])) ++nz;
  int count = BlockReduceT(temp_storage).Sum(nz);

  if (threadIdx.x == 0) {
    count_in_blocks[blockIdx.x] = count;
  }
}

template <typename InputT, int THREADS_PER_BLOCK>
__global__ void NonZeroOutputPositionsKernel(
    const InputT* x, int64_t x_size, int x_rank, const fast_divmod* x_strides,
    const int* prefix_counts, int nonzero_elements, int64_t* results) {
  typedef cub::BlockScan<int, THREADS_PER_BLOCK> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const cub::CastOp<bool> cast_to_bool;
  int nz = 0;
  if (index < x_size && cast_to_bool(x[index])) ++nz;
  int pos_in_block = 0;
  BlockScanT(temp_storage).InclusiveSum(nz, pos_in_block);

  int result_position = ((blockIdx.x == 0) ? 0 : prefix_counts[blockIdx.x - 1]) + pos_in_block - nz;

  if (index < x_size && cast_to_bool(x[index])) {
    int remain = (int)index, dim = 0;
    for (int axis = 0, rp = result_position; axis < x_rank; ++axis, rp += nonzero_elements) {
      x_strides[axis].divmod(remain, dim, remain);
      results[rp] = (int64_t)dim;
    }
  }
}

template <typename InputT>
cudaError_t NonZeroCountEachBlock(const InputT* x, int64_t x_size, int* count_in_blocks) {
  int num_blocks = NonZeroCalcBlockCount(x_size);
  NonZeroCountEachBlockKernel<InputT, NONZERO_THREADS_PER_BLOCK><<<num_blocks, NONZERO_THREADS_PER_BLOCK>>>(
      x, x_size, count_in_blocks);
  return cudaSuccess;
}

template <typename InputT>
cudaError_t NonZeroOutputPositions(
    const InputT* x, int64_t x_size, int x_rank, const fast_divmod* x_strides,
    const int* prefix_counts, int nonzero_elements, int64_t* results) {
  int num_blocks = NonZeroCalcBlockCount(x_size);
  NonZeroOutputPositionsKernel<InputT, NONZERO_THREADS_PER_BLOCK><<<num_blocks, NONZERO_THREADS_PER_BLOCK>>>(
      x, x_size, x_rank, x_strides,
      prefix_counts, nonzero_elements, results);
  return cudaSuccess;
}

template cudaError_t NonZeroCountEachBlock(const bool*, int64_t, int*);
template cudaError_t NonZeroCountEachBlock(const uint8_t*, int64_t, int*);
template cudaError_t NonZeroCountEachBlock(const int64_t*, int64_t, int*);
template cudaError_t NonZeroCountEachBlock(const int32_t*, int64_t, int*);
template cudaError_t NonZeroCountEachBlock(const float*, int64_t, int*);
template cudaError_t NonZeroCountEachBlock(const half*, int64_t, int*);

template cudaError_t NonZeroOutputPositions(const bool*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);
template cudaError_t NonZeroOutputPositions(const uint8_t*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);
template cudaError_t NonZeroOutputPositions(const int64_t*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);
template cudaError_t NonZeroOutputPositions(const int32_t*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);
template cudaError_t NonZeroOutputPositions(const float*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);
template cudaError_t NonZeroOutputPositions(const half*, int64_t, int, const fast_divmod*, const int*, int, int64_t*);

}  // namespace cuda
}  // namespace onnxruntime
