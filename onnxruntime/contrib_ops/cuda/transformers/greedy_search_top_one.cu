// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "greedy_search_top_one.h"

#include <cub/cub.cuh>

#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
struct TopOne {
  int32_t key;
  T value;

  __device__ __host__ __forceinline__ TopOne(int32_t key = -1, T value = NumericLimits<T>::Min()) : key(key), value(value) {
  }

  __device__ __forceinline__ void Reduce(int32_t k, T v) {
    if (value < v || key == -1) {
      key = k;
      value = v;
    }
  }
};

template <typename T>
__device__ __forceinline__ TopOne<T> ReduceTopOneOp(const TopOne<T>& a, const TopOne<T>& b) {
  if ((a.value > b.value) || (a.value == b.value && a.key != -1 && a.key < b.key)) {
    return a;
  }

  return b;
}

// kernel to compute the top 1 on last axis for tensor with shape[batch, parts_of_vocab, vacab_part_size],
// and produce a tensor with shape [batch, parts_of_vocab]
// Its grid is [batch, parts_of_vocab]
template <typename T, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void GreedySearchTopOneStage1Kernel(
    const T* input,
    int32_t vocab_size,
    int32_t vocab_part_size,
    T* output_values,
    int32_t* output_token) {
  TopOne<T> top_one_thread;

  int batch = blockIdx.x;
  int voc_part_id = blockIdx.y;

  int token_id_base = voc_part_id * vocab_part_size;
  const T* input_block = input + batch * vocab_size;
  // voc_part_size
  for (int vocab_idx = threadIdx.x + token_id_base;
       vocab_idx < vocab_part_size + token_id_base;
       vocab_idx += blockDim.x) {
    if (vocab_idx < vocab_size) {
      top_one_thread.Reduce(vocab_idx, input_block[vocab_idx]);
    }
  }

  // reduce in thread block
  typedef cub::BlockReduce<TopOne<T>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopOne<T> top_one_block = BlockReduce(temp_storage).Reduce(top_one_thread, ReduceTopOneOp<T>);
  if (threadIdx.x == 0) {
    output_values[batch * gridDim.y + voc_part_id] = top_one_block.value;
    output_token[batch * gridDim.y + voc_part_id] = top_one_block.key;
  }
}

// kernel to compute the top 1 on last axis for tensor with shape[batch, parts_of_vocab],
// and produce a tensor with shape [batch]
// Its grid is [batch]
template <typename T, int thread_block_size>
__launch_bounds__(thread_block_size) __global__ void GreedySearchTopOneStage2Kernel(
    const T* input_values,
    const int32_t* input_tokens,
    int32_t vocab_size,
    int32_t vocab_parts,
    T* output_values,
    int32_t* output_tokens) {
  const int batch_id = blockIdx.x;
  const int thread_id = threadIdx.x;

  input_values += batch_id * vocab_parts;
  input_tokens += batch_id * vocab_parts;

  TopOne<T> thread_top_one;
  for (int idx = thread_id; idx < vocab_parts; idx += thread_block_size) {
    thread_top_one.Reduce(input_tokens[idx], input_values[idx]);
  }

  typedef cub::BlockReduce<TopOne<T>, thread_block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  TopOne<T> top_one_block = BlockReduce(temp_storage).Reduce(thread_top_one, ReduceTopOneOp<T>);
  if (thread_id == 0) {
    output_values[batch_id] = top_one_block.value;
    output_tokens[batch_id] = top_one_block.key;
  }
}

template <typename T>
void GreedySearchTopOne(
    const T* input,
    int32_t batch_size,
    int32_t vocab_size,
    T* tmp_values,
    int32_t* tmp_tokens,
    T* output_values,
    int32_t* output_tokens,
    cudaStream_t stream) {
  constexpr int kThreadBlockSize = GridDim::maxThreadsPerBlock;

  int voc_parts = 4;
  if (batch_size < 256) {
    voc_parts = (240 + batch_size - 1) / batch_size;
    voc_parts = std::min(128, voc_parts);  // we implement up to 128
  }

  dim3 stage1_grid(batch_size, voc_parts);
  GreedySearchTopOneStage1Kernel<T, kThreadBlockSize><<<stage1_grid, kThreadBlockSize, 0, stream>>>(
      input,
      vocab_size,
      (vocab_size + voc_parts - 1) / voc_parts,
      tmp_values,
      tmp_tokens);

  constexpr int KThreadBlockSizeStage2 = 128;
  GreedySearchTopOneStage2Kernel<T, KThreadBlockSizeStage2><<<batch_size, KThreadBlockSizeStage2, 0, stream>>>(
      tmp_values,
      tmp_tokens,
      vocab_size,
      voc_parts,
      output_values,
      output_tokens);
}

template void GreedySearchTopOne(
    const float* input,
    int32_t batch_size,
    int32_t vocab_size,
    float* tmp_values,
    int32_t* tmp_tokens,
    float* output_values,
    int32_t* output_tokens,
    cudaStream_t stream);

template void GreedySearchTopOne(
    const half* input,
    int32_t batch_size,
    int32_t vocab_size,
    half* tmp_values,
    int32_t* tmp_tokens,
    half* output_values,
    int32_t* output_tokens,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
