// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/varlen_ngram_hash_mapping.h"

#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/engram_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
__device__ __forceinline__ T ReadId(const T* input_ids, const T* initial_ids,
                                    int start, int local_position, int state_length,
                                    int request, T pad_id) {
  if (local_position >= 0) {
    return input_ids[start + local_position];
  }
  const int slot = state_length + local_position;
  return slot >= 0 ? initial_ids[static_cast<int64_t>(request) * state_length + slot] : pad_id;
}

__device__ __forceinline__ int FindRequest(const int32_t* cu_seqlens, int batch_size, int token) {
  int low = 0;
  int high = batch_size;
  while (low + 1 < high) {
    const int mid = low + (high - low) / 2;
    if (cu_seqlens[mid] <= token) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return low;
}

template <typename T>
__global__ void VarlenNGramHashKernel(
    const T* input_ids, const T* multipliers, const T* vocab_sizes,
    const int32_t* cu_seqlens, const T* initial_ids, T* hash_ids,
    int batch_size, int total_tokens, int max_ngram_size,
    int n_head_per_ngram, T pad_id) {
  if (cu_seqlens[0] != 0 || cu_seqlens[batch_size] != total_tokens) {
    return;
  }
  const int state_length = max_ngram_size - 1;
  const int num_heads = state_length * n_head_per_ngram;
  const int64_t total = static_cast<int64_t>(total_tokens) * num_heads;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int head = static_cast<int>(linear % num_heads);
    const int token = static_cast<int>(linear / num_heads);
    const int request = FindRequest(cu_seqlens, batch_size, token);
    const int start = cu_seqlens[request];
    const int end = cu_seqlens[request + 1];
    if (start < 0 || start >= end || end > total_tokens || token < start || token >= end) {
      continue;
    }
    const int local_t = token - start;
    const int n = head / n_head_per_ngram + 2;
    T mix = 0;
    for (int k = 0; k < n; ++k) {
      const T id = ReadId(input_ids, initial_ids, start, local_t - k,
                          state_length, request, pad_id);
      const T product = engram_helper::WrappedMultiply<T>(id, multipliers[k]);
      mix = k == 0 ? product : static_cast<T>(mix ^ product);
    }
    const T mod = vocab_sizes[head];
    hash_ids[linear] = mod > 0 ? engram_helper::PositiveMod(mix, mod) : T{};
  }
}

template <typename T>
__global__ void VarlenNGramStateKernel(
    const T* input_ids, const int32_t* cu_seqlens, const T* initial_ids,
    T* final_ids, T* prefix_ids, int batch_size, int total_tokens,
    int state_length, int max_checkpoints, T pad_id) {
  if (cu_seqlens[0] != 0 || cu_seqlens[batch_size] != total_tokens) {
    return;
  }
  const int64_t final_total = static_cast<int64_t>(batch_size) * state_length;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < final_total; linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int slot = static_cast<int>(linear % state_length);
    const int request = static_cast<int>(linear / state_length);
    const int start = cu_seqlens[request];
    const int end = cu_seqlens[request + 1];
    if (start < 0 || start >= end || end > total_tokens) {
      continue;
    }
    const int local_length = end - start;
    final_ids[linear] = ReadId(input_ids, initial_ids, start,
                               local_length - state_length + slot,
                               state_length, request, pad_id);
    if (prefix_ids != nullptr) {
      const int64_t checkpoint_stride = static_cast<int64_t>(batch_size) * state_length;
      for (int j = 0; j < max_checkpoints && j < local_length; ++j) {
        prefix_ids[static_cast<int64_t>(j) * checkpoint_stride + linear] =
            ReadId(input_ids, initial_ids, start, j - state_length + 1 + slot,
                   state_length, request, pad_id);
      }
    }
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenNGramHashMappingKernel(
    cudaStream_t stream, const T* input_ids, const T* multipliers,
    const T* vocab_sizes, const int32_t* cumulative_sequence_length,
    const T* initial_ids, T* hash_ids, T* final_ids, T* prefix_ids,
    int batch_size, int total_tokens, int max_ngram_size,
    int n_head_per_ngram, int max_checkpoints, T pad_id) {
  const int state_length = max_ngram_size - 1;
  const int num_heads = state_length * n_head_per_ngram;
  const int64_t hash_total = static_cast<int64_t>(total_tokens) * num_heads;
  if (hash_total > 0) {
    VarlenNGramHashKernel<T><<<engram_helper::GridSize(hash_total), engram_helper::kThreads, 0, stream>>>(
        input_ids, multipliers, vocab_sizes, cumulative_sequence_length, initial_ids,
        hash_ids, batch_size, total_tokens, max_ngram_size, n_head_per_ngram, pad_id);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  const int64_t state_total = static_cast<int64_t>(batch_size) * state_length;
  if (state_total > 0) {
    VarlenNGramStateKernel<T><<<engram_helper::GridSize(state_total), engram_helper::kThreads, 0, stream>>>(
        input_ids, cumulative_sequence_length, initial_ids, final_ids, prefix_ids,
        batch_size, total_tokens, state_length, max_checkpoints, pad_id);
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE(T)                                                                              \
  template Status LaunchVarlenNGramHashMappingKernel<T>(cudaStream_t, const T*, const T*, const T*, \
                                                        const int32_t*, const T*, T*, T*, T*,       \
                                                        int, int, int, int, int, T);
INSTANTIATE(int32_t)
INSTANTIATE(int64_t)
#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
