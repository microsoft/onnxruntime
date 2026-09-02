// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/ngram_hash_mapping_impl.h"

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/engram_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
__device__ __forceinline__ T HistoryId(const T* past_ids, int64_t b, int64_t slot, int64_t state_length,
                                       T missing_history_value) {
  if (past_ids == nullptr || slot < 0 || slot >= state_length) {
    return missing_history_value;
  }
  return past_ids[b * state_length + slot];
}

template <typename T>
__device__ __forceinline__ T CombinedValue(
    const T* input_ids, const T* past_ids, T missing_history_value,
    int64_t input_base, int64_t history_length, int64_t idx) {
  if (idx < history_length) {
    return HistoryId<T>(past_ids, 0, idx, history_length, missing_history_value);
  }
  return input_ids[input_base + idx - history_length];
}

template <typename T>
__global__ void NGramHashMappingKernel(
    const T* __restrict__ input_ids,
    const T* __restrict__ multipliers,
    const T* __restrict__ vocab_sizes,
    const T* __restrict__ past_ids,
    const T* __restrict__ head_offsets,
    const T* __restrict__ eos_token_id,
    const int32_t* __restrict__ segment_ids,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool reset_on_eos,
    bool stage_tables) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  const int64_t history_length = max_ngram_size - 1;
  const bool has_eos = eos_token_id != nullptr;
  const T eos_value = has_eos ? eos_token_id[0] : pad_id;
  const bool do_reset = reset_on_eos && has_eos;

  extern __shared__ char ngram_shared_bytes[];
  T* shared_multipliers = reinterpret_cast<T*>(ngram_shared_bytes);
  T* shared_vocab_sizes = shared_multipliers + max_ngram_size;
  if (stage_tables) {
    for (int64_t i = threadIdx.x; i < max_ngram_size; i += blockDim.x) {
      shared_multipliers[i] = multipliers[i];
    }
    for (int64_t i = threadIdx.x; i < num_heads; i += blockDim.x) {
      shared_vocab_sizes[i] = vocab_sizes[i];
    }
    __syncthreads();
  }
  const T* multiplier_table = stage_tables ? shared_multipliers : multipliers;
  const T* vocab_table = stage_tables ? shared_vocab_sizes : vocab_sizes;

  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = linear % sequence_length;
    const int64_t b = linear / sequence_length;
    const int64_t input_base = b * sequence_length;
    const int64_t output_base = linear * num_heads;
    const int64_t idx = history_length + t;
    const T* past_row = past_ids != nullptr ? past_ids + b * history_length : nullptr;

    int64_t last_reset = -(history_length + 2);
    for (int64_t j = idx - 1; j >= idx - history_length && j >= 0; --j) {
      bool boundary = do_reset &&
                      CombinedValue(input_ids, past_row, eos_value, input_base, history_length, j) == eos_value;
      if (!boundary && segment_ids != nullptr && j >= history_length) {
        const int64_t tj = j - history_length;
        if (segment_ids[input_base + tj] != segment_ids[input_base + tj + 1]) {
          boundary = true;
        }
      }
      if (boundary) {
        last_reset = j;
        break;
      }
    }

    for (int64_t n = 2; n <= max_ngram_size; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source = idx - k;
        const T token = (last_reset >= source)
                            ? eos_value
                            : CombinedValue(input_ids, past_row, eos_value, input_base, history_length, source);
        const T product = engram_helper::WrappedMultiply<T>(token, multiplier_table[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_table[out_h];
        T result = mod <= 0 ? T{} : engram_helper::PositiveMod(mix, mod);
        if (head_offsets != nullptr) {
          result = static_cast<T>(result + head_offsets[out_h]);
        }
        output[output_base + out_h] = result;
      }
    }
  }
}

template <typename T>
__global__ void NGramPresentIdsKernel(
    const T* input_ids,
    const T* past_ids,
    const T* eos_token_id,
    T* present_ids,
    int64_t sequence_length,
    int64_t state_length,
    T pad_id) {
  const int64_t b = blockIdx.x;
  const int64_t row_base = b * state_length;
  const T missing_history_value = eos_token_id != nullptr ? eos_token_id[0] : pad_id;
  for (int64_t chunk = 0; chunk < state_length; chunk += blockDim.x) {
    const int64_t slot = chunk + threadIdx.x;
    T token = missing_history_value;
    if (slot < state_length) {
      const int64_t source_t = sequence_length - state_length + slot;
      token = source_t >= 0
                  ? input_ids[b * sequence_length + source_t]
                  : HistoryId<T>(past_ids, b, state_length + source_t, state_length, missing_history_value);
    }
    __syncthreads();
    if (slot < state_length) {
      present_ids[row_base + slot] = token;
    }
    __syncthreads();
  }
}

}  // namespace

template <typename T>
Status LaunchNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const T* past_ids,
    const T* head_offsets,
    const T* eos_token_id,
    const int32_t* segment_ids,
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool reset_on_eos) {
  const int64_t state_length = max_ngram_size - 1;

  const int64_t total = batch_size * sequence_length;
  if (total > 0) {
    constexpr size_t kMaxStagedTableBytes = 16 * 1024;
    const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
    const size_t table_bytes = static_cast<size_t>(max_ngram_size + num_heads) * sizeof(T);
    const bool stage_tables = table_bytes <= kMaxStagedTableBytes;
    const size_t shared_bytes = stage_tables ? table_bytes : 0;
    NGramHashMappingKernel<T><<<engram_helper::GridSize(total), engram_helper::kThreads, shared_bytes, stream>>>(
        input_ids, multipliers, vocab_sizes, past_ids, head_offsets, eos_token_id, segment_ids, output, total,
        sequence_length, max_ngram_size, n_head_per_ngram, pad_id, reset_on_eos, stage_tables);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  if (present_ids != nullptr && batch_size * state_length > 0) {
    const int threads = static_cast<int>(std::min<int64_t>(state_length, engram_helper::kThreads));
    NGramPresentIdsKernel<T><<<static_cast<unsigned int>(batch_size), threads, 0, stream>>>(
        input_ids, past_ids, eos_token_id, present_ids, sequence_length, state_length, pad_id);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  return Status::OK();
}

#define INSTANTIATE_NGRAM_HASH_MAPPING(T)                                                               \
  template Status LaunchNGramHashMappingKernel<T>(cudaStream_t, const T*, const T*, const T*, const T*, \
                                                  const T*, const T*, const int32_t*, T*, T*, int64_t,  \
                                                  int64_t, int64_t, int64_t, T, bool);

INSTANTIATE_NGRAM_HASH_MAPPING(int32_t)
INSTANTIATE_NGRAM_HASH_MAPPING(int64_t)

#undef INSTANTIATE_NGRAM_HASH_MAPPING

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
