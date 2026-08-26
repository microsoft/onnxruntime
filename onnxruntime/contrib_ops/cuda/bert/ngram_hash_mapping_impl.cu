// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/ngram_hash_mapping_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/kernel_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Reads the raw (never EOS-substituted) token id at combined-timeline position `idx`, where
// idx in [0, history_length) comes from past_tokens (or eos_value when past_tokens is absent)
// and idx in [history_length, history_length + sequence_length) comes from input_ids.
template <typename T>
__device__ __forceinline__ T CombinedValue(
    const T* input_ids, const T* past_tokens, T eos_value,
    int64_t input_base, int64_t history_length, int64_t idx) {
  if (idx < history_length) {
    return past_tokens != nullptr ? past_tokens[idx] : eos_value;
  }
  return input_ids[input_base + (idx - history_length)];
}

template <typename T>
__global__ void NGramHashMappingKernel(
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const T* past_tokens,
    const T* head_offsets,
    const T* eos_token_id,
    const int32_t* segment_ids,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool reset_on_eos) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  const int64_t history_length = max_ngram_size - 1;
  const bool has_eos = eos_token_id != nullptr;
  const T eos_value = has_eos ? eos_token_id[0] : pad_id;
  const bool do_reset = reset_on_eos && has_eos;

  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = linear % sequence_length;
    const int64_t b = linear / sequence_length;
    const int64_t input_base = b * sequence_length;
    const int64_t output_base = linear * num_heads;
    const int64_t idx = history_length + t;
    const T* past_row = past_tokens != nullptr ? past_tokens + b * history_length : nullptr;

    // Every n-gram shift for this position reaches back at most history_length positions, so a
    // bounded backward scan over that window is enough to find the most recent reset boundary.
    int64_t last_reset = -(history_length + 2);
    for (int64_t j = idx - 1; j >= idx - history_length && j >= 0; --j) {
      bool boundary = do_reset &&
                      CombinedValue(input_ids, past_row, eos_value, input_base, history_length, j) == eos_value;
      if (!boundary && segment_ids != nullptr && j > history_length) {
        const int64_t tj = j - history_length;
        if (segment_ids[input_base + tj] != segment_ids[input_base + tj - 1]) {
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
        const T product = kernel_helper::WrappedMultiply<T>(token, multipliers[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_sizes[out_h];
        T result = mod <= 0 ? T{} : kernel_helper::PositiveMod(mix, mod);
        if (head_offsets != nullptr) {
          result = static_cast<T>(result + head_offsets[out_h]);
        }
        output[output_base + out_h] = result;
      }
    }
  }
}

template <typename T>
__global__ void NGramPresentTokensKernel(
    const T* input_ids,
    const T* past_tokens,
    const T* eos_token_id,
    T* present_tokens,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t history_length,
    T pad_id) {
  const T eos_value = eos_token_id != nullptr ? eos_token_id[0] : pad_id;
  const int64_t total = batch_size * history_length;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t i = linear % history_length;
    const int64_t b = linear / history_length;
    const int64_t idx = sequence_length + i;  // position in the [past | input_ids] timeline
    const T* past_row = past_tokens != nullptr ? past_tokens + b * history_length : nullptr;
    present_tokens[linear] = CombinedValue(input_ids + b * sequence_length, past_row,
                                           eos_value, 0, history_length, idx);
  }
}

}  // namespace

template <typename T>
Status LaunchNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const T* past_tokens,
    const T* head_offsets,
    const T* eos_token_id,
    const int32_t* segment_ids,
    T* output,
    T* present_tokens,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool reset_on_eos) {
  const int64_t total = batch_size * sequence_length;
  if (total > 0) {
    NGramHashMappingKernel<T><<<kernel_helper::GridSize(total), kernel_helper::kThreads, 0, stream>>>(
        input_ids, multipliers, vocab_sizes, past_tokens, head_offsets, eos_token_id, segment_ids,
        output, total, sequence_length, max_ngram_size, n_head_per_ngram, pad_id, reset_on_eos);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  if (present_tokens != nullptr) {
    const int64_t history_length = max_ngram_size - 1;
    const int64_t present_total = batch_size * history_length;
    if (present_total > 0) {
      NGramPresentTokensKernel<T><<<kernel_helper::GridSize(present_total), kernel_helper::kThreads, 0, stream>>>(
          input_ids, past_tokens, eos_token_id, present_tokens, batch_size, sequence_length, history_length, pad_id);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
  }

  return Status::OK();
}

template Status LaunchNGramHashMappingKernel<int32_t>(cudaStream_t, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t, int32_t, bool);
template Status LaunchNGramHashMappingKernel<int64_t>(cudaStream_t, const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int32_t*, int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t, bool);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
