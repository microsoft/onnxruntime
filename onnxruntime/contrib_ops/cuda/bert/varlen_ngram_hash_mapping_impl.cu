// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Packed variable-length n-gram hash mapping. One block is assigned per packed request so that the
// n-gram window for every token in that request is clamped at the request's own boundary and never
// reads across into an adjacent packed request, mirroring VarlenCausalConvWithState's block-per-
// request design.

#include "contrib_ops/cuda/bert/varlen_ngram_hash_mapping_impl.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/engram_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Reads the id at right-aligned history slot `slot` of past_ids for request `b`. Slots outside the
// provided history (or a missing past_ids) are positions before the start of the whole sequence, so
// they use pad_id.
template <typename T>
__device__ __forceinline__ T HistoryId(const T* past_ids, int64_t b, int64_t slot, int64_t state_length,
                                       T pad_id) {
  if (past_ids == nullptr || slot < 0 || slot >= state_length) {
    return pad_id;
  }
  return past_ids[b * state_length + slot];
}

// For memory-safety containment, each block validates cumulative_sequence_length[0] == 0,
// cumulative_sequence_length[batch_size] == total_tokens, and its own local range
// 0 <= start <= end <= total_tokens before accessing input_ids, past_ids, or the outputs.
// Malformed offsets cause the affected block to return without those accesses.
template <typename T>
__global__ void VarlenNGramHashMappingKernel(
    const T* __restrict__ input_ids,
    const T* __restrict__ multipliers,
    const T* __restrict__ vocab_sizes,
    const int32_t* __restrict__ cu_seqlens,
    const T* __restrict__ past_ids,
    T* output,
    T* present_ids,
    int batch_size,
    int total_tokens,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int b = blockIdx.x;
  const int32_t first = cu_seqlens[0];
  const int32_t last = cu_seqlens[batch_size];
  const int32_t start = cu_seqlens[b];
  const int32_t end = cu_seqlens[b + 1];
  if (first != 0 || last != total_tokens || start < 0 || start >= end || end > total_tokens) {
    return;
  }
  const int64_t local_length = end - start;
  const int64_t state_length = max_ngram_size - 1;
  const int64_t num_heads = state_length * n_head_per_ngram;

  // present_ids is the right-aligned trailing window of (past_ids ++ this request's tokens), so it
  // is well defined even when this call is shorter than the window.
  if (present_ids != nullptr) {
    for (int64_t j = threadIdx.x; j < state_length; j += blockDim.x) {
      const int64_t source_t = local_length - state_length + j;
      present_ids[b * state_length + j] =
          source_t >= 0 ? input_ids[start + source_t]
                        : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
    }
  }

  for (int64_t out_h = threadIdx.x; out_h < num_heads; out_h += blockDim.x) {
    const int64_t n = out_h / n_head_per_ngram + 2;
    const T mod = vocab_sizes[out_h];
    for (int64_t t = 0; t < local_length; ++t) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source_t = t - k;
        const T token = source_t >= 0
                            ? input_ids[start + source_t]
                            : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
        const T product = engram_helper::WrappedMultiply<T>(token, multipliers[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }
      output[(start + t) * num_heads + out_h] = mod <= 0 ? T{} : engram_helper::PositiveMod(mix, mod);
    }
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenNGramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const int32_t* cu_seqlens,
    const T* past_ids,
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t total_tokens,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    int max_threads_per_block) {
  if (batch_size == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(batch_size <= std::numeric_limits<int>::max() &&
                        total_tokens <= std::numeric_limits<int>::max(),
                    "VarlenNGramHashMapping: batch_size and total_tokens must fit in a 32-bit int");

  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  const int threads = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(num_heads, std::min(256, max_threads_per_block))));
  VarlenNGramHashMappingKernel<T><<<static_cast<unsigned int>(batch_size), threads, 0, stream>>>(
      input_ids, multipliers, vocab_sizes, cu_seqlens, past_ids, output, present_ids,
      static_cast<int>(batch_size), static_cast<int>(total_tokens), max_ngram_size, n_head_per_ngram,
      pad_id);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(T)                                                      \
  template Status LaunchVarlenNGramHashMappingKernel<T>(                                              \
      cudaStream_t, const T*, const T*, const T*, const int32_t*, const T*, T*, T*, int64_t, int64_t, \
      int64_t, int64_t, T, int);

INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(int32_t)
INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(int64_t)

#undef INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
