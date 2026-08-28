// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/ngram_hash_mapping_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/engram_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Reads the id at right-aligned history slot `slot`. Slots outside the provided history (or a missing
// past_ids) are positions before the start of the whole sequence, so they use pad_id.
template <typename T>
__device__ __forceinline__ T HistoryId(const T* past_ids, int64_t b, int64_t slot, int64_t state_length,
                                       T pad_id) {
  if (past_ids == nullptr || slot < 0 || slot >= state_length) {
    return pad_id;
  }
  return past_ids[b * state_length + slot];
}

template <typename T>
__global__ void NGramHashMappingKernel(
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    const T* past_ids,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  const int64_t state_length = max_ngram_size - 1;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = linear % sequence_length;
    const int64_t b = linear / sequence_length;
    const int64_t input_base = b * sequence_length;
    const int64_t output_base = linear * num_heads;

    for (int64_t n = 2; n <= max_ngram_size; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source_t = t - k;
        const T token = source_t >= 0
                            ? input_ids[input_base + source_t]
                            : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
        const T product = engram_helper::WrappedMultiply<T>(token, multipliers[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_sizes[out_h];
        output[output_base + out_h] = mod <= 0 ? T{} : engram_helper::PositiveMod(mix, mod);
      }
    }
  }
}

// present_ids is the right-aligned trailing window of (past_ids ++ input_ids), so it is well defined
// even when this call is shorter than the window.
template <typename T>
__global__ void NGramPresentIdsKernel(
    const T* input_ids,
    const T* past_ids,
    T* present_ids,
    int64_t total,
    int64_t sequence_length,
    int64_t state_length,
    T pad_id) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t slot = linear % state_length;
    const int64_t b = linear / state_length;
    const int64_t source_t = sequence_length - state_length + slot;
    present_ids[linear] = source_t >= 0
                              ? input_ids[b * sequence_length + source_t]
                              : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
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
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t state_length = max_ngram_size - 1;
  if (present_ids != nullptr && batch_size * state_length > 0) {
    const int64_t present_total = batch_size * state_length;
    NGramPresentIdsKernel<T><<<engram_helper::GridSize(present_total), engram_helper::kThreads, 0, stream>>>(
        input_ids, past_ids, present_ids, present_total, sequence_length, state_length, pad_id);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  const int64_t total = batch_size * sequence_length;
  if (total == 0) {
    return Status::OK();
  }
  NGramHashMappingKernel<T><<<engram_helper::GridSize(total), engram_helper::kThreads, 0, stream>>>(
      input_ids, multipliers, vocab_sizes, past_ids, output, total, sequence_length, max_ngram_size,
      n_head_per_ngram, pad_id);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_NGRAM_HASH_MAPPING(T)                                                      \
  template Status LaunchNGramHashMappingKernel<T>(cudaStream_t, const T*, const T*, const T*,  \
                                                  const T*, T*, T*, int64_t, int64_t, int64_t, \
                                                  int64_t, T);

INSTANTIATE_NGRAM_HASH_MAPPING(int32_t)
INSTANTIATE_NGRAM_HASH_MAPPING(int64_t)

#undef INSTANTIATE_NGRAM_HASH_MAPPING

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
