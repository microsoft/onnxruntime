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
    const T* __restrict__ input_ids,
    const T* __restrict__ multipliers,
    const T* __restrict__ vocab_sizes,
    const T* __restrict__ past_ids,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id,
    bool stage_tables) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  const int64_t state_length = max_ngram_size - 1;

  // multipliers and vocab_sizes are uniform across the whole grid and tiny, but they are read in the
  // two innermost loops. Stage them into shared memory once per block so those reads never leave the
  // SM. The launch clears stage_tables if the tables would not fit, in which case the __restrict__
  // pointers let the compiler serve them from the read-only cache instead.
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

    for (int64_t n = 2; n <= max_ngram_size; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source_t = t - k;
        const T token = source_t >= 0
                            ? input_ids[input_base + source_t]
                            : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
        const T product = engram_helper::WrappedMultiply<T>(token, multiplier_table[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_table[out_h];
        output[output_base + out_h] = mod <= 0 ? T{} : engram_helper::PositiveMod(mix, mod);
      }
    }
  }
}

// present_ids is the right-aligned trailing window of (past_ids ++ input_ids), so it is well defined
// even when this call is shorter than the window.
//
// past_ids and present_ids may be the same allocation, which is what a decode loop that feeds
// present_ids straight back as past_ids naturally produces. Slot `slot` writes index `slot` and may
// read index `slot + sequence_length`, so the write range overlaps the read range and the two must be
// separated. One block owns one batch row and processes it in ascending blockDim.x-sized chunks: a
// barrier separates the whole chunk's reads from the whole chunk's writes, and a chunk only ever
// writes indices strictly below the read indices of every later chunk.
template <typename T>
__global__ void NGramPresentIdsKernel(
    const T* input_ids,
    const T* past_ids,
    T* present_ids,
    int64_t sequence_length,
    int64_t state_length,
    T pad_id) {
  const int64_t b = blockIdx.x;
  const int64_t row_base = b * state_length;
  for (int64_t chunk = 0; chunk < state_length; chunk += blockDim.x) {
    const int64_t slot = chunk + threadIdx.x;
    T token = pad_id;
    if (slot < state_length) {
      const int64_t source_t = sequence_length - state_length + slot;
      token = source_t >= 0
                  ? input_ids[b * sequence_length + source_t]
                  : HistoryId<T>(past_ids, b, state_length + source_t, state_length, pad_id);
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
    T* output,
    T* present_ids,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t state_length = max_ngram_size - 1;

  // The hash kernel reads past_ids and the present kernel writes present_ids, so when the caller
  // aliases the two the hash kernel must run first. Both launches are on the same stream, which
  // orders them.
  const int64_t total = batch_size * sequence_length;
  if (total > 0) {
    // Shared-memory staging for the two lookup tables. 16 KB keeps occupancy unaffected on every
    // architecture ORT targets; realistic Engram configurations need only a few hundred bytes.
    constexpr size_t kMaxStagedTableBytes = 16 * 1024;
    const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
    const size_t table_bytes = static_cast<size_t>(max_ngram_size + num_heads) * sizeof(T);
    const bool stage_tables = table_bytes <= kMaxStagedTableBytes;
    const size_t shared_bytes = stage_tables ? table_bytes : 0;
    NGramHashMappingKernel<T><<<engram_helper::GridSize(total), engram_helper::kThreads, shared_bytes, stream>>>(
        input_ids, multipliers, vocab_sizes, past_ids, output, total, sequence_length, max_ngram_size,
        n_head_per_ngram, pad_id, stage_tables);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  if (present_ids != nullptr && batch_size * state_length > 0) {
    // One block per batch row; the kernel walks the row in chunks so state_length may exceed the
    // block size.
    const int threads = static_cast<int>(std::min<int64_t>(state_length, engram_helper::kThreads));
    NGramPresentIdsKernel<T><<<static_cast<unsigned int>(batch_size), threads, 0, stream>>>(
        input_ids, past_ids, present_ids, sequence_length, state_length, pad_id);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }
  return Status::OK();
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
