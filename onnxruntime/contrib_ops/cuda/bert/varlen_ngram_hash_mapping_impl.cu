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

// Scans the complete cumulative_sequence_length array once, on a single thread, to establish
// global monotonicity: cu_seqlens[0] == 0, cu_seqlens[batch_size] == total_tokens, and every
// adjacent pair strictly increasing. Per-block local checks (start < end within [0, total_tokens])
// are not sufficient on their own: a single out-of-order middle entry (e.g. [0, 3, 2, 5]) makes one
// block's local check fail while a different, unrelated block's local check still passes, so that
// block still writes an output range that overlaps a different valid block's range, producing a
// data race between two blocks computing different values for the same output element. Gating
// every output-producing kernel on this single global flag (computed before they run, on the same
// stream) makes such overlap impossible: writes only ever occur when the whole array is valid, in
// which case ranges are provably disjoint.
__global__ void ValidateVarlenCuSeqlensKernel(
    const int32_t* __restrict__ cu_seqlens,
    int batch_size,
    int total_tokens,
    int32_t* is_valid) {
  bool valid = batch_size > 0 && cu_seqlens[0] == 0 && cu_seqlens[batch_size] == total_tokens;
  for (int b = 0; valid && b < batch_size; ++b) {
    const int32_t start = cu_seqlens[b];
    const int32_t end = cu_seqlens[b + 1];
    if (start < 0 || start >= end || end > total_tokens) {
      valid = false;
    }
  }
  *is_valid = valid ? 1 : 0;
}

// Writes deterministic default contents (zero hash ids, pad_id present_ids) whenever the global
// validity flag is false. Every thread reads the flag exactly once and either writes its assigned
// default element or returns immediately, so these writes and VarlenNGramHashMappingKernel's writes
// are mutually exclusive per output element: this kernel only writes when the array is invalid, the
// main kernel only writes when it is valid.
template <typename T>
__global__ void VarlenNGramFillDefaultKernel(
    T* __restrict__ output,
    T* __restrict__ present_ids,
    int64_t output_count,
    int64_t present_count,
    T pad_id,
    const int32_t* __restrict__ is_valid) {
  if (*is_valid) {
    return;
  }
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t idx = start; idx < output_count; idx += stride) {
    output[idx] = T{};
  }
  if (present_ids != nullptr) {
    for (int64_t idx = start; idx < present_count; idx += stride) {
      present_ids[idx] = pad_id;
    }
  }
}

// Gated on the global validity flag computed by ValidateVarlenCuSeqlensKernel: a block only
// accesses input_ids, past_ids, or the outputs once the entire cumulative_sequence_length array is
// known to be well-formed, so its own local range 0 <= start < end <= total_tokens is guaranteed
// disjoint from every other block's range. The local recheck below is defense in depth.
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
    T pad_id,
    const int32_t* __restrict__ is_valid) {
  if (!(*is_valid)) {
    return;
  }
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
    int max_threads_per_block,
    int32_t* is_valid_scratch) {
  if (batch_size == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(batch_size <= std::numeric_limits<int>::max() &&
                        total_tokens <= std::numeric_limits<int>::max(),
                    "VarlenNGramHashMapping: batch_size and total_tokens must fit in a 32-bit int");

  const int64_t state_length = max_ngram_size - 1;
  ORT_RETURN_IF_NOT(n_head_per_ngram > 0 &&
                        state_length <= std::numeric_limits<int64_t>::max() / n_head_per_ngram,
                    "VarlenNGramHashMapping: number of heads overflows int64_t");
  const int64_t num_heads = state_length * n_head_per_ngram;
  ORT_RETURN_IF_NOT(total_tokens == 0 ||
                        total_tokens <= std::numeric_limits<int64_t>::max() / num_heads,
                    "VarlenNGramHashMapping: output dimensions overflow int64_t");
  ORT_RETURN_IF_NOT(batch_size == 0 ||
                        state_length <= std::numeric_limits<int64_t>::max() / batch_size,
                    "VarlenNGramHashMapping: present dimensions overflow int64_t");

  // 1) Compute global validity once, before any output-producing kernel runs.
  ValidateVarlenCuSeqlensKernel<<<1, 1, 0, stream>>>(
      cu_seqlens, static_cast<int>(batch_size), static_cast<int>(total_tokens), is_valid_scratch);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  // 2) Fill deterministic defaults if (and only if) the array turned out to be invalid. This and
  // step 3 below are mutually exclusive on every output element because both are gated on the same
  // flag computed in step 1, so their relative launch order cannot create a race.
  const int64_t output_count = total_tokens * num_heads;
  const int64_t present_count = present_ids == nullptr ? 0 : batch_size * state_length;
  if (output_count > 0 || present_count > 0) {
    const int fill_threads = std::min(256, max_threads_per_block);
    const int64_t max_elements = std::max(output_count, present_count);
    const int64_t fill_blocks = std::min<int64_t>(
        65535, (max_elements + fill_threads - 1) / fill_threads);
    VarlenNGramFillDefaultKernel<T><<<static_cast<unsigned int>(std::max<int64_t>(1, fill_blocks)),
                                      fill_threads, 0, stream>>>(
        output, present_ids, output_count, present_count, pad_id, is_valid_scratch);
    ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  }

  // 3) Compute real values if (and only if) the array is valid.
  const int threads = static_cast<int>(std::max<int64_t>(
      1, std::min<int64_t>(num_heads, std::min(256, max_threads_per_block))));
  VarlenNGramHashMappingKernel<T><<<static_cast<unsigned int>(batch_size), threads, 0, stream>>>(
      input_ids, multipliers, vocab_sizes, cu_seqlens, past_ids, output, present_ids,
      static_cast<int>(batch_size), static_cast<int>(total_tokens), max_ngram_size, n_head_per_ngram,
      pad_id, is_valid_scratch);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(T)                                                      \
  template Status LaunchVarlenNGramHashMappingKernel<T>(                                              \
      cudaStream_t, const T*, const T*, const T*, const int32_t*, const T*, T*, T*, int64_t, int64_t, \
      int64_t, int64_t, T, int, int32_t*);

INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(int32_t)
INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING(int64_t)

#undef INSTANTIATE_VARLEN_NGRAM_HASH_MAPPING

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
