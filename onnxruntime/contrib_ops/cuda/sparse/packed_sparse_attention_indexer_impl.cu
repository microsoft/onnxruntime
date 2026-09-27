// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Correctness-first implementation of com.microsoft.PackedSparseAttentionIndexer. It shares its
// block reductions, deterministic argmax/selection helpers, RoPE math and causal-threshold formula
// with com.microsoft.SparseAttentionIndexer via sparse_attention_indexer_device_math.cuh, and
// reuses the CSA window-plan arithmetic directly on the device from
// sparse_attention_indexer_common.h (its helpers are SAI_HOST_DEVICE). What is packed-specific:
// token-major (not batch-major) layout, cumulative_sequence_lengths / past_sequence_lengths driven
// per-request bookkeeping, fully in-place fixed-capacity state update (no growing/concatenating
// state), and plain causal visibility derived from packed metadata (no dense mask).
//
// Device-side safety: every per-request quantity (past_sequence_lengths, past_state_lengths,
// cumulative offsets) is read directly from device memory inside the kernels below -- there is no
// host readback or stream synchronization. Values are always clamped into the fixed-capacity range
// before use, so malformed metadata can make the result semantically wrong but can never cause an
// out-of-bounds access or an overlapping write. State-capacity overflow is *rejected*, not
// silently truncated: if a request's new blocks/windows would not all fit in state_capacity this
// call, the update kernel applies none of them (present_state_lengths / present_key_state /
// present_kv_buffer / present_gate_buffer for that request are left exactly as their past_*
// counterparts) and records the rejection in a small overflow_flags workspace; the select kernels
// then force that request's selected_indices/selected_counts to the deterministic safe empty
// result (-1 / 0) for this call instead of selecting against a partially updated state. See
// QsaUpdateStateKernel / CsaUpdateStateKernel / QsaSelectKernel / CsaSelectKernel.
//
// See docs/contrib_ops/cuda/packed_sparse_attention_indexer.md for the full operator contract.

#include "contrib_ops/cuda/sparse/packed_sparse_attention_indexer_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>

#include "contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h"
#include "contrib_ops/cuda/sparse/sparse_attention_indexer_device_math.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace psai = onnxruntime::contrib::packed_sparse_attention_indexer;
namespace topk = onnxruntime::cuda::topk;

namespace {

// The block reductions below halve the active thread count, so this must stay a power of two.
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kBoundedTopKMax = 512;
constexpr int kBoundedTopKItemsPerThread = kBoundedTopKMax / kThreads;
constexpr int kDistributedTopKMaxRows = 64;
constexpr int kDistributedTopKMinBlocks = 2048;
constexpr int kHierarchicalTileBlocks = 8;
constexpr int kHierarchicalScoreThreads = 1024;
constexpr int kHierarchicalMergeThreads = 256;
constexpr int kQwenHeadSize = 128;
constexpr int kQwenCompressRatio = 4;
constexpr int kQwenNumHeads = 4;

bool UseHierarchicalQsaTopK(const PackedSparseAttentionIndexerParams& params) {
  return params.total_tokens > 0 && params.total_tokens <= kDistributedTopKMaxRows &&
         params.state_capacity >= kDistributedTopKMinBlocks &&
         params.head_size == kQwenHeadSize && params.compress_ratio == kQwenCompressRatio &&
         params.num_heads == kQwenNumHeads && params.block_topk > kSaiFastTopKMax &&
         params.block_topk <= kBoundedTopKMax;
}

int GetHierarchicalTileCount(const PackedSparseAttentionIndexerParams& params) {
  return (params.state_capacity + kHierarchicalTileBlocks - 1) / kHierarchicalTileBlocks;
}

template <typename T>
struct alignas(sizeof(T) * 4) QsaVector4 {
  T values[4];
};

template <typename T>
__device__ __forceinline__ float4 LoadQsaVector4(const T* source) {
  const QsaVector4<T> packed = *reinterpret_cast<const QsaVector4<T>*>(source);
  return make_float4(to_float<T>(packed.values[0]), to_float<T>(packed.values[1]),
                     to_float<T>(packed.values[2]), to_float<T>(packed.values[3]));
}

// Largest b such that cumulative_sequence_lengths[b] <= token, assuming the array is nondecreasing.
// If the data itself is malformed this may attribute a token to the wrong request, but the result
// is always an index in [0, batch_size), so it can never cause an out-of-bounds access.
__device__ __forceinline__ int PackedBatchOfToken(const int32_t* cumulative_sequence_lengths, int batch_size,
                                                  int token) {
  int lo = 0;
  int hi = batch_size - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo + 1) / 2;
    if (cumulative_sequence_lengths[mid] <= token) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

template <typename T>
__global__ void ElementwiseCopyKernel(const T* src, T* dst, int64_t count) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    dst[i] = src[i];
  }
}

// ---------------------------------------------------------------------------------------------
// policy_mode = "qsa"
// ---------------------------------------------------------------------------------------------

// One block per request: forms every newly-closed compress_ratio block (mean-pool -> RMSNorm ->
// leading RoPE -> append into fixed-capacity key_state) and publishes the raw trailing buffer.
// Reads only past_kv_buffer / key (never present_kv_buffer / present_key_state), so it is correct
// whether or not the present/past tensors are the same aliased allocation.
template <typename T>
__global__ void QsaUpdateStateKernel(const T* key, const T* key_norm_weight, const T* cos_cache,
                                     const T* sin_cache, const int32_t* cumulative_sequence_lengths,
                                     const int32_t* past_sequence_lengths, const T* past_kv_buffer,
                                     const int32_t* past_state_lengths,
                                     T* present_key_state, T* present_kv_buffer,
                                     int32_t* present_state_lengths, int32_t* overflow_flags,
                                     PackedSparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* rotated = shared + params.head_size;
  float* reduction = shared + 2 * params.head_size;

  for (int b = static_cast<int>(blockIdx.x); b < params.batch_size; b += static_cast<int>(gridDim.x)) {
    const int req_start = cumulative_sequence_lengths[b];
    const int req_end = cumulative_sequence_lengths[b + 1];
    const int raw_key_len = past_state_lengths[b * 2 + psai::kKeyStateLength];
    const int raw_buf_len = past_state_lengths[b * 2 + psai::kBufferLength];
    const int past_sequence_length = past_sequence_lengths[b];
    const bool invalid_metadata =
        cumulative_sequence_lengths[0] != 0 ||
        cumulative_sequence_lengths[params.batch_size] != params.total_tokens ||
        req_start < 0 || req_end < req_start || req_end > params.total_tokens ||
        past_sequence_length < 0 || raw_key_len < 0 || raw_key_len > params.state_capacity ||
        raw_buf_len < 0 || raw_buf_len >= params.compress_ratio ||
        raw_key_len != past_sequence_length / params.compress_ratio ||
        raw_buf_len != past_sequence_length % params.compress_ratio;
    const int req_len = invalid_metadata ? 0 : req_end - req_start;
    const int old_key_len = min(max(raw_key_len, 0), params.state_capacity);
    const int old_buf_len = min(max(raw_buf_len, 0), params.compress_ratio - 1);

    const int pending = old_buf_len + req_len;
    const int full_new_block_count = pending / params.compress_ratio;
    const int capacity_left = params.state_capacity - old_key_len;  // >= 0 by construction of old_key_len
    // Reject (do not partially apply) a step that would need more than the fixed state_capacity:
    // no new blocks are formed and the buffer is left exactly as it was, so a rejected step is a
    // deterministic no-op on state rather than a silent partial truncation.
    const bool rejected = invalid_metadata || full_new_block_count > capacity_left;
    const int new_block_count = rejected ? 0 : full_new_block_count;
    const int new_buf_len = rejected ? old_buf_len : (pending % params.compress_ratio);

    // Barrier: every thread has now read past_state_lengths (identically) before any thread below
    // writes present_state_lengths, which keeps this correct even if the two tensors alias.
    __syncthreads();

    if (threadIdx.x == 0) {
      present_state_lengths[b * 2 + psai::kKeyStateLength] = old_key_len + new_block_count;
      present_state_lengths[b * 2 + psai::kBufferLength] = new_buf_len;
      overflow_flags[b] = rejected ? 1 : 0;
    }

    for (int k = 0; k < new_block_count; ++k) {
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        float sum = 0.0f;
        for (int t = 0; t < params.compress_ratio; ++t) {
          const int virtual_pos = k * params.compress_ratio + t;
          sum += virtual_pos < old_buf_len
                     ? to_float<T>(past_kv_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) *
                                                      params.head_size +
                                                  d])
                     : to_float<T>(key[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) *
                                           params.key_row_stride +
                                       d]);
        }
        pooled[d] = sum / static_cast<float>(params.compress_ratio);
      }
      __syncthreads();

      float sum_squares = 0.0f;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        sum_squares += pooled[d] * pooled[d];
      }
      sum_squares = SaiBlockSum(sum_squares, reduction);
      const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        pooled[d] = pooled[d] * inverse_rms * to_float<T>(key_norm_weight[d]);
      }
      __syncthreads();

      const int entry = old_key_len + k;
      const int rope_position =
          SaiClampPosition(static_cast<int64_t>(entry) * params.compress_ratio, params.max_rotary_length);
      const int64_t cache_offset =
          (static_cast<int64_t>(params.cos_cache_batched ? b : 0) * params.max_rotary_length + rope_position) *
          params.rotary_width;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        rotated[d] = SaiLeadingRope<T>(pooled, params.rotary_width, cos_cache + cache_offset,
                                       sin_cache + cache_offset, d);
      }
      __syncthreads();

      const int64_t out_base = (static_cast<int64_t>(b) * params.state_capacity + entry) * params.head_size;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        present_key_state[out_base + d] = from_float<T>(rotated[d]);
      }
      __syncthreads();
    }

    // Publish the raw trailing buffer. Skipped entirely on overflow: present_kv_buffer already
    // holds past_kv_buffer's contents unchanged (from the baseline copy in the Launch function
    // below), which is exactly the prior valid buffer this rejected step must preserve.
    if (!rejected) {
      for (int t = static_cast<int>(threadIdx.x); t < new_buf_len; t += static_cast<int>(blockDim.x)) {
        const int virtual_pos = new_block_count * params.compress_ratio + t;
        const int64_t out_base = (static_cast<int64_t>(b) * params.buffer_capacity + t) * params.head_size;
        for (int d = 0; d < params.head_size; ++d) {
          const float value =
              virtual_pos < old_buf_len
                  ? to_float<T>(
                        past_kv_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) *
                                           params.head_size +
                                       d])
                  : to_float<T>(
                        key[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) *
                                params.key_row_stride +
                            d]);
          present_kv_buffer[out_base + d] = from_float<T>(value);
        }
      }
    }
    __syncthreads();
  }
}

// One block per (token, head): rotates the query once so downstream scoring kernels only ever dot
// two already-rotated/prepared vectors. kUseLeadingRope selects the qsa convention (position
// defaults to past_sequence_lengths[batch] + request-local offset when position_ids is absent);
// otherwise the csa trailing convention with positions always taken from position_ids.
template <typename T, bool kUseLeadingRope>
__global__ void PackedRotateQueryKernel(const T* query, const T* query_norm_weight,
                                        const T* cos_cache, const T* sin_cache,
                                        const int32_t* cumulative_sequence_lengths,
                                        const int32_t* past_sequence_lengths, const int64_t* position_ids,
                                        float* query_rotated, PackedSparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* reduction = shared + params.head_size;
  const int64_t rows = static_cast<int64_t>(params.total_tokens) * params.num_heads;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int token = static_cast<int>(row / params.num_heads);
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    const int64_t input_base = static_cast<int64_t>(token) * params.query_row_stride +
                               static_cast<int64_t>(row % params.num_heads) * params.head_size;

    for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
      shared[d] = to_float<T>(query[input_base + d]);
    }
    __syncthreads();

    float sum_squares = 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
      sum_squares += shared[d] * shared[d];
    }
    sum_squares = SaiBlockSum(sum_squares, reduction);
    const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
    for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
      shared[d] = shared[d] * inverse_rms * to_float<T>(query_norm_weight[d]);
    }
    __syncthreads();

    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int position = SaiClampPosition(abs_position, params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(params.cos_cache_batched ? batch : 0) * params.max_rotary_length + position) *
        params.rotary_width;
    const T* cos_row = cos_cache + cache_offset;
    const T* sin_row = sin_cache + cache_offset;

    for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
      query_rotated[row * params.head_size + d] =
          kUseLeadingRope ? SaiLeadingRope<T>(shared, params.rotary_width, cos_row, sin_row, d)
                          : SaiTrailingRope<T>(shared, params.head_size, params.rotary_width, cos_row, sin_row, d);
    }
    __syncthreads();
  }
}

// One block per (token, key_state slot). Scores are directly dotted against the already-prepared
// present_key_state entry (unlike the dense op, no per-query pooling/normalize/rotate is repeated
// here because the packed contract stores fully-prepared blocks in key_state).
template <typename T>
__global__ void QsaBlockScoreKernel(const T* present_key_state, const float* query_rotated,
                                    const int32_t* cumulative_sequence_lengths,
                                    const int32_t* past_sequence_lengths, const int64_t* position_ids,
                                    const int32_t* present_state_lengths, float* block_scores,
                                    PackedSparseAttentionIndexerParams params) {
  extern __shared__ float reduction[];
  const int64_t total = static_cast<int64_t>(params.total_tokens) * params.state_capacity;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int token = static_cast<int>(work / params.state_capacity);
    const int block_index = static_cast<int>(work % params.state_capacity);
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];

    if (block_index >= key_len_after) {
      if (threadIdx.x == 0) {
        block_scores[work] = SaiNegativeInfinity();
      }
      continue;
    }

    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int64_t causal_count = SaiCausalThreshold(abs_position, params.compress_ratio);
    const int64_t visible_block_count = causal_count < key_len_after ? causal_count : key_len_after;
    if (static_cast<int64_t>(block_index) >= visible_block_count) {
      if (threadIdx.x == 0) {
        block_scores[work] = SaiNegativeInfinity();
      }
      continue;
    }

    const int64_t key_base = (static_cast<int64_t>(batch) * params.state_capacity + block_index) * params.head_size;
    float score = 0.0f;
    for (int head = 0; head < params.num_heads; ++head) {
      const float* query_head =
          query_rotated + (static_cast<int64_t>(token) * params.num_heads + head) * params.head_size;
      float partial = 0.0f;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        partial += query_head[d] * to_float<T>(present_key_state[key_base + d]);
      }
      score += fmaxf(SaiBlockSum(partial, reduction), 0.0f);
    }
    if (threadIdx.x == 0) {
      block_scores[work] = score * params.scale;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void QsaScoreTileTopKKernel(const T* present_key_state, const float* query_rotated,
                                       const int32_t* cumulative_sequence_lengths,
                                       const int32_t* past_sequence_lengths, const int64_t* position_ids,
                                       const int32_t* present_state_lengths, const int32_t* overflow_flags,
                                       uint64_t* tile_keys, int tile_count,
                                       PackedSparseAttentionIndexerParams params) {
  __shared__ uint64_t keys[kHierarchicalTileBlocks];
  __shared__ float head_scores[kHierarchicalTileBlocks][kQwenNumHeads];

  const int64_t total_tiles = static_cast<int64_t>(params.total_tokens) * tile_count;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int block_slot = warp / kQwenNumHeads;
  const int head = warp % kQwenNumHeads;
  for (int64_t work = blockIdx.x; work < total_tiles; work += gridDim.x) {
    const int tile = static_cast<int>(work % tile_count);
    const int token = static_cast<int>(work / tile_count);
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];
    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int64_t causal_count = SaiCausalThreshold(abs_position, kQwenCompressRatio);
    const int64_t visible_64 = causal_count < key_len_after ? causal_count : static_cast<int64_t>(key_len_after);
    const int block_count = overflow_flags[batch] == 0 ? static_cast<int>(visible_64 < 0 ? 0 : visible_64) : 0;

    const int block_index = tile * kHierarchicalTileBlocks + block_slot;
    float score = 0.0f;
    if (block_index < block_count) {
      const int d = lane * 4;
      const int64_t query_base =
          (static_cast<int64_t>(token) * kQwenNumHeads + head) * kQwenHeadSize;
      const int64_t key_base =
          (static_cast<int64_t>(batch) * params.state_capacity + block_index) * kQwenHeadSize;
      const float4 query_value = LoadQsaVector4(query_rotated + query_base + d);
      const float4 key_value = LoadQsaVector4(present_key_state + key_base + d);
      score = query_value.x * key_value.x + query_value.y * key_value.y +
              query_value.z * key_value.z + query_value.w * key_value.w;
      score = topk::WarpReduceSum(score);
    }
    if (lane == 0) {
      head_scores[block_slot][head] = fmaxf(score, 0.0f);
    }
    __syncthreads();
    if (threadIdx.x < kHierarchicalTileBlocks) {
      const int tile_block = tile * kHierarchicalTileBlocks + static_cast<int>(threadIdx.x);
      uint64_t key = topk::kPaddingSortKey;
      if (tile_block < block_count) {
        float total_score = 0.0f;
#pragma unroll
        for (int query_head = 0; query_head < kQwenNumHeads; ++query_head) {
          total_score += head_scores[threadIdx.x][query_head];
        }
        const float scaled_score = total_score * params.scale;
        key = topk::PackStableSortKey(scaled_score == 0.0f ? 0.0f : scaled_score, tile_block);
      }
      keys[threadIdx.x] = key;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int index = 1; index < kHierarchicalTileBlocks; ++index) {
        const uint64_t key = keys[index];
        int insertion = index;
        while (insertion > 0 && key > keys[insertion - 1]) {
          keys[insertion] = keys[insertion - 1];
          --insertion;
        }
        keys[insertion] = key;
      }
    }
    __syncthreads();
    const int64_t output_base = work * kHierarchicalTileBlocks;
    if (threadIdx.x < kHierarchicalTileBlocks) {
      tile_keys[output_base + threadIdx.x] = keys[threadIdx.x];
    }
    __syncthreads();
  }
}

__device__ __forceinline__ uint64_t QsaMergeRank(const uint64_t* left, int left_count,
                                                 const uint64_t* right, int right_count, int rank) {
  int low = max(0, rank - right_count);
  int high = min(rank, left_count);
  while (low <= high) {
    const int left_rank = (low + high) / 2;
    const int right_rank = rank - left_rank;
    if (left_rank > 0 && right_rank < right_count && left[left_rank - 1] < right[right_rank]) {
      high = left_rank - 1;
    } else if (right_rank > 0 && left_rank < left_count && right[right_rank - 1] < left[left_rank]) {
      low = left_rank + 1;
    } else {
      const uint64_t left_key = left_rank < left_count ? left[left_rank] : topk::kPaddingSortKey;
      const uint64_t right_key = right_rank < right_count ? right[right_rank] : topk::kPaddingSortKey;
      return max(left_key, right_key);
    }
  }
  return topk::kPaddingSortKey;
}

__global__ void QsaMergeTileTopKKernel(const uint64_t* input_keys, uint64_t* output_keys,
                                       int key_stride, int list_width, int rows) {
  const int pairs_per_row = (key_stride + 2 * list_width - 1) / (2 * list_width);
  const int64_t total_pairs = static_cast<int64_t>(rows) * pairs_per_row;
  for (int64_t work = blockIdx.x; work < total_pairs; work += gridDim.x) {
    const int pair = static_cast<int>(work % pairs_per_row);
    const int64_t row = work / pairs_per_row;
    const int left_start = pair * 2 * list_width;
    const int right_start = left_start + list_width;
    const int left_count = min(min(list_width, kBoundedTopKMax), max(0, key_stride - left_start));
    const int right_count = min(min(list_width, kBoundedTopKMax), max(0, key_stride - right_start));
    const int output_count = min(left_count + right_count, kBoundedTopKMax);
    const int64_t row_base = row * key_stride;
    for (int rank = threadIdx.x; rank < output_count; rank += blockDim.x) {
      output_keys[row_base + left_start + rank] =
          QsaMergeRank(input_keys + row_base + left_start, left_count,
                       input_keys + row_base + right_start, right_count, rank);
    }
  }
}

__global__ void QsaEmitHierarchicalTopKKernel(const uint64_t* tile_keys,
                                              const int32_t* cumulative_sequence_lengths,
                                              const int32_t* past_sequence_lengths,
                                              const int64_t* position_ids,
                                              const int32_t* present_state_lengths,
                                              const int32_t* overflow_flags, int key_stride,
                                              int32_t* selected_indices, int32_t* selected_counts,
                                              PackedSparseAttentionIndexerParams params) {
  for (int token = static_cast<int>(blockIdx.x); token < params.total_tokens; token += static_cast<int>(gridDim.x)) {
    int32_t* output = selected_indices + static_cast<int64_t>(token) * params.capacity;
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];
    const int64_t causal_count = SaiCausalThreshold(abs_position, params.compress_ratio);
    const int64_t visible_64 = causal_count < key_len_after ? causal_count : static_cast<int64_t>(key_len_after);
    const int block_count = overflow_flags[batch] == 0 ? static_cast<int>(visible_64 < 0 ? 0 : visible_64) : 0;
    const int selected = min(params.block_topk, block_count);
    for (int rank = threadIdx.x; rank < selected; rank += blockDim.x) {
      const int selected_block = topk::UnpackStableSortIndex(tile_keys[static_cast<int64_t>(token) * key_stride + rank]);
      for (int t = 0; t < params.compress_ratio; ++t) {
        output[rank * params.compress_ratio + t] = selected_block * params.compress_ratio + t;
      }
    }
    const int64_t block_start = static_cast<int64_t>(block_count) * params.compress_ratio;
    const int64_t natural_tail = abs_position >= block_start ? abs_position - block_start + 1 : 0;
    const int remaining_capacity = params.capacity - selected * params.compress_ratio;
    const int64_t tail_count_64 = natural_tail < remaining_capacity ? natural_tail : remaining_capacity;
    const int tail_count = overflow_flags[batch] == 0 ? static_cast<int>(tail_count_64 < 0 ? 0 : tail_count_64) : 0;
    for (int t = threadIdx.x; t < tail_count; t += blockDim.x) {
      output[selected * params.compress_ratio + t] = static_cast<int32_t>(block_start + t);
    }
    for (int position = selected * params.compress_ratio + tail_count + threadIdx.x;
         position < params.capacity; position += blockDim.x) {
      output[position] = -1;
    }
    if (threadIdx.x == 0) {
      selected_counts[token] = selected * params.compress_ratio + tail_count;
    }
  }
}

// One block per query token. Emits the token indices of the highest scoring blocks followed by the
// causally visible tokens of the trailing incomplete block, and the exact active count. A request
// whose update step was rejected for exceeding state_capacity this call (overflow_flags[batch] set)
// always gets the safe empty result: indices stay -1 (already reset below) and count is 0.
__global__ void QsaSelectKernel(const float* block_scores, const int32_t* topk_indices,
                                const int32_t* cumulative_sequence_lengths,
                                const int32_t* past_sequence_lengths, const int64_t* position_ids,
                                const int32_t* present_state_lengths, const int32_t* overflow_flags,
                                int32_t* selected_indices, int32_t* selected_counts,
                                PackedSparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* shared_value = shared;
  const bool use_fast_topk = params.block_topk <= kSaiFastTopKMax;
  const int shared_entries = use_fast_topk ? static_cast<int>(blockDim.x) * params.block_topk
                                           : static_cast<int>(blockDim.x);
  int* shared_index = reinterpret_cast<int*>(shared + shared_entries);

  for (int token = static_cast<int>(blockIdx.x); token < params.total_tokens; token += static_cast<int>(gridDim.x)) {
    int32_t* out_row = selected_indices + static_cast<int64_t>(token) * params.capacity;
    for (int p = static_cast<int>(threadIdx.x); p < params.capacity; p += static_cast<int>(blockDim.x)) {
      out_row[p] = -1;
    }
    __syncthreads();

    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    if (overflow_flags[batch] != 0) {
      if (threadIdx.x == 0) {
        selected_counts[token] = 0;
      }
      __syncthreads();
      continue;
    }

    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];
    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int64_t causal_count = SaiCausalThreshold(abs_position, params.compress_ratio);
    const int64_t visible_64 = causal_count < key_len_after ? causal_count : static_cast<int64_t>(key_len_after);
    const int visible_block_count = static_cast<int>(visible_64 < 0 ? 0 : visible_64);
    const int selected = params.block_topk < visible_block_count ? params.block_topk : visible_block_count;

    const float* scores_row = block_scores + static_cast<int64_t>(token) * params.state_capacity;

    int emitted_blocks = selected;
    if (topk_indices != nullptr) {
      const int32_t* topk_row = topk_indices + static_cast<int64_t>(token) * params.state_capacity;
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = topk_row[rank];
        for (int t = static_cast<int>(threadIdx.x); t < params.compress_ratio;
             t += static_cast<int>(blockDim.x)) {
          out_row[rank * params.compress_ratio + t] = selected_block * params.compress_ratio + t;
        }
      }
      __syncthreads();
    } else if (selected > 0 && use_fast_topk) {
      SaiBlockTopK(scores_row, visible_block_count, selected, shared_value, shared_index);
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = shared_index[rank];
        for (int t = static_cast<int>(threadIdx.x); t < params.compress_ratio;
             t += static_cast<int>(blockDim.x)) {
          out_row[rank * params.compress_ratio + t] = selected_block * params.compress_ratio + t;
        }
      }
      __syncthreads();
    } else {
      float previous_score = 0.0f;
      int previous_index = -1;
      emitted_blocks = 0;
      for (int rank = 0; rank < selected; ++rank) {
        float best_value = 0.0f;
        int best_index = -1;
        SaiScanForNext(scores_row, visible_block_count, previous_score, previous_index, &best_value, &best_index);
        shared_value[threadIdx.x] = best_value;
        shared_index[threadIdx.x] = best_index;
        __syncthreads();
        SaiBlockArgMax(shared_value, shared_index);
        previous_index = shared_index[0];
        previous_score = shared_value[0];
        __syncthreads();
        if (previous_index < 0) {
          break;
        }
        for (int t = static_cast<int>(threadIdx.x); t < params.compress_ratio;
             t += static_cast<int>(blockDim.x)) {
          out_row[rank * params.compress_ratio + t] = previous_index * params.compress_ratio + t;
        }
        emitted_blocks = rank + 1;
        __syncthreads();
      }
    }

    // The trailing incomplete block is always causally visible in full up to this query's own
    // position; only its indices are needed (SparsePagedAttention reads the raw main cache).
    const int64_t block_start = static_cast<int64_t>(visible_block_count) * params.compress_ratio;
    const int64_t natural_tail = abs_position >= block_start ? (abs_position - block_start + 1) : 0;
    const int remaining_capacity = params.capacity - emitted_blocks * params.compress_ratio;
    const int64_t tail_count_64 = natural_tail < remaining_capacity ? natural_tail : remaining_capacity;
    const int tail_count = static_cast<int>(tail_count_64 < 0 ? 0 : tail_count_64);
    for (int t = static_cast<int>(threadIdx.x); t < tail_count; t += static_cast<int>(blockDim.x)) {
      out_row[emitted_blocks * params.compress_ratio + t] = static_cast<int32_t>(block_start + t);
    }
    if (threadIdx.x == 0) {
      selected_counts[token] = emitted_blocks * params.compress_ratio + tail_count;
    }
    __syncthreads();
  }
}

// One block per packed query token. Radix-selects a bounded candidate set, then sorts only that
// set. The output aliases the corresponding score row after all scores for that row have been read.
__global__ void QsaPartialTopKKernel(float* block_scores, const int32_t* cumulative_sequence_lengths,
                                     const int32_t* past_sequence_lengths, const int64_t* position_ids,
                                     const int32_t* present_state_lengths, const int32_t* overflow_flags,
                                     PackedSparseAttentionIndexerParams params) {
  using Sort = cub::BlockRadixSort<uint64_t, kThreads, kBoundedTopKItemsPerThread>;
  __shared__ union {
    uint32_t histogram[256];
    typename Sort::TempStorage sort;
  } temp;
  __shared__ uint64_t selected_keys[kBoundedTopKMax];
  __shared__ int32_t equal_warp_offsets[kThreads / kWarpSize + 1];
  __shared__ uint32_t prefix;
  __shared__ int remaining;
  __shared__ int gathered;
  __shared__ int equal_seen;

  for (int token = static_cast<int>(blockIdx.x); token < params.total_tokens; token += static_cast<int>(gridDim.x)) {
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    if (overflow_flags[batch] != 0) {
      continue;
    }
    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];
    const int64_t abs_position =
        params.has_position_ids
            ? position_ids[token]
            : static_cast<int64_t>(past_sequence_lengths[batch]) + (token - cumulative_sequence_lengths[batch]);
    const int64_t causal_count = SaiCausalThreshold(abs_position, params.compress_ratio);
    const int64_t visible_64 = causal_count < key_len_after ? causal_count : static_cast<int64_t>(key_len_after);
    const int block_count = static_cast<int>(visible_64 < 0 ? 0 : visible_64);
    const int selected = min(params.block_topk, block_count);
    if (selected == 0) {
      continue;
    }
    float* scores = block_scores + static_cast<int64_t>(token) * params.state_capacity;

    if (threadIdx.x == 0) {
      prefix = 0;
      remaining = selected;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int bucket = threadIdx.x; bucket < 256; bucket += blockDim.x) {
        temp.histogram[bucket] = 0;
      }
      __syncthreads();
      const uint32_t current_prefix = prefix;
      for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
        const uint32_t score_key = static_cast<uint32_t>(topk::PackStableSortKey(scores[index], 0) >> 32);
        if (shift == 24 || (score_key >> (shift + 8)) == (current_prefix >> (shift + 8))) {
          atomicAdd(&temp.histogram[(score_key >> shift) & 0xffu], 1u);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int rank = remaining;
        for (int bucket = 255; bucket >= 0; --bucket) {
          const int count = static_cast<int>(temp.histogram[bucket]);
          if (rank > count) {
            rank -= count;
          } else {
            prefix |= static_cast<uint32_t>(bucket) << shift;
            remaining = rank;
            break;
          }
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      gathered = 0;
    }
    __syncthreads();
    const uint32_t threshold = prefix;
    for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
      const uint64_t key = topk::PackStableSortKey(scores[index], index);
      if (static_cast<uint32_t>(key >> 32) > threshold) {
        const int slot = atomicAdd(&gathered, 1);
        selected_keys[slot] = key;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      equal_seen = 0;
    }
    __syncthreads();
    for (int base = 0; base < block_count && equal_seen < remaining; base += blockDim.x) {
      const int index = base + static_cast<int>(threadIdx.x);
      const bool equal = index < block_count &&
                         static_cast<uint32_t>(topk::PackStableSortKey(scores[index], 0) >> 32) == threshold;
      const unsigned int warp_mask = __ballot_sync(0xffffffffu, equal);
      const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
      const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
      const int lane_rank = __popc(warp_mask & (lane == 0 ? 0u : (1u << lane) - 1u));
      if (lane == 0) {
        equal_warp_offsets[warp] = __popc(warp_mask);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int tile_total = 0;
        for (int i = 0; i < kThreads / kWarpSize; ++i) {
          const int warp_count = equal_warp_offsets[i];
          equal_warp_offsets[i] = tile_total;
          tile_total += warp_count;
        }
        equal_warp_offsets[kThreads / kWarpSize] = tile_total;
      }
      __syncthreads();
      const int equal_rank = equal_seen + equal_warp_offsets[warp] + lane_rank;
      if (equal && equal_rank < remaining) {
        selected_keys[gathered + equal_rank] = topk::PackStableSortKey(scores[index], index);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        equal_seen += equal_warp_offsets[kThreads / kWarpSize];
      }
      __syncthreads();
    }

    uint64_t keys[kBoundedTopKItemsPerThread];
#pragma unroll
    for (int item = 0; item < kBoundedTopKItemsPerThread; ++item) {
      const int rank = threadIdx.x * kBoundedTopKItemsPerThread + item;
      keys[item] = rank < selected ? selected_keys[rank] : topk::kPaddingSortKey;
    }
    Sort(temp.sort).SortDescending(keys);
    int32_t* topk_row = reinterpret_cast<int32_t*>(scores);
#pragma unroll
    for (int item = 0; item < kBoundedTopKItemsPerThread; ++item) {
      const int rank = threadIdx.x * kBoundedTopKItemsPerThread + item;
      if (rank < selected) {
        topk_row[rank] = topk::UnpackStableSortIndex(keys[item]);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
// policy_mode = "csa"
// ---------------------------------------------------------------------------------------------

// One block per request: closes every new compression window (softmax-gated pool -> RMSNorm ->
// trailing RoPE -> append into fixed-capacity key_state) using the shared CsaWindowPlan helper,
// and publishes the raw overlap+leftover buffer. Reads only past_kv_buffer / past_gate_buffer /
// key / gate (never the present_* tensors), so it is correct whether or not present/past alias.
template <typename T>
__global__ void CsaUpdateStateKernel(const T* key, const T* gate, const T* key_norm_weight, const T* cos_cache,
                                     const T* sin_cache, const T* position_bias,
                                     const int32_t* cumulative_sequence_lengths,
                                     const int32_t* past_sequence_lengths, const T* past_kv_buffer,
                                     const T* past_gate_buffer, const int32_t* past_state_lengths,
                                     T* present_key_state, T* present_kv_buffer, T* present_gate_buffer,
                                     int32_t* present_state_lengths, int32_t* overflow_flags,
                                     PackedSparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* reduction = shared + params.head_size;
  const int width = 2 * params.head_size;

  for (int b = static_cast<int>(blockIdx.x); b < params.batch_size; b += static_cast<int>(gridDim.x)) {
    const int req_start = cumulative_sequence_lengths[b];
    const int req_end = cumulative_sequence_lengths[b + 1];
    const int raw_key_len = past_state_lengths[b * 2 + psai::kKeyStateLength];
    const int raw_buf_len = past_state_lengths[b * 2 + psai::kBufferLength];
    const bool invalid_metadata =
        cumulative_sequence_lengths[0] != 0 ||
        cumulative_sequence_lengths[params.batch_size] != params.total_tokens ||
        req_start < 0 || req_end < req_start || req_end > params.total_tokens ||
        past_sequence_lengths[b] < 0 || raw_key_len < 0 || raw_key_len > params.state_capacity ||
        raw_buf_len < 0 || raw_buf_len > params.buffer_capacity;
    const int req_len = invalid_metadata ? 0 : req_end - req_start;
    const int old_key_len = min(max(raw_key_len, 0), params.state_capacity);
    const int old_buf_len = min(max(raw_buf_len, 0), params.buffer_capacity);

    psai::CsaWindowPlan plan;
    const bool plan_ok = psai::TryComputeCsaWindowPlan(old_buf_len, req_len, params.compress_ratio, plan);
    // plan_ok is always true here: old_buf_len is clamped into [0, buffer_capacity) ==
    // [0, 2 * compress_ratio) and req_len >= 0, which are exactly the documented preconditions.
    const int full_new_window_count = plan_ok ? static_cast<int>(plan.new_window_count) : 0;
    const int capacity_left_raw = params.state_capacity - old_key_len;
    const int capacity_left = capacity_left_raw > 0 ? capacity_left_raw : 0;
    // Reject (do not partially apply) a step that would need more than the fixed state_capacity:
    // no new windows are closed and the buffer is left exactly as it was, so a rejected step is a
    // deterministic no-op on state rather than a silent partial truncation.
    const bool rejected = invalid_metadata || full_new_window_count > capacity_left;
    const int new_window_count = rejected ? 0 : full_new_window_count;
    const int present_buffer_length =
        rejected ? old_buf_len
                 : (static_cast<int>(plan.present_buffer_length) < params.buffer_capacity
                        ? static_cast<int>(plan.present_buffer_length)
                        : params.buffer_capacity);
    const int present_buffer_start = rejected ? 0 : static_cast<int>(plan.present_buffer_start);
    const int overlap_length = static_cast<int>(plan.overlap_length);

    // Barrier: every thread has now read past_state_lengths / computed the plan (identically)
    // before any thread below writes present_state_lengths, which keeps this correct even if the
    // two tensors alias.
    __syncthreads();

    if (threadIdx.x == 0) {
      present_state_lengths[b * 2 + psai::kKeyStateLength] = old_key_len + new_window_count;
      present_state_lengths[b * 2 + psai::kBufferLength] = present_buffer_length;
      overflow_flags[b] = rejected ? 1 : 0;
    }

    for (int k = 0; k < new_window_count; ++k) {
      const bool has_previous = k >= 1 || overlap_length >= params.compress_ratio;
      const int previous_base = overlap_length + (k - 1) * params.compress_ratio;
      const int current_base = overlap_length + k * params.compress_ratio;

      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        float max_gate = SaiNegativeInfinity();
        if (has_previous) {
          for (int slot = 0; slot < params.compress_ratio; ++slot) {
            const int virtual_pos = previous_base + slot;
            const float value =
                (virtual_pos < old_buf_len
                     ? to_float<T>(past_gate_buffer[(static_cast<int64_t>(b) * params.buffer_capacity +
                                                     virtual_pos) *
                                                        width +
                                                    d])
                     : to_float<T>(gate[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width +
                                        d])) +
                to_float<T>(position_bias[static_cast<int64_t>(slot) * width + d]);
            max_gate = fmaxf(max_gate, value);
          }
        }
        for (int slot = 0; slot < params.compress_ratio; ++slot) {
          const int virtual_pos = current_base + slot;
          const float value =
              (virtual_pos < old_buf_len
                   ? to_float<T>(past_gate_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) *
                                                      width +
                                                  params.head_size + d])
                   : to_float<T>(gate[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width +
                                      params.head_size + d])) +
              to_float<T>(position_bias[static_cast<int64_t>(slot) * width + params.head_size + d]);
          max_gate = fmaxf(max_gate, value);
        }

        float denominator = 0.0f;
        float accumulator = 0.0f;
        if (has_previous) {
          for (int slot = 0; slot < params.compress_ratio; ++slot) {
            const int virtual_pos = previous_base + slot;
            const float logit =
                (virtual_pos < old_buf_len
                     ? to_float<T>(past_gate_buffer[(static_cast<int64_t>(b) * params.buffer_capacity +
                                                     virtual_pos) *
                                                        width +
                                                    d])
                     : to_float<T>(gate[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width +
                                        d])) +
                to_float<T>(position_bias[static_cast<int64_t>(slot) * width + d]);
            const float weight = __expf(logit - max_gate);
            denominator += weight;
            accumulator +=
                weight *
                (virtual_pos < old_buf_len
                     ? to_float<T>(past_kv_buffer[(static_cast<int64_t>(b) * params.buffer_capacity +
                                                   virtual_pos) *
                                                      width +
                                                  d])
                     : to_float<T>(key[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width +
                                       d]));
          }
        }
        for (int slot = 0; slot < params.compress_ratio; ++slot) {
          const int virtual_pos = current_base + slot;
          const float logit =
              (virtual_pos < old_buf_len
                   ? to_float<T>(past_gate_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) *
                                                      width +
                                                  params.head_size + d])
                   : to_float<T>(gate[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width +
                                      params.head_size + d])) +
              to_float<T>(position_bias[static_cast<int64_t>(slot) * width + params.head_size + d]);
          const float weight = __expf(logit - max_gate);
          denominator += weight;
          accumulator +=
              weight * (virtual_pos < old_buf_len
                            ? to_float<T>(past_kv_buffer[(static_cast<int64_t>(b) * params.buffer_capacity +
                                                          virtual_pos) *
                                                             width +
                                                         params.head_size + d])
                            : to_float<T>(key[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) *
                                                  width +
                                              params.head_size + d]));
        }
        pooled[d] = denominator > 0.0f ? accumulator / denominator : 0.0f;
      }
      __syncthreads();

      float sum_squares = 0.0f;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        sum_squares += pooled[d] * pooled[d];
      }
      sum_squares = SaiBlockSum(sum_squares, reduction);
      const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        pooled[d] = pooled[d] * inverse_rms * to_float<T>(key_norm_weight[d]);
      }
      __syncthreads();

      const int entry = old_key_len + k;
      const int rope_position =
          SaiClampPosition(static_cast<int64_t>(entry) * params.compress_ratio, params.max_rotary_length);
      const int64_t cache_offset =
          (static_cast<int64_t>(params.cos_cache_batched ? b : 0) * params.max_rotary_length + rope_position) *
          params.rotary_width;
      const int64_t out_base = (static_cast<int64_t>(b) * params.state_capacity + entry) * params.head_size;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        present_key_state[out_base + d] = from_float<T>(SaiTrailingRope<T>(
            pooled, params.head_size, params.rotary_width, cos_cache + cache_offset, sin_cache + cache_offset, d));
      }
      __syncthreads();
    }

    // Publish the raw overlap+leftover buffer. Skipped entirely on overflow: present_kv_buffer /
    // present_gate_buffer already hold past_kv_buffer's / past_gate_buffer's contents unchanged
    // (from the baseline copy in the Launch function below), which is exactly the prior valid
    // buffer this rejected step must preserve.
    if (!rejected) {
      for (int t = static_cast<int>(threadIdx.x); t < present_buffer_length; t += static_cast<int>(blockDim.x)) {
        const int virtual_pos = present_buffer_start + t;
        const int64_t out_base = (static_cast<int64_t>(b) * params.buffer_capacity + t) * width;
        for (int c = 0; c < width; ++c) {
          const float key_value =
              virtual_pos < old_buf_len
                  ? to_float<T>(
                        past_kv_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) * width + c])
                  : to_float<T>(key[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width + c]);
          const float gate_value =
              virtual_pos < old_buf_len
                  ? to_float<T>(past_gate_buffer[(static_cast<int64_t>(b) * params.buffer_capacity + virtual_pos) *
                                                     width +
                                                 c])
                  : to_float<T>(gate[(static_cast<int64_t>(req_start) + (virtual_pos - old_buf_len)) * width + c]);
          present_kv_buffer[out_base + c] = from_float<T>(key_value);
          present_gate_buffer[out_base + c] = from_float<T>(gate_value);
        }
      }
    }
    __syncthreads();
  }
}

// One block per (token, key_state slot). Also applies the causal mask so the selection kernel only
// has to read scores.
template <typename T>
__global__ void CsaScoreKernel(const T* present_key_state, const float* query_rotated, const T* head_weights,
                               const int32_t* cumulative_sequence_lengths, const int64_t* position_ids,
                               const int32_t* present_state_lengths, float* scores,
                               PackedSparseAttentionIndexerParams params) {
  extern __shared__ float reduction[];
  const int64_t total = static_cast<int64_t>(params.total_tokens) * params.state_capacity;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int token = static_cast<int>(work / params.state_capacity);
    const int entry = static_cast<int>(work % params.state_capacity);
    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];

    if (entry >= key_len_after) {
      if (threadIdx.x == 0) {
        scores[work] = SaiNegativeInfinity();
      }
      continue;
    }
    const int64_t threshold = SaiCausalThreshold(position_ids[token], params.compress_ratio);
    if (static_cast<int64_t>(entry) >= threshold) {
      if (threadIdx.x == 0) {
        scores[work] = SaiNegativeInfinity();
      }
      continue;
    }

    const int64_t key_base = (static_cast<int64_t>(batch) * params.state_capacity + entry) * params.head_size;
    float total_score = 0.0f;
    for (int head = 0; head < params.num_heads; ++head) {
      const float* query_head =
          query_rotated + (static_cast<int64_t>(token) * params.num_heads + head) * params.head_size;
      float partial = 0.0f;
      for (int d = static_cast<int>(threadIdx.x); d < params.head_size; d += static_cast<int>(blockDim.x)) {
        partial += query_head[d] * to_float<T>(present_key_state[key_base + d]);
      }
      const float dot = SaiBlockSum(partial, reduction);
      if (threadIdx.x == 0) {
        total_score += fmaxf(dot, 0.0f) * to_float<T>(head_weights[static_cast<int64_t>(token) * params.num_heads +
                                                                   head]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      scores[work] = total_score * params.scale * params.head_weight_scale;
    }
    __syncthreads();
  }
}

// One block per query token. Selects the index_topk highest scoring, causally-visible compressed
// entries and the exact active count. A request whose update step was rejected for exceeding
// state_capacity this call (overflow_flags[batch] set) always gets the safe empty result: indices
// stay -1 (already reset below) and count is 0.
__global__ void CsaSelectKernel(const float* scores, const int32_t* cumulative_sequence_lengths,
                                const int64_t* position_ids, const int32_t* present_state_lengths,
                                const int32_t* overflow_flags, int32_t* selected_indices,
                                int32_t* selected_counts, PackedSparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* shared_value = shared;
  int* shared_index = reinterpret_cast<int*>(shared + blockDim.x);

  for (int token = static_cast<int>(blockIdx.x); token < params.total_tokens; token += static_cast<int>(gridDim.x)) {
    int32_t* out_row = selected_indices + static_cast<int64_t>(token) * params.capacity;
    for (int p = static_cast<int>(threadIdx.x); p < params.capacity; p += static_cast<int>(blockDim.x)) {
      out_row[p] = -1;
    }
    __syncthreads();

    const int batch = PackedBatchOfToken(cumulative_sequence_lengths, params.batch_size, token);
    if (overflow_flags[batch] != 0) {
      if (threadIdx.x == 0) {
        selected_counts[token] = 0;
      }
      __syncthreads();
      continue;
    }

    const int key_len_after = present_state_lengths[batch * 2 + psai::kKeyStateLength];
    const int64_t threshold = SaiCausalThreshold(position_ids[token], params.compress_ratio);
    const int64_t visible_64 = threshold < key_len_after ? threshold : static_cast<int64_t>(key_len_after);
    const int visible = static_cast<int>(visible_64 < 0 ? 0 : visible_64);
    const int selected = params.index_topk < visible ? params.index_topk : visible;

    const float* scores_row = scores + static_cast<int64_t>(token) * params.state_capacity;
    float previous_score = 0.0f;
    int previous_index = -1;
    int emitted = 0;
    for (int rank = 0; rank < selected; ++rank) {
      float best_value = 0.0f;
      int best_index = -1;
      SaiScanForNext(scores_row, visible, previous_score, previous_index, &best_value, &best_index);
      shared_value[threadIdx.x] = best_value;
      shared_index[threadIdx.x] = best_index;
      __syncthreads();
      SaiBlockArgMax(shared_value, shared_index);
      previous_index = shared_index[0];
      previous_score = shared_value[0];
      __syncthreads();
      if (previous_index < 0) {
        break;
      }
      if (threadIdx.x == 0) {
        out_row[rank] = previous_index;
      }
      emitted = rank + 1;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      selected_counts[token] = emitted;
    }
    __syncthreads();
  }
}

}  // namespace

size_t GetQsaPackedWorkspaceFloatCount(const PackedSparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.total_tokens);
  const size_t query_count = rows * params.num_heads * params.head_size;
  if (UseHierarchicalQsaTopK(params)) {
    const size_t key_count = rows * GetHierarchicalTileCount(params) * kHierarchicalTileBlocks;
    return query_count + 4 * key_count;
  }
  return query_count + rows * static_cast<size_t>(std::max(params.state_capacity, 1));
}

size_t GetCsaPackedWorkspaceFloatCount(const PackedSparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.total_tokens);
  return rows * params.num_heads * params.head_size + rows * static_cast<size_t>(std::max(params.state_capacity, 1));
}

template <typename T>
Status LaunchQsaPackedSparseAttentionIndexer(
    cudaStream_t stream, const PackedSparseAttentionIndexerParams& params, const T* query, const T* key,
    const T* query_norm_weight, const T* key_norm_weight, const T* cos_cache, const T* sin_cache,
    const int32_t* cumulative_sequence_lengths,
    const int32_t* past_sequence_lengths, const int64_t* position_ids, const T* past_key_state,
    const T* past_kv_buffer, const int32_t* past_state_lengths, int32_t* selected_indices,
    int32_t* selected_counts, T* present_key_state, T* present_kv_buffer, int32_t* present_state_lengths,
    float* float_workspace, int32_t* overflow_flags) {
  if (params.batch_size > 0) {
    const int64_t key_state_elems =
        static_cast<int64_t>(params.batch_size) * params.state_capacity * params.head_size;
    if (key_state_elems > 0 && present_key_state != past_key_state) {
      ElementwiseCopyKernel<T><<<SaiGridForElements(key_state_elems, kThreads), kThreads, 0, stream>>>(
          past_key_state, present_key_state, key_state_elems);
    }
    const int64_t buffer_elems = static_cast<int64_t>(params.batch_size) * params.buffer_capacity * params.head_size;
    if (buffer_elems > 0 && present_kv_buffer != past_kv_buffer) {
      ElementwiseCopyKernel<T><<<SaiGridForElements(buffer_elems, kThreads), kThreads, 0, stream>>>(
          past_kv_buffer, present_kv_buffer, buffer_elems);
    }
    if (present_state_lengths != past_state_lengths) {
      const int64_t length_elems = static_cast<int64_t>(params.batch_size) * psai::kStateLengthColumns;
      ElementwiseCopyKernel<int32_t><<<SaiGridForElements(length_elems, kThreads), kThreads, 0, stream>>>(
          past_state_lengths, present_state_lengths, length_elems);
    }
  }

  if (params.batch_size == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);
  const int state_blocks = static_cast<int>(std::min<int64_t>(params.batch_size, kSaiMaxGridDimX));
  QsaUpdateStateKernel<T><<<state_blocks, kThreads, 2 * value_bytes + kThreads * sizeof(float), stream>>>(
      key, key_norm_weight, cos_cache, sin_cache, cumulative_sequence_lengths, past_sequence_lengths, past_kv_buffer,
      past_state_lengths, present_key_state, present_kv_buffer, present_state_lengths, overflow_flags, params);

  if (params.total_tokens == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* selection_workspace =
      query_rotated + static_cast<int64_t>(params.total_tokens) * params.num_heads * params.head_size;
  const bool use_hierarchical_topk = UseHierarchicalQsaTopK(params);
  float* block_scores = use_hierarchical_topk ? nullptr : selection_workspace;

  const int64_t rotate_rows = static_cast<int64_t>(params.total_tokens) * params.num_heads;
  const int rotate_blocks = static_cast<int>(std::min<int64_t>(rotate_rows, kSaiMaxGridDimX));
  PackedRotateQueryKernel<T, true><<<rotate_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, cumulative_sequence_lengths, past_sequence_lengths,
      position_ids, query_rotated, params);

  const int token_blocks = static_cast<int>(std::min<int64_t>(params.total_tokens, kSaiMaxGridDimX));
  if (use_hierarchical_topk) {
    const int tile_count = GetHierarchicalTileCount(params);
    const int key_stride = tile_count * kHierarchicalTileBlocks;
    const size_t key_count = static_cast<size_t>(params.total_tokens) * key_stride;
    uint64_t* merge_input = reinterpret_cast<uint64_t*>(selection_workspace);
    uint64_t* merge_output = merge_input + key_count;
    const int64_t tile_work = static_cast<int64_t>(params.total_tokens) * tile_count;
    QsaScoreTileTopKKernel<T><<<static_cast<int>(std::min<int64_t>(tile_work, kSaiMaxGridDimX)),
                                kHierarchicalScoreThreads, 0, stream>>>(
        present_key_state, query_rotated, cumulative_sequence_lengths, past_sequence_lengths, position_ids,
        present_state_lengths, overflow_flags, merge_input, tile_count, params);
    for (int list_width = kHierarchicalTileBlocks; list_width < key_stride; list_width *= 2) {
      const int pairs_per_row = (key_stride + 2 * list_width - 1) / (2 * list_width);
      const int64_t merge_work = static_cast<int64_t>(params.total_tokens) * pairs_per_row;
      QsaMergeTileTopKKernel<<<static_cast<int>(std::min<int64_t>(merge_work, kSaiMaxGridDimX)),
                               kHierarchicalMergeThreads, 0, stream>>>(
          merge_input, merge_output, key_stride, list_width, params.total_tokens);
      std::swap(merge_input, merge_output);
    }
    QsaEmitHierarchicalTopKKernel<<<token_blocks, kThreads, 0, stream>>>(
        merge_input, cumulative_sequence_lengths, past_sequence_lengths, position_ids, present_state_lengths,
        overflow_flags, key_stride, selected_indices, selected_counts, params);
  } else if (params.state_capacity > 0) {
    const int64_t score_work = static_cast<int64_t>(params.total_tokens) * params.state_capacity;
    const int score_blocks = static_cast<int>(std::min<int64_t>(score_work, kSaiMaxGridDimX));
    QsaBlockScoreKernel<T><<<score_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        present_key_state, query_rotated, cumulative_sequence_lengths, past_sequence_lengths, position_ids,
        present_state_lengths, block_scores, params);
  }

  const bool use_bounded_topk =
      !use_hierarchical_topk && params.block_topk > kSaiFastTopKMax && params.block_topk <= kBoundedTopKMax;
  int32_t* topk_indices = use_bounded_topk ? reinterpret_cast<int32_t*>(block_scores) : nullptr;
  if (use_bounded_topk) {
    QsaPartialTopKKernel<<<token_blocks, kThreads, 0, stream>>>(
        block_scores, cumulative_sequence_lengths, past_sequence_lengths, position_ids, present_state_lengths,
        overflow_flags, params);
  }
  const int topk_shared_entries =
      !use_bounded_topk && params.block_topk <= kSaiFastTopKMax ? kThreads * params.block_topk : kThreads;
  if (!use_hierarchical_topk) {
    QsaSelectKernel<<<token_blocks, kThreads,
                      static_cast<size_t>(topk_shared_entries) * (sizeof(float) + sizeof(int)), stream>>>(
        block_scores, topk_indices, cumulative_sequence_lengths, past_sequence_lengths, position_ids,
        present_state_lengths, overflow_flags, selected_indices, selected_counts, params);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchCsaPackedSparseAttentionIndexer(
    cudaStream_t stream, const PackedSparseAttentionIndexerParams& params, const T* query, const T* key,
    const T* query_norm_weight, const T* key_norm_weight, const T* cos_cache, const T* sin_cache, const T* gate,
    const T* position_bias,
    const T* head_weights, const int32_t* cumulative_sequence_lengths, const int32_t* past_sequence_lengths,
    const int64_t* position_ids, const T* past_key_state, const T* past_kv_buffer, const T* past_gate_buffer,
    const int32_t* past_state_lengths, int32_t* selected_indices, int32_t* selected_counts, T* present_key_state,
    T* present_kv_buffer, T* present_gate_buffer, int32_t* present_state_lengths, float* float_workspace,
    int32_t* overflow_flags) {
  if (params.batch_size > 0) {
    const int64_t key_state_elems =
        static_cast<int64_t>(params.batch_size) * params.state_capacity * params.head_size;
    if (key_state_elems > 0 && present_key_state != past_key_state) {
      ElementwiseCopyKernel<T><<<SaiGridForElements(key_state_elems, kThreads), kThreads, 0, stream>>>(
          past_key_state, present_key_state, key_state_elems);
    }
    const int64_t buffer_elems =
        static_cast<int64_t>(params.batch_size) * params.buffer_capacity * 2 * params.head_size;
    if (buffer_elems > 0) {
      if (present_kv_buffer != past_kv_buffer) {
        ElementwiseCopyKernel<T><<<SaiGridForElements(buffer_elems, kThreads), kThreads, 0, stream>>>(
            past_kv_buffer, present_kv_buffer, buffer_elems);
      }
      if (present_gate_buffer != past_gate_buffer) {
        ElementwiseCopyKernel<T><<<SaiGridForElements(buffer_elems, kThreads), kThreads, 0, stream>>>(
            past_gate_buffer, present_gate_buffer, buffer_elems);
      }
    }
    if (present_state_lengths != past_state_lengths) {
      const int64_t length_elems = static_cast<int64_t>(params.batch_size) * psai::kStateLengthColumns;
      ElementwiseCopyKernel<int32_t><<<SaiGridForElements(length_elems, kThreads), kThreads, 0, stream>>>(
          past_state_lengths, present_state_lengths, length_elems);
    }
  }

  if (params.batch_size == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);
  const int state_blocks = static_cast<int>(std::min<int64_t>(params.batch_size, kSaiMaxGridDimX));
  CsaUpdateStateKernel<T><<<state_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
      key, gate, key_norm_weight, cos_cache, sin_cache, position_bias, cumulative_sequence_lengths,
      past_sequence_lengths, past_kv_buffer, past_gate_buffer, past_state_lengths, present_key_state,
      present_kv_buffer, present_gate_buffer, present_state_lengths, overflow_flags, params);

  if (params.total_tokens == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* scores = float_workspace + static_cast<int64_t>(params.total_tokens) * params.num_heads * params.head_size;

  const int64_t rotate_rows = static_cast<int64_t>(params.total_tokens) * params.num_heads;
  const int rotate_blocks = static_cast<int>(std::min<int64_t>(rotate_rows, kSaiMaxGridDimX));
  PackedRotateQueryKernel<T, false><<<rotate_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, cumulative_sequence_lengths, past_sequence_lengths,
      position_ids, query_rotated, params);

  if (params.state_capacity > 0) {
    const int64_t score_work = static_cast<int64_t>(params.total_tokens) * params.state_capacity;
    const int score_blocks = static_cast<int>(std::min<int64_t>(score_work, kSaiMaxGridDimX));
    CsaScoreKernel<T><<<score_blocks, kThreads, kThreads * sizeof(float), stream>>>(
        present_key_state, query_rotated, head_weights, cumulative_sequence_lengths, position_ids,
        present_state_lengths, scores, params);
  }

  const int token_blocks = static_cast<int>(std::min<int64_t>(params.total_tokens, kSaiMaxGridDimX));
  CsaSelectKernel<<<token_blocks, kThreads, kThreads * (sizeof(float) + sizeof(int)), stream>>>(
      scores, cumulative_sequence_lengths, position_ids, present_state_lengths, overflow_flags, selected_indices,
      selected_counts, params);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_PACKED_SPARSE_ATTENTION_INDEXER(T)                                                            \
  template Status LaunchQsaPackedSparseAttentionIndexer<T>(                                                       \
      cudaStream_t, const PackedSparseAttentionIndexerParams&, const T*, const T*, const T*, const T*,            \
      const T*, const T*, const int32_t*, const int32_t*, const int64_t*, const T*, const T*, const int32_t*,     \
      int32_t*,                                                                                                   \
      int32_t*, T*, T*, int32_t*, float*, int32_t*);                                                              \
  template Status LaunchCsaPackedSparseAttentionIndexer<T>(                                                       \
      cudaStream_t, const PackedSparseAttentionIndexerParams&, const T*, const T*, const T*, const T*,            \
      const T*, const T*, const T*, const T*, const T*, const int32_t*, const int32_t*, const int64_t*, const T*, \
      const T*, const T*, const int32_t*, int32_t*, int32_t*, T*, T*, T*, int32_t*, float*, int32_t*);

INSTANTIATE_PACKED_SPARSE_ATTENTION_INDEXER(float)
INSTANTIATE_PACKED_SPARSE_ATTENTION_INDEXER(half)
INSTANTIATE_PACKED_SPARSE_ATTENTION_INDEXER(__nv_bfloat16)

#undef INSTANTIATE_PACKED_SPARSE_ATTENTION_INDEXER

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
