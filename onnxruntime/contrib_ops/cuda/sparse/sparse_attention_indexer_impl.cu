// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Correctness-first implementation of com.microsoft.SparseAttentionIndexer. Every stage is a
// straightforward kernel that mirrors the reference semantics; see
// docs/contrib_ops/cuda/sparse_attention_indexer.md for the performance follow-ups.

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_radix_sort.cuh>

#include <algorithm>

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_device_math.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace topk = onnxruntime::cuda::topk;

namespace {

// The block reductions below halve the active thread count, so this must stay a power of two.
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int64_t kMaxGridDimX = kSaiMaxGridDimX;
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

// Thin, same-signature aliases over the device math shared with the packed indexer implementation
// (sparse_attention_indexer_device_math.cuh), so the kernels below are unchanged.
__device__ __forceinline__ float NegativeInfinity() { return SaiNegativeInfinity(); }

int GridForElements(int64_t count) { return SaiGridForElements(count, kThreads); }

bool UseDistributedQsaTopK(const SparseAttentionIndexerParams& params, bool has_mask) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  return !has_mask && rows > 0 && rows <= kDistributedTopKMaxRows &&
         params.max_block_count >= kDistributedTopKMinBlocks &&
      params.head_size == kQwenHeadSize && params.compress_ratio == kQwenCompressRatio &&
      params.num_heads == kQwenNumHeads &&
         params.block_topk > kSaiFastTopKMax && params.block_topk <= kBoundedTopKMax;
}

bool UseHierarchicalQsaTopK(const SparseAttentionIndexerParams& params, bool has_mask) {
  return params.use_block_representatives && UseDistributedQsaTopK(params, has_mask);
}

int GetHierarchicalTileCount(const SparseAttentionIndexerParams& params) {
  return (params.max_block_count + kHierarchicalTileBlocks - 1) / kHierarchicalTileBlocks;
}

__device__ __forceinline__ float BlockSum(float value, float* shared) { return SaiBlockSum(value, shared); }

__device__ __forceinline__ void BlockArgMax(float* shared_value, int* shared_index) {
  SaiBlockArgMax(shared_value, shared_index);
}

__device__ __forceinline__ void ScanForNext(const float* scores, int count, float previous_score,
                                            int previous_index, float* best_value, int* best_index) {
  SaiScanForNext(scores, count, previous_score, previous_index, best_value, best_index);
}

template <typename T>
__device__ __forceinline__ float LeadingRope(const float* value, int rotary_width, const T* cos_row,
                                             const T* sin_row, int d) {
  return SaiLeadingRope<T>(value, rotary_width, cos_row, sin_row, d);
}

template <typename T>
__device__ __forceinline__ float TrailingRope(const float* value, int head_size, int rotary_width,
                                              const T* cos_row, const T* sin_row, int d) {
  return SaiTrailingRope<T>(value, head_size, rotary_width, cos_row, sin_row, d);
}

__device__ __forceinline__ int64_t CausalThreshold(int64_t position, int compress_ratio) {
  return SaiCausalThreshold(position, compress_ratio);
}

__device__ __forceinline__ int ClampPosition(int64_t position, int max_rotary_length) {
  return SaiClampPosition(position, max_rotary_length);
}

// ---------------------------------------------------------------------------------------------
// Shared kernels
// ---------------------------------------------------------------------------------------------

// One block per (batch, token, head). kUseLeadingRope selects the qsa convention; otherwise the
// csa convention with positions taken from position_ids.
template <typename T, bool kUseLeadingRope>
__global__ void RotateQueryKernel(const T* query, const T* query_norm_weight,
                                  const T* cos_cache, const T* sin_cache,
                                  const int64_t* position_ids, float* query_rotated,
                                  SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* reduction = shared + params.head_size;
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.num_heads;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int token = static_cast<int>((row / params.num_heads) % params.sequence_length);
    const int batch = static_cast<int>(row / (static_cast<int64_t>(params.num_heads) * params.sequence_length));
    const int64_t base = row * params.head_size;
    const int64_t source_base = (row / params.num_heads) * params.query_row_stride +
                                (row % params.num_heads) * params.head_size;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      shared[d] = to_float<T>(query[source_base + d]);
    }
    __syncthreads();

    float sum_squares = 0.0f;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      sum_squares += shared[d] * shared[d];
    }
    sum_squares = BlockSum(sum_squares, reduction);
    const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      shared[d] = shared[d] * inverse_rms * to_float<T>(query_norm_weight[d]);
    }
    __syncthreads();

    const int64_t raw_position = kUseLeadingRope
                                     ? static_cast<int64_t>(params.past_sequence_length) + token
                                     : position_ids[static_cast<int64_t>(batch) * params.sequence_length + token];
    const int position = ClampPosition(raw_position, params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(batch) * params.rotary_cache_batch_stride + position) * params.rotary_width;
    const T* cos_row = cos_cache + cache_offset;
    const T* sin_row = sin_cache + cache_offset;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      query_rotated[base + d] =
          kUseLeadingRope ? LeadingRope<T>(shared, params.rotary_width, cos_row, sin_row, d)
                          : TrailingRope<T>(shared, params.head_size, params.rotary_width, cos_row, sin_row, d);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
// policy_mode = "qsa"
// ---------------------------------------------------------------------------------------------

template <typename T>
__global__ void CopyQsaPastKeyKernel(const T* past_key, T* present_key, SparseAttentionIndexerParams params) {
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.past_key_capacity * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(index % params.head_size);
    const int64_t token_row = index / params.head_size;
    const int token = static_cast<int>(token_row % params.past_key_capacity);
    const int batch = static_cast<int>(token_row / params.past_key_capacity);
    const int64_t output_index =
        (static_cast<int64_t>(batch) * params.key_cache_capacity + token) * params.head_size + d;
    present_key[output_index] = past_key[index];
  }
}

template <typename T>
__global__ void AppendQsaKeyKernel(const T* key, T* present_key, SparseAttentionIndexerParams params) {
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(index % params.head_size);
    const int64_t token_row = index / params.head_size;
    const int token = static_cast<int>(token_row % params.sequence_length);
    const int batch = static_cast<int>(token_row / params.sequence_length);
    present_key[(static_cast<int64_t>(batch) * params.key_cache_capacity + params.past_sequence_length + token) *
                    params.head_size +
                d] = key[token_row * params.key_row_stride + d];
  }
}

template <typename T>
__global__ void AppendQsaKeyTailKernel(const T* key, T* present_key, SparseAttentionIndexerParams params) {
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(index % params.head_size);
    const int64_t token_row = index / params.head_size;
    const int token = static_cast<int>(token_row % params.sequence_length);
    const int batch = static_cast<int>(token_row / params.sequence_length);
    const int position = params.past_sequence_length + token;
    const int completed_tokens = params.total_sequence_length / kQwenCompressRatio * kQwenCompressRatio;
    if (position >= completed_tokens) {
      const int representative_capacity = params.key_cache_capacity / kQwenCompressRatio;
      const int cache_position = representative_capacity + position % kQwenCompressRatio;
      present_key[(static_cast<int64_t>(batch) * params.key_cache_capacity + cache_position) * params.head_size + d] =
          key[token_row * params.key_row_stride + d];
    }
  }
}

// Shared-buffer Qwen caches are opaque state. Completed block representatives occupy a contiguous
// prefix, followed by four scratch slots that carry the current incomplete raw block.
template <typename T>
__global__ void QsaBuildBlockRepresentativesKernel(const T* past_key, const T* key, T* present_key,
                                                   const T* key_norm_weight,
                                                   const T* cos_cache, const T* sin_cache,
                                                   SparseAttentionIndexerParams params) {
  __shared__ float pooled[kQwenHeadSize];
  __shared__ float reduction[kThreads];
  const int first_block = params.past_sequence_length / kQwenCompressRatio;
  const int completed_blocks = params.total_sequence_length / kQwenCompressRatio;
  const int blocks_per_batch = completed_blocks - first_block;
  const int total_blocks = params.batch_size * blocks_per_batch;
  for (int work = blockIdx.x; work < total_blocks; work += gridDim.x) {
    const int batch = work / blocks_per_batch;
    const int block = first_block + work % blocks_per_batch;
    const int first_position = block * kQwenCompressRatio;
    const int representative_capacity = params.key_cache_capacity / kQwenCompressRatio;
    for (int d = threadIdx.x; d < kQwenHeadSize; d += blockDim.x) {
      float sum = 0.0f;
#pragma unroll
      for (int token = 0; token < kQwenCompressRatio; ++token) {
        const int position = first_position + token;
        if (position < params.past_sequence_length) {
          const int cache_position = representative_capacity + position % kQwenCompressRatio;
          sum += to_float<T>(past_key[(static_cast<int64_t>(batch) * params.past_key_capacity + cache_position) *
                                         kQwenHeadSize + d]);
        } else {
          const int key_token = position - params.past_sequence_length;
          sum += to_float<T>(key[(static_cast<int64_t>(batch) * params.sequence_length + key_token) *
                                     params.key_row_stride + d]);
        }
      }
      pooled[d] = sum * (1.0f / static_cast<float>(kQwenCompressRatio));
    }
    __syncthreads();
    float sum_squares = 0.0f;
    for (int d = threadIdx.x; d < kQwenHeadSize; d += blockDim.x) {
      sum_squares += pooled[d] * pooled[d];
    }
    sum_squares = BlockSum(sum_squares, reduction);
    const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(kQwenHeadSize) + params.epsilon);
    for (int d = threadIdx.x; d < kQwenHeadSize; d += blockDim.x) {
      pooled[d] *= inverse_rms * to_float<T>(key_norm_weight[d]);
    }
    __syncthreads();
    const int position = ClampPosition(first_position, params.max_rotary_length);
    const int64_t rotary_offset =
        (static_cast<int64_t>(batch) * params.rotary_cache_batch_stride + position) * params.rotary_width;
    for (int d = threadIdx.x; d < kQwenHeadSize; d += blockDim.x) {
      const int64_t output_base =
          (static_cast<int64_t>(batch) * params.key_cache_capacity + block) * kQwenHeadSize;
      present_key[output_base + d] = from_float<T>(LeadingRope<T>(
          pooled, params.rotary_width, cos_cache + rotary_offset, sin_cache + rotary_offset, d));
    }
    __syncthreads();
  }
}

// Encodes a prefix-visible row as -count-1. QSA can then index that row directly without
// materializing its visible positions. Non-prefix rows keep a nonnegative count and are compacted
// by CompactNonPrefixVisibleKernel.
__device__ __forceinline__ bool IsVisible(const int64_t* mask, int64_t row, int position,
                                          const SparseAttentionIndexerParams& params) {
  const int batch = static_cast<int>(row / params.sequence_length);
  const int query = static_cast<int>(row % params.sequence_length);
  return mask[static_cast<int64_t>(batch) * params.total_sequence_length + position] != 0 &&
         position <= params.past_sequence_length + query;
}

__global__ void AnalyzeVisibleKernel(const int64_t* mask, int32_t* visible_count,
                                     SparseAttentionIndexerParams params) {
  __shared__ int32_t counts[kThreads];
  __shared__ int32_t maxima[kThreads];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t count = 0;
    int32_t maximum = -1;
    for (int position = threadIdx.x; position < params.total_sequence_length; position += blockDim.x) {
      if (IsVisible(mask, row, position, params)) {
        ++count;
        maximum = position;
      }
    }
    counts[threadIdx.x] = count;
    maxima[threadIdx.x] = maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        counts[threadIdx.x] += counts[threadIdx.x + stride];
        maxima[threadIdx.x] = max(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const int32_t visible = counts[0];
      visible_count[row] = maxima[0] == visible - 1 ? -visible - 1 : visible;
    }
    __syncthreads();
  }
}

__global__ void InitializePrefixVisibleCountKernel(int32_t* visible_count,
                                                   SparseAttentionIndexerParams params) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int query = static_cast<int>(row % params.sequence_length);
    visible_count[row] = -(params.past_sequence_length + query + 1) - 1;
  }
}

// One block per non-prefix query row; compacts visible key positions into visible_indices.
__global__ void CompactNonPrefixVisibleKernel(const int64_t* mask, int32_t* visible_indices,
                                              const int32_t* visible_count,
                                              SparseAttentionIndexerParams params) {
  __shared__ int32_t warp_offsets[kThreads / kWarpSize + 1];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    if (visible_count[row] < 0) {
      continue;
    }
    int32_t* out_row = visible_indices + row * params.total_sequence_length;
    int32_t offset = 0;
    for (int base = 0; base < params.total_sequence_length; base += blockDim.x) {
      const int position = base + static_cast<int>(threadIdx.x);
      const bool visible = position < params.total_sequence_length &&
                           IsVisible(mask, row, position, params);
      const unsigned int warp_mask = __ballot_sync(0xffffffffu, visible);
      const int lane = threadIdx.x % kWarpSize;
      const int warp = threadIdx.x / kWarpSize;
      const int lane_rank = __popc(warp_mask & (lane == 0 ? 0u : (1u << lane) - 1u));
      if (lane == 0) {
        warp_offsets[warp] = __popc(warp_mask);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        int32_t tile_total = 0;
        for (int i = 0; i < kThreads / kWarpSize; ++i) {
          const int32_t warp_count = warp_offsets[i];
          warp_offsets[i] = tile_total;
          tile_total += warp_count;
        }
        warp_offsets[kThreads / kWarpSize] = tile_total;
      }
      __syncthreads();
      const int32_t warp_offset = warp_offsets[warp];
      const int32_t tile_total = warp_offsets[kThreads / kWarpSize];
      if (visible) {
        out_row[offset + warp_offset + lane_rank] = position;
      }
      offset += tile_total;
    }
    __syncthreads();
  }
}

// One block per (query row, block index). Pools compress_ratio visible keys, normalizes, rotates
// and scores the result against every query head.
template <typename T>
__global__ void QsaBlockScoreKernel(const float* query_rotated, const T* present_key,
                                    const T* key_norm_weight,
                                    const T* cos_cache, const T* sin_cache,
                                    const int32_t* visible_indices, const int32_t* visible_count,
                                    float* block_scores, SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* rotated = shared + params.head_size;
  float* reduction = shared + 2 * params.head_size;

  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.max_block_count;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int block_index = static_cast<int>(work % params.max_block_count);
    const int64_t row = work / params.max_block_count;
    const int batch = static_cast<int>(row / params.sequence_length);
    const int encoded_visible = visible_count[row];
    const bool prefix_visible = encoded_visible < 0;
    const int visible = prefix_visible ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;

    if (block_index >= block_count) {
      if (threadIdx.x == 0) {
        block_scores[row * params.max_block_count + block_index] = NegativeInfinity();
      }
      continue;
    }

    const int32_t* index_row = prefix_visible ? nullptr : visible_indices + row * params.total_sequence_length;
    const int32_t* group = prefix_visible ? nullptr
                                          : index_row + static_cast<int64_t>(block_index) * params.compress_ratio;
    const int64_t key_base = static_cast<int64_t>(batch) * params.key_cache_capacity * params.head_size;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      float sum = 0.0f;
      for (int t = 0; t < params.compress_ratio; ++t) {
        const int position = prefix_visible ? block_index * params.compress_ratio + t : group[t];
        sum += to_float<T>(present_key[key_base + static_cast<int64_t>(position) * params.head_size + d]);
      }
      pooled[d] = sum / static_cast<float>(params.compress_ratio);
    }
    __syncthreads();

    float sum_squares = 0.0f;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      sum_squares += pooled[d] * pooled[d];
    }
    sum_squares = BlockSum(sum_squares, reduction);
    const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      pooled[d] = pooled[d] * inverse_rms * to_float<T>(key_norm_weight[d]);
    }
    __syncthreads();

    const int first_position = prefix_visible ? block_index * params.compress_ratio : group[0];
    const int position = ClampPosition(first_position, params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(batch) * params.rotary_cache_batch_stride + position) * params.rotary_width;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      rotated[d] = LeadingRope<T>(pooled, params.rotary_width, cos_cache + cache_offset,
                                  sin_cache + cache_offset, d);
    }
    __syncthreads();

    float score = 0.0f;
    for (int head = 0; head < params.num_heads; ++head) {
      const int64_t query_base = (row * params.num_heads + head) * params.head_size;
      float partial = 0.0f;
      for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
        partial += query_rotated[query_base + d] * rotated[d];
      }
      score += fmaxf(BlockSum(partial, reduction), 0.0f);
    }

    if (threadIdx.x == 0) {
      const float scaled_score = score * params.scale;
      block_scores[row * params.max_block_count + block_index] = scaled_score == 0.0f ? 0.0f : scaled_score;
    }
    __syncthreads();
  }
}

// Qwen's common QSA layout has four query heads of width 128 and pools four keys per block.
// One warp prepares the vectorized pooled key, then each query head is scored by one warp.
template <typename T>
__global__ void QsaBlockScoreQwenKernel(const float* query_rotated, const T* present_key,
                                        const T* key_norm_weight,
                                        const T* cos_cache, const T* sin_cache,
                                        const int32_t* visible_indices, const int32_t* visible_count,
                                        float* block_scores, uint32_t* high_histogram,
                                        SparseAttentionIndexerParams params) {
  __shared__ float pooled[kQwenHeadSize];
  __shared__ float rotated[kQwenHeadSize];
  __shared__ float head_scores[kQwenNumHeads];
  __shared__ float inverse_rms;

  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.max_block_count;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int block_index = static_cast<int>(work % params.max_block_count);
    const int64_t row = work / params.max_block_count;
    const int batch = static_cast<int>(row / params.sequence_length);
    const int encoded_visible = visible_count[row];
    const bool prefix_visible = encoded_visible < 0;
    const int visible = prefix_visible ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / kQwenCompressRatio;

    if (block_index >= block_count) {
      if (threadIdx.x == 0) {
        block_scores[row * params.max_block_count + block_index] = NegativeInfinity();
      }
      continue;
    }

    const int32_t* group = prefix_visible
                               ? nullptr
                               : visible_indices + row * params.total_sequence_length +
                                     static_cast<int64_t>(block_index) * kQwenCompressRatio;
    const int64_t key_base = static_cast<int64_t>(batch) * params.key_cache_capacity * kQwenHeadSize;

    if (warp == 0) {
      const int d = lane * 4;
      float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int token = 0; token < kQwenCompressRatio; ++token) {
        const int position = prefix_visible ? block_index * kQwenCompressRatio + token : group[token];
        const float4 value = LoadQsaVector4(
            present_key + key_base + static_cast<int64_t>(position) * kQwenHeadSize + d);
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
      }
      constexpr float kInverseCompressRatio = 1.0f / static_cast<float>(kQwenCompressRatio);
      pooled[d] = sum.x * kInverseCompressRatio;
      pooled[d + 1] = sum.y * kInverseCompressRatio;
      pooled[d + 2] = sum.z * kInverseCompressRatio;
      pooled[d + 3] = sum.w * kInverseCompressRatio;

      float sum_squares = pooled[d] * pooled[d] + pooled[d + 1] * pooled[d + 1] +
                          pooled[d + 2] * pooled[d + 2] + pooled[d + 3] * pooled[d + 3];
      sum_squares = topk::WarpReduceSum(sum_squares);
      if (lane == 0) {
        inverse_rms = rsqrtf(sum_squares / static_cast<float>(kQwenHeadSize) + params.epsilon);
      }
    }
    __syncthreads();

    if (warp == 0) {
      const int d = lane * 4;
      const float4 weight = LoadQsaVector4(key_norm_weight + d);
      pooled[d] *= inverse_rms * weight.x;
      pooled[d + 1] *= inverse_rms * weight.y;
      pooled[d + 2] *= inverse_rms * weight.z;
      pooled[d + 3] *= inverse_rms * weight.w;
    }
    __syncthreads();

    if (warp == 0) {
      const int d = lane * 4;
      const int first_position = prefix_visible ? block_index * kQwenCompressRatio : group[0];
      const int position = ClampPosition(first_position, params.max_rotary_length);
      const int64_t cache_offset =
          (static_cast<int64_t>(batch) * params.rotary_cache_batch_stride + position) * params.rotary_width;
#pragma unroll
      for (int element = 0; element < 4; ++element) {
        rotated[d + element] = LeadingRope<T>(pooled, params.rotary_width, cos_cache + cache_offset,
                                              sin_cache + cache_offset, d + element);
      }
    }
    __syncthreads();

    const int d = lane * 4;
    const int64_t query_base = (row * kQwenNumHeads + warp) * kQwenHeadSize;
    const float4 query_value = LoadQsaVector4(query_rotated + query_base + d);
    const float4 key_value = LoadQsaVector4(rotated + d);
    float dot = query_value.x * key_value.x + query_value.y * key_value.y +
                query_value.z * key_value.z + query_value.w * key_value.w;
    dot = topk::WarpReduceSum(dot);
    if (lane == 0) {
      head_scores[warp] = fmaxf(dot, 0.0f);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float score = 0.0f;
#pragma unroll
      for (int head = 0; head < kQwenNumHeads; ++head) {
        score += head_scores[head];
      }
      const float scaled_score = score * params.scale;
      const float stored_score = scaled_score == 0.0f ? 0.0f : scaled_score;
      block_scores[row * params.max_block_count + block_index] = stored_score;
      if (high_histogram != nullptr) {
        const uint32_t score_key = static_cast<uint32_t>(topk::PackStableSortKey(stored_score, 0) >> 32);
        atomicAdd(high_histogram + row * 256 + (score_key >> 24), 1u);
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void QsaBlockScoreQwenCachedKernel(const float* query_rotated, const T* present_key,
                                              const int32_t* visible_count, float* block_scores,
                                              uint32_t* high_histogram,
                                              SparseAttentionIndexerParams params) {
  __shared__ float head_scores[kQwenNumHeads];
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.max_block_count;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int block_index = static_cast<int>(work % params.max_block_count);
    const int64_t row = work / params.max_block_count;
    const int batch = static_cast<int>(row / params.sequence_length);
    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / kQwenCompressRatio;
    if (block_index >= block_count) {
      if (threadIdx.x == 0) {
        block_scores[row * params.max_block_count + block_index] = NegativeInfinity();
      }
      continue;
    }

    const int d = lane * 4;
    const int64_t query_base = (row * kQwenNumHeads + warp) * kQwenHeadSize;
    const int64_t key_base =
      (static_cast<int64_t>(batch) * params.key_cache_capacity + block_index) *
        kQwenHeadSize;
    const float4 query_value = LoadQsaVector4(query_rotated + query_base + d);
    const float4 key_value = LoadQsaVector4(present_key + key_base + d);
    float dot = query_value.x * key_value.x + query_value.y * key_value.y +
                query_value.z * key_value.z + query_value.w * key_value.w;
    dot = topk::WarpReduceSum(dot);
    if (lane == 0) {
      head_scores[warp] = fmaxf(dot, 0.0f);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      float score = 0.0f;
#pragma unroll
      for (int head = 0; head < kQwenNumHeads; ++head) {
        score += head_scores[head];
      }
      const float scaled_score = score * params.scale;
      const float stored_score = scaled_score == 0.0f ? 0.0f : scaled_score;
      block_scores[row * params.max_block_count + block_index] = stored_score;
      if (high_histogram != nullptr) {
        const uint32_t score_key = static_cast<uint32_t>(topk::PackStableSortKey(stored_score, 0) >> 32);
        atomicAdd(high_histogram + row * 256 + (score_key >> 24), 1u);
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void QsaScoreTileTopKKernel(const float* query_rotated, const T* present_key,
                                      const int32_t* visible_count, uint64_t* tile_keys,
                                      int tile_count, SparseAttentionIndexerParams params) {
  __shared__ uint64_t keys[kHierarchicalTileBlocks];
  __shared__ float head_scores[kHierarchicalTileBlocks][kQwenNumHeads];

  const int64_t total_tiles =
      static_cast<int64_t>(params.batch_size) * params.sequence_length * tile_count;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int block_slot = warp / kQwenNumHeads;
  const int head = warp % kQwenNumHeads;
  for (int64_t work = blockIdx.x; work < total_tiles; work += gridDim.x) {
    const int tile = static_cast<int>(work % tile_count);
    const int64_t row = work / tile_count;
    const int batch = static_cast<int>(row / params.sequence_length);
    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / kQwenCompressRatio;

    const int block_index = tile * kHierarchicalTileBlocks + block_slot;
    float score = 0.0f;
    if (block_index < block_count) {
      const int d = lane * 4;
      const int64_t query_base = (row * kQwenNumHeads + head) * kQwenHeadSize;
      const int64_t key_base =
          (static_cast<int64_t>(batch) * params.key_cache_capacity + block_index) * kQwenHeadSize;
      const float4 query_value = LoadQsaVector4(query_rotated + query_base + d);
      const float4 key_value = LoadQsaVector4(present_key + key_base + d);
      score = query_value.x * key_value.x + query_value.y * key_value.y +
              query_value.z * key_value.z + query_value.w * key_value.w;
      score = topk::WarpReduceSum(score);
    }
    if (lane == 0) {
      head_scores[block_slot][head] = fmaxf(score, 0.0f);
    }
    __syncthreads();
    if (threadIdx.x < kHierarchicalTileBlocks) {
      uint64_t key = topk::kPaddingSortKey;
      if (block_index - block_slot + threadIdx.x < block_count) {
        float total_score = 0.0f;
#pragma unroll
        for (int query_head = 0; query_head < kQwenNumHeads; ++query_head) {
          total_score += head_scores[threadIdx.x][query_head];
        }
        const float scaled_score = total_score * params.scale;
        key = topk::PackStableSortKey(scaled_score == 0.0f ? 0.0f : scaled_score,
                                      block_index - block_slot + threadIdx.x);
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
                                       int key_stride, int list_width,
                                      SparseAttentionIndexerParams params) {
  const int pairs_per_row = (key_stride + 2 * list_width - 1) / (2 * list_width);
  const int64_t total_pairs =
      static_cast<int64_t>(params.batch_size) * params.sequence_length * pairs_per_row;
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
                                              const int32_t* visible_count,
                                              int key_stride,
                                              int32_t* selected_indices,
                                              SparseAttentionIndexerParams params) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t* output = selected_indices + row * params.capacity;
    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    for (int rank = threadIdx.x; rank < selected; rank += blockDim.x) {
      const int selected_block = topk::UnpackStableSortIndex(
          tile_keys[row * key_stride + rank]);
      for (int token = 0; token < params.compress_ratio; ++token) {
        output[rank * params.compress_ratio + token] = selected_block * params.compress_ratio + token;
      }
    }
    const int tail_start = block_count * params.compress_ratio;
    for (int token = threadIdx.x; token < visible - tail_start; token += blockDim.x) {
      output[selected * params.compress_ratio + token] = tail_start + token;
    }
    for (int position = selected * params.compress_ratio + visible - tail_start + threadIdx.x;
         position < params.capacity; position += blockDim.x) {
      output[position] = -1;
    }
  }
}

__global__ void QsaChooseRadixBucketKernel(const uint32_t* histogram, uint64_t* prefixes,
                                           int32_t* remaining, int shift,
                                           SparseAttentionIndexerParams params) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    if (threadIdx.x == 0) {
      int rank = shift == 56 ? min(params.block_topk, params.max_block_count) : remaining[row];
      uint64_t prefix = shift == 56 ? 0u : prefixes[row];
      for (int bucket = 255; bucket >= 0; --bucket) {
        const int count = static_cast<int>(histogram[row * 256 + bucket]);
        if (rank > count) {
          rank -= count;
        } else {
          prefix |= static_cast<uint64_t>(bucket) << shift;
          prefixes[row] = prefix;
          remaining[row] = rank;
          break;
        }
      }
    }
  }
}

__global__ void QsaCompactHighBucketKernel(const float* block_scores, const int32_t* visible_count,
                                           const uint64_t* prefixes, int32_t* candidate_counts,
                                           int32_t* candidate_indices,
                                           SparseAttentionIndexerParams params) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  const int64_t total = rows * params.max_block_count;
  for (int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; work < total;
       work += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int index = static_cast<int>(work % params.max_block_count);
    const int64_t row = work / params.max_block_count;
    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;
    if (index < block_count) {
      const float score = block_scores[row * params.max_block_count + index];
      const uint64_t key = topk::PackStableSortKey(score, index);
      if ((key >> 56) >= (prefixes[row] >> 56)) {
        const int slot = atomicAdd(candidate_counts + row, 1);
        candidate_indices[row * params.max_block_count + slot] = index;
      }
    }
  }
}

__global__ void QsaCandidateHistogramKernel(const float* block_scores, const int32_t* candidate_counts,
                                            const int32_t* candidate_indices, const uint64_t* prefixes,
                                            uint32_t* histogram, int shift,
                                            SparseAttentionIndexerParams params) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  const int64_t total = rows * params.max_block_count;
  for (int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; work < total;
       work += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int slot = static_cast<int>(work % params.max_block_count);
    const int64_t row = work / params.max_block_count;
    if (slot < candidate_counts[row]) {
      const int index = candidate_indices[row * params.max_block_count + slot];
      const float score = block_scores[row * params.max_block_count + index];
      const uint64_t key = topk::PackStableSortKey(score, index);
      if ((key >> (shift + 8)) == (prefixes[row] >> (shift + 8))) {
        atomicAdd(histogram + row * 256 + ((key >> shift) & 0xffu), 1u);
      }
    }
  }
}

// Gathers the bounded winner set from the compacted high radix bucket, sorts it, and writes token
// indices directly. Stable score ties prefer lower block indices, matching the reference path.
__global__ void QsaDistributedSelectKernel(const float* block_scores, const int32_t* visible_count,
                                           const int32_t* candidate_counts,
                                           const int32_t* candidate_indices,
                                           const uint64_t* prefixes,
                                           int32_t* selected_indices,
                                           SparseAttentionIndexerParams params) {
  using Sort = cub::BlockRadixSort<uint64_t, kThreads, kBoundedTopKItemsPerThread>;
  __shared__ typename Sort::TempStorage sort_temp;
  __shared__ uint64_t selected_keys[kBoundedTopKMax];
  __shared__ int gathered;

  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t* output = selected_indices + row * params.capacity;
    for (int position = threadIdx.x; position < params.capacity; position += blockDim.x) {
      output[position] = -1;
    }
    if (threadIdx.x == 0) {
      gathered = 0;
    }
    __syncthreads();

    const uint64_t threshold = prefixes[row];
    const int candidate_count = candidate_counts[row];
    for (int slot = threadIdx.x; slot < candidate_count; slot += blockDim.x) {
      const int index = candidate_indices[row * params.max_block_count + slot];
      const uint64_t key = topk::PackStableSortKey(
          block_scores[row * params.max_block_count + index], index);
      if (key >= threshold) {
        const int output_slot = atomicAdd(&gathered, 1);
        selected_keys[output_slot] = key;
      }
    }
    __syncthreads();

    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    uint64_t keys[kBoundedTopKItemsPerThread];
#pragma unroll
    for (int item = 0; item < kBoundedTopKItemsPerThread; ++item) {
      const int rank = threadIdx.x * kBoundedTopKItemsPerThread + item;
      keys[item] = rank < selected ? selected_keys[rank] : topk::kPaddingSortKey;
    }
    Sort(sort_temp).SortDescending(keys);
#pragma unroll
    for (int item = 0; item < kBoundedTopKItemsPerThread; ++item) {
      const int rank = threadIdx.x * kBoundedTopKItemsPerThread + item;
      if (rank < selected) {
        const int selected_block = topk::UnpackStableSortIndex(keys[item]);
        for (int token = 0; token < params.compress_ratio; ++token) {
          output[rank * params.compress_ratio + token] =
              selected_block * params.compress_ratio + token;
        }
      }
    }
    const int tail_start = block_count * params.compress_ratio;
    for (int token = threadIdx.x; token < visible - tail_start; token += blockDim.x) {
      output[selected * params.compress_ratio + token] = tail_start + token;
    }
    __syncthreads();
  }
}

// One block per query row. Emits the token indices of the highest scoring blocks followed by the
// visible tokens of the trailing incomplete block.
__global__ void QsaSelectKernel(const float* block_scores, const int32_t* visible_indices,
                                const int32_t* visible_count, const int32_t* topk_indices,
                                int32_t* selected_indices,
                                SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* shared_value = shared;
  const bool use_fast_topk = params.block_topk <= kSaiFastTopKMax;
  const int shared_entries = use_fast_topk ? static_cast<int>(blockDim.x) * params.block_topk
                                           : static_cast<int>(blockDim.x);
  int* shared_index = reinterpret_cast<int*>(shared + shared_entries);

  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t* out_row = selected_indices + row * params.capacity;
    for (int position = threadIdx.x; position < params.capacity; position += blockDim.x) {
      out_row[position] = -1;
    }
    __syncthreads();

    const int encoded_visible = visible_count[row];
    const bool prefix_visible = encoded_visible < 0;
    const int visible = prefix_visible ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    const float* scores_row = block_scores + row * params.max_block_count;
    const int32_t* index_row = visible_indices + row * params.total_sequence_length;

    int emitted = selected;
    if (topk_indices != nullptr) {
      const int32_t* topk_row = topk_indices + row * params.block_topk;
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = static_cast<int>(topk_row[rank]);
        for (int t = threadIdx.x; t < params.compress_ratio; t += blockDim.x) {
          out_row[rank * params.compress_ratio + t] =
              prefix_visible ? selected_block * params.compress_ratio + t
                             : index_row[selected_block * params.compress_ratio + t];
        }
      }
      __syncthreads();
    } else if (selected > 0 && use_fast_topk) {
      SaiBlockTopK(scores_row, block_count, selected, shared_value, shared_index);
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = shared_index[rank];
        for (int t = threadIdx.x; t < params.compress_ratio; t += blockDim.x) {
          out_row[rank * params.compress_ratio + t] =
              prefix_visible ? selected_block * params.compress_ratio + t
                             : index_row[selected_block * params.compress_ratio + t];
        }
      }
      __syncthreads();
    } else {
      float previous_score = 0.0f;
      int previous_index = -1;
      emitted = 0;
      for (int rank = 0; rank < selected; ++rank) {
        float best_value = 0.0f;
        int best_index = -1;
        ScanForNext(scores_row, block_count, previous_score, previous_index, &best_value, &best_index);
        shared_value[threadIdx.x] = best_value;
        shared_index[threadIdx.x] = best_index;
        __syncthreads();
        BlockArgMax(shared_value, shared_index);
        previous_index = shared_index[0];
        previous_score = shared_value[0];
        __syncthreads();
        if (previous_index < 0) {
          break;
        }
        for (int t = threadIdx.x; t < params.compress_ratio; t += blockDim.x) {
          out_row[rank * params.compress_ratio + t] =
              prefix_visible ? previous_index * params.compress_ratio + t
                             : index_row[previous_index * params.compress_ratio + t];
        }
        emitted = rank + 1;
        __syncthreads();
      }
    }

    const int tail_start = block_count * params.compress_ratio;
    for (int t = threadIdx.x; t < visible - tail_start; t += blockDim.x) {
      out_row[emitted * params.compress_ratio + t] =
          prefix_visible ? tail_start + t : index_row[tail_start + t];
    }
    __syncthreads();
  }
}

// One block per query row. Four radix passes find the TopK score threshold, then the winners are
// gathered with stable ties and only that bounded set is sorted by score and index.
__global__ void QsaPartialTopKKernel(const float* block_scores, const int32_t* visible_count,
                                     int32_t* topk_indices, SparseAttentionIndexerParams params) {
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

  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int encoded_visible = visible_count[row];
    const int visible = encoded_visible < 0 ? -encoded_visible - 1 : encoded_visible;
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    if (selected == 0) {
      continue;
    }
    const float* scores = block_scores + row * params.max_block_count;

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
#pragma unroll
    for (int item = 0; item < kBoundedTopKItemsPerThread; ++item) {
      const int rank = threadIdx.x * kBoundedTopKItemsPerThread + item;
      if (rank < selected) {
        topk_indices[row * params.block_topk + rank] = topk::UnpackStableSortIndex(keys[item]);
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------------------------
// policy_mode = "csa"
// ---------------------------------------------------------------------------------------------

// Reads channel `channel` of token `position` of the virtual sequence [past buffer | new tokens].
template <typename T>
__device__ __forceinline__ float ExtendedValue(const T* past_buffer, const T* current, int batch,
                                               int position, int channel,
                                               const SparseAttentionIndexerParams& params) {
  const int width = 2 * params.head_size;
  if (position < params.past_buffer_length) {
    return to_float<T>(
        past_buffer[(static_cast<int64_t>(batch) * params.past_buffer_length + position) * width + channel]);
  }
  return to_float<T>(
      current[(static_cast<int64_t>(batch) * params.sequence_length + (position - params.past_buffer_length)) * width +
              channel]);
}

// One block per (batch, new window). Softmax-pools the 2 * compress_ratio window slots, normalizes
// and rotates the result into present_compressed_key.
template <typename T>
__global__ void CsaCompressKernel(const T* key, const T* gate, const T* past_kv_buffer,
                                  const T* past_gate_buffer, const T* position_bias,
                                  const T* key_norm_weight, const T* cos_cache, const T* sin_cache,
                                  T* present_compressed_key, SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* reduction = shared + params.head_size;

  const int width = 2 * params.head_size;
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.new_window_count;
  for (int64_t work = blockIdx.x; work < total; work += gridDim.x) {
    const int window = static_cast<int>(work % params.new_window_count);
    const int batch = static_cast<int>(work / params.new_window_count);
    const bool has_previous = window >= 1 || params.overlap_length >= params.compress_ratio;
    const int previous_base = params.overlap_length + (window - 1) * params.compress_ratio;
    const int current_base = params.overlap_length + window * params.compress_ratio;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      float max_gate = NegativeInfinity();
      if (has_previous) {
        for (int slot = 0; slot < params.compress_ratio; ++slot) {
          const float value =
              ExtendedValue<T>(past_gate_buffer, gate, batch, previous_base + slot, d, params) +
              to_float<T>(position_bias[static_cast<int64_t>(slot) * width + d]);
          max_gate = fmaxf(max_gate, value);
        }
      }
      for (int slot = 0; slot < params.compress_ratio; ++slot) {
        const float value =
            ExtendedValue<T>(past_gate_buffer, gate, batch, current_base + slot, params.head_size + d, params) +
            to_float<T>(position_bias[static_cast<int64_t>(slot) * width + params.head_size + d]);
        max_gate = fmaxf(max_gate, value);
      }

      float denominator = 0.0f;
      float accumulator = 0.0f;
      if (has_previous) {
        for (int slot = 0; slot < params.compress_ratio; ++slot) {
          const float logit =
              ExtendedValue<T>(past_gate_buffer, gate, batch, previous_base + slot, d, params) +
              to_float<T>(position_bias[static_cast<int64_t>(slot) * width + d]);
          const float weight = __expf(logit - max_gate);
          denominator += weight;
          accumulator += weight * ExtendedValue<T>(past_kv_buffer, key, batch, previous_base + slot, d, params);
        }
      }
      for (int slot = 0; slot < params.compress_ratio; ++slot) {
        const float logit =
            ExtendedValue<T>(past_gate_buffer, gate, batch, current_base + slot, params.head_size + d, params) +
            to_float<T>(position_bias[static_cast<int64_t>(slot) * width + params.head_size + d]);
        const float weight = __expf(logit - max_gate);
        denominator += weight;
        accumulator +=
            weight * ExtendedValue<T>(past_kv_buffer, key, batch, current_base + slot, params.head_size + d, params);
      }
      pooled[d] = denominator > 0.0f ? accumulator / denominator : 0.0f;
    }
    __syncthreads();

    float sum_squares = 0.0f;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      sum_squares += pooled[d] * pooled[d];
    }
    sum_squares = BlockSum(sum_squares, reduction);
    const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(params.head_size) + params.epsilon);
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      pooled[d] = pooled[d] * inverse_rms * to_float<T>(key_norm_weight[d]);
    }
    __syncthreads();

    const int64_t entry = static_cast<int64_t>(params.past_compressed_length) + window;
    const int position = ClampPosition(entry * params.compress_ratio, params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(batch) * params.rotary_cache_batch_stride + position) * params.rotary_width;
    const int64_t out_base =
        (static_cast<int64_t>(batch) * params.compressed_cache_capacity + entry) * params.head_size;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      present_compressed_key[out_base + d] = from_float<T>(TrailingRope<T>(
          pooled, params.head_size, params.rotary_width, cos_cache + cache_offset, sin_cache + cache_offset, d));
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void CsaCopyPastCompressedKernel(const T* past_compressed_key, T* present_compressed_key,
                                            SparseAttentionIndexerParams params) {
  const int64_t total =
      static_cast<int64_t>(params.batch_size) * params.past_compressed_capacity * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t entry_row = index / params.head_size;
    const int entry = static_cast<int>(entry_row % params.past_compressed_capacity);
    const int batch = static_cast<int>(entry_row / params.past_compressed_capacity);
    const int d = static_cast<int>(index % params.head_size);
    present_compressed_key[(static_cast<int64_t>(batch) * params.compressed_cache_capacity + entry) *
                               params.head_size +
                           d] =
        past_compressed_key[(static_cast<int64_t>(batch) * params.past_compressed_capacity + entry) *
                                params.head_size +
                            d];
  }
}

template <typename T>
__global__ void CsaCopyBufferKernel(const T* key, const T* gate, const T* past_kv_buffer,
                                    const T* past_gate_buffer, T* present_kv_buffer, T* present_gate_buffer,
                                    SparseAttentionIndexerParams params) {
  const int width = 2 * params.head_size;
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.present_buffer_length * width;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int channel = static_cast<int>(index % width);
    const int64_t token_row = index / width;
    const int token = static_cast<int>(token_row % params.present_buffer_length);
    const int batch = static_cast<int>(token_row / params.present_buffer_length);
    const int source = params.present_buffer_start + token;
    present_kv_buffer[index] = from_float<T>(ExtendedValue<T>(past_kv_buffer, key, batch, source, channel, params));
    present_gate_buffer[index] =
        from_float<T>(ExtendedValue<T>(past_gate_buffer, gate, batch, source, channel, params));
  }
}

// One thread per (query row, compressed entry). Also applies the causal mask so that the selection
// kernel only has to read scores.
template <typename T>
__global__ void CsaScoreKernel(const float* query_rotated, const T* present_compressed_key,
                               const T* head_weights, const int64_t* position_ids, float* scores,
                               SparseAttentionIndexerParams params) {
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.sequence_length *
                        params.present_compressed_length;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int entry = static_cast<int>(index % params.present_compressed_length);
    const int64_t row = index / params.present_compressed_length;
    const int batch = static_cast<int>(row / params.sequence_length);

    const int64_t threshold = CausalThreshold(position_ids[row], params.compress_ratio);
    if (static_cast<int64_t>(entry) >= threshold) {
      scores[index] = NegativeInfinity();
      continue;
    }

    const int64_t key_base =
        (static_cast<int64_t>(batch) * params.compressed_cache_capacity + entry) * params.head_size;
    float total_score = 0.0f;
    for (int head = 0; head < params.num_heads; ++head) {
      const float* query_head = query_rotated + (row * params.num_heads + head) * params.head_size;
      float dot = 0.0f;
      for (int d = 0; d < params.head_size; ++d) {
        dot += query_head[d] * to_float<T>(present_compressed_key[key_base + d]);
      }
      total_score += fmaxf(dot, 0.0f) * to_float<T>(head_weights[row * params.num_heads + head]);
    }
    scores[index] = total_score * params.scale * params.head_weight_scale;
  }
}

__global__ void CsaSelectKernel(const float* scores, const int64_t* position_ids, int32_t* selected_indices,
                                SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* shared_value = shared;
  int* shared_index = reinterpret_cast<int*>(shared + blockDim.x);

  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t* out_row = selected_indices + row * params.capacity;
    for (int position = threadIdx.x; position < params.capacity; position += blockDim.x) {
      out_row[position] = -1;
    }
    __syncthreads();

    const int count = params.present_compressed_length;
    const int selected = min(params.index_topk, count);
    const int64_t threshold = CausalThreshold(position_ids[row], params.compress_ratio);
    const float* scores_row = scores + row * count;

    float previous_score = 0.0f;
    int previous_index = -1;
    for (int rank = 0; rank < selected; ++rank) {
      float best_value = 0.0f;
      int best_index = -1;
      ScanForNext(scores_row, count, previous_score, previous_index, &best_value, &best_index);
      shared_value[threadIdx.x] = best_value;
      shared_index[threadIdx.x] = best_index;
      __syncthreads();
      BlockArgMax(shared_value, shared_index);
      previous_index = shared_index[0];
      previous_score = shared_value[0];
      __syncthreads();
      if (previous_index < 0) {
        break;
      }
      if (threadIdx.x == 0 && static_cast<int64_t>(previous_index) < threshold) {
        out_row[rank] = previous_index;
      }
      __syncthreads();
    }
  }
}

}  // namespace

size_t GetQsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  const size_t query_count = rows * params.num_heads * params.head_size;
  return UseHierarchicalQsaTopK(params, false)
             ? query_count
             : query_count + rows * std::max(params.max_block_count, 1);
}

size_t GetQsaWorkspaceIntCount(const SparseAttentionIndexerParams& params, bool has_mask) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  if (UseHierarchicalQsaTopK(params, has_mask)) {
    const size_t key_count = rows * GetHierarchicalTileCount(params) * kHierarchicalTileBlocks;
    return rows + 1 + 4 * key_count;
  }
  if (UseDistributedQsaTopK(params, has_mask)) {
    return rows + rows * std::max(params.max_block_count, 1) + rows * 256 + rows * 4 + 1;
  }
  return (has_mask ? rows * std::max(params.total_sequence_length, 1) : 0) + rows +
         rows * std::min(params.block_topk, kBoundedTopKMax);
}

size_t GetCsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  return rows * params.num_heads * params.head_size + rows * std::max(params.present_compressed_length, 1);
}

template <typename T>
Status LaunchQsaSparseAttentionIndexer(cudaStream_t stream, const SparseAttentionIndexerParams& params,
                                       const T* query, const T* key, const T* query_norm_weight,
                                       const T* key_norm_weight,
                                       const T* cos_cache, const T* sin_cache, const int64_t* mask,
                                       const T* past_key, int32_t* selected_indices, T* present_key,
                                       float* float_workspace, int32_t* int_workspace) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;

  // The present state is produced even when there is no query row to score, so that a zero-length step still
  // forwards the incoming cache unchanged.
  const int64_t past_key_elements =
      static_cast<int64_t>(params.batch_size) * params.past_key_capacity * params.head_size;
  if (past_key_elements > 0 && past_key != present_key) {
    CopyQsaPastKeyKernel<T><<<GridForElements(past_key_elements), kThreads, 0, stream>>>(
        past_key, present_key, params);
  }
  if (params.use_block_representatives) {
    const int first_block = params.past_sequence_length / kQwenCompressRatio;
    const int completed_blocks = params.total_sequence_length / kQwenCompressRatio;
    const int64_t new_blocks = static_cast<int64_t>(params.batch_size) * (completed_blocks - first_block);
    if (new_blocks > 0) {
      QsaBuildBlockRepresentativesKernel<T><<<GridForElements(new_blocks), kThreads, 0, stream>>>(
          past_key, key, present_key, key_norm_weight, cos_cache, sin_cache, params);
    }
    const int64_t new_key_elements =
        static_cast<int64_t>(params.batch_size) * params.sequence_length * params.head_size;
    if (new_key_elements > 0) {
      AppendQsaKeyTailKernel<T><<<GridForElements(new_key_elements), kThreads, 0, stream>>>(
          key, present_key, params);
    }
  } else {
    const int64_t new_key_elements =
        static_cast<int64_t>(params.batch_size) * params.sequence_length * params.head_size;
    if (new_key_elements > 0) {
      AppendQsaKeyKernel<T><<<GridForElements(new_key_elements), kThreads, 0, stream>>>(key, present_key, params);
    }
  }

  if (rows == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* block_scores = query_rotated + rows * params.num_heads * params.head_size;
  int32_t* visible_indices = mask == nullptr ? nullptr : int_workspace;
  int32_t* visible_count = mask == nullptr ? int_workspace : int_workspace + rows * params.total_sequence_length;
  int32_t* topk_indices = visible_count + rows;
  const bool use_hierarchical_topk = UseHierarchicalQsaTopK(params, mask != nullptr);
  const bool use_distributed_topk = UseDistributedQsaTopK(params, mask != nullptr);
  uint64_t* hierarchical_keys = use_hierarchical_topk
                                    ? reinterpret_cast<uint64_t*>((reinterpret_cast<uintptr_t>(topk_indices) +
                                                                   alignof(uint64_t) - 1) &
                                                                  ~(alignof(uint64_t) - 1))
                                    : nullptr;
                          const size_t hierarchical_key_count = use_hierarchical_topk
                                        ? static_cast<size_t>(rows) * GetHierarchicalTileCount(params) *
                                          kHierarchicalTileBlocks
                                        : 0;
                          uint64_t* hierarchical_scratch = use_hierarchical_topk ? hierarchical_keys + hierarchical_key_count : nullptr;
  int32_t* candidate_indices = use_distributed_topk && !use_hierarchical_topk ? topk_indices : nullptr;
  uint32_t* radix_histogram = use_distributed_topk && !use_hierarchical_topk
                                  ? reinterpret_cast<uint32_t*>(candidate_indices + rows * params.max_block_count)
                                  : nullptr;
  uint64_t* radix_prefixes = use_distributed_topk && !use_hierarchical_topk
                                 ? reinterpret_cast<uint64_t*>((reinterpret_cast<uintptr_t>(radix_histogram + rows * 256) +
                                                                alignof(uint64_t) - 1) &
                                                               ~(alignof(uint64_t) - 1))
                                 : nullptr;
  int32_t* radix_remaining = radix_prefixes ? reinterpret_cast<int32_t*>(radix_prefixes + rows) : nullptr;
  int32_t* candidate_counts = radix_remaining ? radix_remaining + rows : nullptr;

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  if (mask == nullptr) {
    InitializePrefixVisibleCountKernel<<<GridForElements(rows), kThreads, 0, stream>>>(visible_count, params);
  } else {
    AnalyzeVisibleKernel<<<row_blocks, kThreads, 0, stream>>>(mask, visible_count, params);
    CompactNonPrefixVisibleKernel<<<row_blocks, kThreads, 0, stream>>>(
        mask, visible_indices, visible_count, params);
  }

  const int64_t query_rows = rows * params.num_heads;
  const int rotate_blocks = static_cast<int>(std::min<int64_t>(query_rows, kMaxGridDimX));
  RotateQueryKernel<T, true><<<rotate_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, nullptr, query_rotated, params);

  if (params.max_block_count > 0) {
    if (use_distributed_topk && !use_hierarchical_topk) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(radix_histogram, 0, rows * 256 * sizeof(uint32_t), stream));
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(candidate_counts, 0, rows * sizeof(int32_t), stream));
    }
    if (use_hierarchical_topk) {
      const int tile_count = GetHierarchicalTileCount(params);
      const int64_t tile_work = rows * tile_count;
      const int tile_blocks = static_cast<int>(std::min<int64_t>(tile_work, kMaxGridDimX));
        QsaScoreTileTopKKernel<T><<<tile_blocks, kHierarchicalScoreThreads, 0, stream>>>(
          query_rotated, present_key, visible_count, hierarchical_keys, tile_count, params);
    } else {
      const int64_t block_work = rows * params.max_block_count;
      const int score_blocks = static_cast<int>(std::min<int64_t>(block_work, kMaxGridDimX));
      if (params.head_size == kQwenHeadSize && params.compress_ratio == kQwenCompressRatio &&
        params.num_heads == kQwenNumHeads) {
        if (params.use_block_representatives) {
          QsaBlockScoreQwenCachedKernel<T><<<score_blocks, kThreads, 0, stream>>>(
              query_rotated, present_key, visible_count, block_scores, radix_histogram, params);
        } else {
          QsaBlockScoreQwenKernel<T><<<score_blocks, kThreads, 0, stream>>>(
              query_rotated, present_key, key_norm_weight, cos_cache, sin_cache, visible_indices, visible_count,
              block_scores, radix_histogram, params);
        }
      } else {
        QsaBlockScoreKernel<T><<<score_blocks, kThreads, 2 * value_bytes + kThreads * sizeof(float), stream>>>(
            query_rotated, present_key, key_norm_weight, cos_cache, sin_cache, visible_indices, visible_count,
            block_scores, params);
      }
    }
  }

  const bool use_bounded_topk =
      params.block_topk > kSaiFastTopKMax && params.block_topk <= kBoundedTopKMax;
  if (use_hierarchical_topk) {
    const int block_count = GetHierarchicalTileCount(params) * kHierarchicalTileBlocks;
    uint64_t* merge_input = hierarchical_keys;
    uint64_t* merge_output = hierarchical_scratch;
    for (int list_width = kHierarchicalTileBlocks; list_width < block_count; list_width *= 2) {
      const int pairs_per_row = (block_count + 2 * list_width - 1) / (2 * list_width);
      const int64_t merge_work = rows * pairs_per_row;
      QsaMergeTileTopKKernel<<<static_cast<int>(std::min<int64_t>(merge_work, kMaxGridDimX)),
                               kHierarchicalMergeThreads, 0, stream>>>(
          merge_input, merge_output, block_count, list_width, params);
      std::swap(merge_input, merge_output);
    }
    QsaEmitHierarchicalTopKKernel<<<row_blocks, kThreads, 0, stream>>>(
      merge_input, visible_count, block_count, selected_indices, params);
  } else if (use_distributed_topk) {
    QsaChooseRadixBucketKernel<<<row_blocks, kThreads, 0, stream>>>(
      radix_histogram, radix_prefixes, radix_remaining, 56, params);
    QsaCompactHighBucketKernel<<<GridForElements(rows * params.max_block_count), kThreads, 0, stream>>>(
      block_scores, visible_count, radix_prefixes, candidate_counts, candidate_indices, params);
    for (int shift = 48; shift >= 0; shift -= 8) {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(radix_histogram, 0, rows * 256 * sizeof(uint32_t), stream));
      QsaCandidateHistogramKernel<<<GridForElements(rows * params.max_block_count), kThreads, 0, stream>>>(
        block_scores, candidate_counts, candidate_indices, radix_prefixes, radix_histogram, shift, params);
      QsaChooseRadixBucketKernel<<<row_blocks, kThreads, 0, stream>>>(
        radix_histogram, radix_prefixes, radix_remaining, shift, params);
    }
    QsaDistributedSelectKernel<<<row_blocks, kThreads, 0, stream>>>(
      block_scores, visible_count, candidate_counts, candidate_indices, radix_prefixes,
      selected_indices, params);
    } else if (use_bounded_topk) {
    QsaPartialTopKKernel<<<row_blocks, kThreads, 0, stream>>>(
        block_scores, visible_count, topk_indices, params);
  }

  const int topk_shared_entries = !use_bounded_topk && params.block_topk <= kSaiFastTopKMax
                                      ? kThreads * params.block_topk
                                      : kThreads;
  if (!use_distributed_topk) {
    QsaSelectKernel<<<row_blocks, kThreads,
                      static_cast<size_t>(topk_shared_entries) * (sizeof(float) + sizeof(int)), stream>>>(
        block_scores, visible_indices, visible_count, use_bounded_topk ? topk_indices : nullptr,
        selected_indices, params);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchCsaSparseAttentionIndexer(cudaStream_t stream, const SparseAttentionIndexerParams& params,
                                       const T* query, const T* key, const T* query_norm_weight,
                                       const T* key_norm_weight,
                                       const T* cos_cache, const T* sin_cache, const T* gate,
                                       const T* position_bias, const T* head_weights,
                                       const int64_t* position_ids, const T* past_compressed_key,
                                       const T* past_kv_buffer, const T* past_gate_buffer,
                                       int32_t* selected_indices, T* present_compressed_key,
                                       T* present_kv_buffer, T* present_gate_buffer, float* float_workspace) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);

  // The present state is produced even when there is no query row to score, so that a zero-length step still
  // forwards the incoming cache unchanged.
  const int64_t past_compressed_elements =
      static_cast<int64_t>(params.batch_size) * params.past_compressed_capacity * params.head_size;
  if (past_compressed_elements > 0 && past_compressed_key != present_compressed_key) {
    CsaCopyPastCompressedKernel<T><<<GridForElements(past_compressed_elements), kThreads, 0, stream>>>(
        past_compressed_key, present_compressed_key, params);
  }

  if (params.batch_size > 0 && params.new_window_count > 0) {
    const int compress_blocks = static_cast<int>(
        std::min<int64_t>(static_cast<int64_t>(params.batch_size) * params.new_window_count, kMaxGridDimX));
    CsaCompressKernel<T><<<compress_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
        key, gate, past_kv_buffer, past_gate_buffer, position_bias, key_norm_weight, cos_cache, sin_cache,
        present_compressed_key, params);
  }

  const int64_t present_buffer_elements =
      static_cast<int64_t>(params.batch_size) * params.present_buffer_length * 2 * params.head_size;
  if (present_buffer_elements > 0) {
    CsaCopyBufferKernel<T><<<GridForElements(present_buffer_elements), kThreads, 0, stream>>>(
        key, gate, past_kv_buffer, past_gate_buffer, present_kv_buffer, present_gate_buffer, params);
  }

  if (rows == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* scores = float_workspace + rows * params.num_heads * params.head_size;

  const int rotate_blocks = static_cast<int>(std::min<int64_t>(rows * params.num_heads, kMaxGridDimX));
  RotateQueryKernel<T, false><<<rotate_blocks, kThreads, value_bytes + kThreads * sizeof(float), stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, position_ids, query_rotated, params);

  if (params.present_compressed_length > 0) {
    CsaScoreKernel<T><<<GridForElements(rows * params.present_compressed_length), kThreads, 0, stream>>>(
        query_rotated, present_compressed_key, head_weights, position_ids, scores, params);
  }

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  CsaSelectKernel<<<row_blocks, kThreads, kThreads * (sizeof(float) + sizeof(int)), stream>>>(
      scores, position_ids, selected_indices, params);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_SPARSE_ATTENTION_INDEXER(T)                                                                      \
  template Status LaunchQsaSparseAttentionIndexer<T>(cudaStream_t, const SparseAttentionIndexerParams&,              \
                                                     const T*, const T*, const T*, const T*, const T*, const T*,     \
                                                     const int64_t*, const T*, int32_t*, T*, float*, int32_t*);      \
  template Status LaunchCsaSparseAttentionIndexer<T>(                                                                \
      cudaStream_t, const SparseAttentionIndexerParams&, const T*, const T*, const T*, const T*, const T*, const T*, \
      const T*, const T*, const T*, const int64_t*, const T*, const T*, const T*, int32_t*, T*, T*, T*, float*);

INSTANTIATE_SPARSE_ATTENTION_INDEXER(float)
INSTANTIATE_SPARSE_ATTENTION_INDEXER(half)
INSTANTIATE_SPARSE_ATTENTION_INDEXER(__nv_bfloat16)

#undef INSTANTIATE_SPARSE_ATTENTION_INDEXER

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
