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
#include "core/providers/cuda/cu_inc/cub.cuh"

#include <algorithm>
#include <climits>
#include <limits>

#include "core/common/safeint.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace topk {
constexpr uint64_t kPaddingSortKey = 0;

__device__ __forceinline__ uint64_t PackStableSortKey(float score, int index) {
  const uint32_t score_bits = score == 0.0f ? 0u : __float_as_uint(score);
  const uint32_t sortable_score =
      (score_bits & 0x80000000u) ? (~score_bits) : (score_bits | 0x80000000u);
  const uint32_t inverted_index = UINT_MAX - static_cast<uint32_t>(index);
  return (static_cast<uint64_t>(sortable_score) << 32) | inverted_index;
}

__device__ __forceinline__ int UnpackStableSortIndex(uint64_t key) {
  const uint32_t inverted_index = static_cast<uint32_t>(key & 0xffffffffu);
  return static_cast<int>(UINT_MAX - inverted_index);
}
}

namespace {

// The block reductions below halve the active thread count, so this must stay a power of two.
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int kBoundedTopKMax = 512;
constexpr int kBoundedTopKItemsPerThread = kBoundedTopKMax / kThreads;
constexpr int64_t kMaxGridDimX = 2147483647;

template <typename Kernel>
Status ConfigureDynamicSharedMemory(Kernel kernel, size_t bytes, const char* kernel_name) {
  int device = 0;
  int max_bytes = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
  CUDA_RETURN_IF_ERROR(
      cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  ORT_RETURN_IF(bytes > static_cast<size_t>(max_bytes),
                "SparseAttentionIndexer: ", kernel_name, " requires ", bytes,
                " bytes of dynamic shared memory, but the device supports at most ", max_bytes);
  return CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes));
}

__device__ __forceinline__ float NegativeInfinity() { return -CUDART_INF_F; }

int GridForElements(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::clamp<int64_t>(blocks, 1, 65535));
}

// ---------------------------------------------------------------------------------------------
// Shared device helpers
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float total = shared[0];
  __syncthreads();
  return total;
}

// Reduces (value, index) pairs to the largest value, breaking ties towards the smaller index.
// A negative index marks an empty slot. shared_value/shared_index must already be filled and synced.
__device__ __forceinline__ void BlockArgMax(float* shared_value, int* shared_index) {
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other_index = shared_index[threadIdx.x + stride];
      if (other_index >= 0) {
        const int this_index = shared_index[threadIdx.x];
        const float other_value = shared_value[threadIdx.x + stride];
        const float this_value = shared_value[threadIdx.x];
        if (this_index < 0 || other_value > this_value ||
            (other_value == this_value && other_index < this_index)) {
          shared_value[threadIdx.x] = other_value;
          shared_index[threadIdx.x] = other_index;
        }
      }
    }
    __syncthreads();
  }
}

// Per-thread scan for the best entry that comes strictly after (previous_score, previous_index) in
// the total order "score descending, then index ascending". Entries already emitted are therefore
// skipped without needing a visited bitmap.
__device__ __forceinline__ void ScanForNext(const float* scores, int count, float previous_score,
                                            int previous_index, float* best_value, int* best_index) {
  *best_index = -1;
  *best_value = 0.0f;
  for (int64_t candidate = threadIdx.x; candidate < count; candidate += blockDim.x) {
    const float value = scores[candidate];
    if (previous_index >= 0 &&
        !(value < previous_score || (value == previous_score && candidate > previous_index))) {
      continue;
    }
    if (*best_index < 0 || value > *best_value ||
        (value == *best_value && candidate < *best_index)) {
      *best_value = value;
      *best_index = static_cast<int>(candidate);
    }
  }
}

// Split-half rotary over the leading `rotary_width` channels (the convention used by the qsa
// reference). Channels beyond `rotary_width` pass through unchanged.
template <typename T>
__device__ __forceinline__ float LeadingRope(const float* value, int rotary_width, const T* cos_row,
                                             const T* sin_row, int d) {
  if (d >= rotary_width) {
    return value[d];
  }
  const int half = rotary_width / 2;
  const float paired = (d < half) ? -value[d + half] : value[d - half];
  return value[d] * to_float<T>(cos_row[d]) + paired * to_float<T>(sin_row[d]);
}

// Interleaved rotary over the trailing 2 * rotary_width channels (the convention used by the csa
// reference). Each cos/sin entry covers one channel pair, matching repeat_interleave(2).
template <typename T>
__device__ __forceinline__ float TrailingRope(const float* value, int head_size, int rotary_width,
                                              const T* cos_row, const T* sin_row, int d) {
  const int base = head_size - 2 * rotary_width;
  if (d < base) {
    return value[d];
  }
  const int offset = d - base;
  const float paired = ((offset & 1) == 0) ? -value[d + 1] : value[d - 1];
  return value[d] * to_float<T>(cos_row[offset >> 1]) + paired * to_float<T>(sin_row[offset >> 1]);
}

// Highest compressed entry a query at `position` may attend to, matching (position + 1) // ratio.
__device__ __forceinline__ int64_t CausalThreshold(int64_t position, int compress_ratio) {
  if (position < 0) {
    return 0;
  }
  const int64_t quotient = position / compress_ratio;
  const bool round_up = position % compress_ratio == compress_ratio - 1;
  return round_up && quotient < std::numeric_limits<int64_t>::max() ? quotient + 1 : quotient;
}

__device__ __forceinline__ int ClampPosition(int64_t position, int max_rotary_length) {
  if (position < 0) {
    return 0;
  }
  const int64_t limit = max_rotary_length - 1;
  return static_cast<int>(position < limit ? position : limit);
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
        (static_cast<int64_t>(batch) * params.max_rotary_length + position) * params.rotary_width;
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

// One block per query row; compacts the visible key positions of that row into visible_indices.
__global__ void CompactVisibleKernel(const int64_t* mask, int32_t* visible_indices, int32_t* visible_count,
                                     SparseAttentionIndexerParams params) {
  extern __shared__ int32_t shared_scan[];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int batch = static_cast<int>(row / params.sequence_length);
    const int query = static_cast<int>(row % params.sequence_length);
    const int64_t* mask_row = mask + static_cast<int64_t>(batch) * params.total_sequence_length;
    int32_t* out_row = visible_indices + row * params.total_sequence_length;
    int32_t offset = 0;
    for (int64_t base = 0; base < params.total_sequence_length; base += blockDim.x) {
      const int64_t position = base + threadIdx.x;
      const int32_t flag =
          (position < params.total_sequence_length && position <= params.past_sequence_length + query &&
           mask_row[position] != 0)
              ? 1
              : 0;
      shared_scan[threadIdx.x] = flag;
      __syncthreads();
      for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        const int32_t addend = (threadIdx.x >= static_cast<unsigned>(stride)) ? shared_scan[threadIdx.x - stride] : 0;
        __syncthreads();
        shared_scan[threadIdx.x] += addend;
        __syncthreads();
      }
      if (flag != 0) {
        out_row[offset + shared_scan[threadIdx.x] - 1] = static_cast<int32_t>(position);
      }
      const int32_t tile_total = shared_scan[blockDim.x - 1];
      __syncthreads();
      offset += tile_total;
    }
    if (threadIdx.x == 0) {
      visible_count[row] = offset;
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
    const int block_count = visible_count[row] / params.compress_ratio;

    if (block_index >= block_count) {
      if (threadIdx.x == 0) {
        block_scores[row * params.max_block_count + block_index] = NegativeInfinity();
      }
      continue;
    }

    const int32_t* index_row = visible_indices + row * params.total_sequence_length;
    const int32_t* group = index_row + static_cast<int64_t>(block_index) * params.compress_ratio;
    const int64_t key_base = static_cast<int64_t>(batch) * params.key_cache_capacity * params.head_size;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      float sum = 0.0f;
      for (int t = 0; t < params.compress_ratio; ++t) {
        sum += to_float<T>(present_key[key_base + static_cast<int64_t>(group[t]) * params.head_size + d]);
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

    const int position = ClampPosition(group[0], params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(batch) * params.max_rotary_length + position) * params.rotary_width;
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
      block_scores[row * params.max_block_count + block_index] = score * params.scale;
    }
    __syncthreads();
  }
}

// Prefill scoring, step 1: every complete block's pooled / normalized / rotated key is the same for all query rows
// that can see it (the visible-token list of a row is a prefix of the list of the batch's last row), so it is computed
// once per (batch, block) instead of once per (row, block). One CTA per block; the arithmetic is the one of
// QsaBlockScoreKernel.
template <typename T>
__global__ void QsaPoolBlockKeysKernel(const T* present_key, const T* key_norm_weight, const T* cos_cache,
                                       const T* sin_cache, const int32_t* visible_indices,
                                       const int32_t* visible_count, float* pooled_keys,
                                       SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* rotated = shared + params.head_size;
  float* reduction = shared + 2 * params.head_size;

  const int batch = static_cast<int>(blockIdx.y);
  const int64_t last_row = static_cast<int64_t>(batch) * params.sequence_length + params.sequence_length - 1;
  const int block_count = visible_count[last_row] / params.compress_ratio;
  const int32_t* index_row = visible_indices + last_row * params.total_sequence_length;
  const int64_t key_base = static_cast<int64_t>(batch) * params.key_cache_capacity * params.head_size;

  for (int block_index = blockIdx.x; block_index < block_count; block_index += gridDim.x) {
    const int32_t* group = index_row + static_cast<int64_t>(block_index) * params.compress_ratio;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      float sum = 0.0f;
      for (int t = 0; t < params.compress_ratio; ++t) {
        sum += to_float<T>(present_key[key_base + static_cast<int64_t>(group[t]) * params.head_size + d]);
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
    const int position = ClampPosition(group[0], params.max_rotary_length);
    const int64_t cache_offset =
        (static_cast<int64_t>(batch) * params.max_rotary_length + position) * params.rotary_width;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      pooled_keys[(static_cast<int64_t>(batch) * params.max_block_count + block_index) * params.head_size + d] =
          LeadingRope<T>(pooled, params.rotary_width, cos_cache + cache_offset, sin_cache + cache_offset, d);
    }
    __syncthreads();
  }
}

constexpr int kScoreTileRows = 64;
constexpr int kScoreTileBlocks = 64;
constexpr int kScoreChunk = 32;

// Prefill scoring, step 2: block_scores[row, j] = scale * sum_h relu(q_rot[row, h] . pooled[j]) for j below the row's
// visible block count, as a register-tiled FP32 GEMM (64 rows x 64 blocks per CTA, 4 x 4 per thread).
__global__ void __launch_bounds__(256) QsaTiledBlockScoreKernel(const float* query_rotated, const float* pooled_keys,
                                                                const int32_t* visible_count, float* block_scores,
                                                                SparseAttentionIndexerParams params) {
  __shared__ float q_tile[kScoreTileRows][kScoreChunk + 1];
  __shared__ float k_tile[kScoreTileBlocks][kScoreChunk + 1];

  const int batch = static_cast<int>(blockIdx.z);
  const int row0 = static_cast<int>(blockIdx.y) * kScoreTileRows;
  const int block0 = static_cast<int>(blockIdx.x) * kScoreTileBlocks;
  const int last_row_in_tile = min(row0 + kScoreTileRows, params.sequence_length) - 1;
  const int64_t row_base = static_cast<int64_t>(batch) * params.sequence_length;
  // Visible counts are non-decreasing with the row, so the last row of the tile bounds the tile's block range.
  if (block0 >= visible_count[row_base + last_row_in_tile] / params.compress_ratio) {
    return;
  }

  const int tx = threadIdx.x & 15;
  const int ty = threadIdx.x >> 4;
  float total[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) total[i][j] = 0.0f;
  }

  for (int head = 0; head < params.num_heads; ++head) {
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    }
    for (int d0 = 0; d0 < params.head_size; d0 += kScoreChunk) {
      for (int idx = threadIdx.x; idx < kScoreTileRows * kScoreChunk; idx += 256) {
        const int r = idx / kScoreChunk;
        const int c = idx % kScoreChunk;
        const int row = row0 + r;
        q_tile[r][c] = row < params.sequence_length
                           ? query_rotated[((row_base + row) * params.num_heads + head) * params.head_size + d0 + c]
                           : 0.0f;
        const int block = block0 + r;
        k_tile[r][c] = block < params.max_block_count
                           ? pooled_keys[(static_cast<int64_t>(batch) * params.max_block_count + block) * params.head_size +
                                         d0 + c]
                           : 0.0f;
      }
      __syncthreads();
#pragma unroll 8
      for (int dd = 0; dd < kScoreChunk; ++dd) {
        float a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = q_tile[ty + 16 * i][dd];
#pragma unroll
        for (int j = 0; j < 4; ++j) b[j] = k_tile[tx + 16 * j][dd];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) total[i][j] += fmaxf(acc[i][j], 0.0f);
    }
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = row0 + ty + 16 * i;
    if (row >= params.sequence_length) continue;
    const int block_count = visible_count[row_base + row] / params.compress_ratio;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int block = block0 + tx + 16 * j;
      if (block < block_count) {
        block_scores[(row_base + row) * params.max_block_count + block] = total[i][j] * params.scale;
      }
    }
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
  int* shared_index = reinterpret_cast<int*>(shared + blockDim.x);

  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int32_t* out_row = selected_indices + row * params.capacity;
    for (int64_t position = threadIdx.x; position < params.capacity; position += blockDim.x) {
      out_row[position] = -1;
    }
    __syncthreads();

    const int visible = visible_count[row];
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    const float* scores_row = block_scores + row * params.max_block_count;
    const int32_t* index_row = visible_indices + row * params.total_sequence_length;

    if (topk_indices != nullptr && selected > 0 && topk_indices[row * params.block_topk] >= 0) {
      const int32_t* topk_row = topk_indices + row * params.block_topk;
      for (int rank = threadIdx.x; rank < selected; rank += blockDim.x) {
        const int selected_block = topk_row[rank];
        for (int token = 0; token < params.compress_ratio; ++token) {
          out_row[rank * params.compress_ratio + token] =
              index_row[selected_block * params.compress_ratio + token];
        }
      }
      const int tail_start = block_count * params.compress_ratio;
      for (int token = threadIdx.x; token < visible - tail_start; token += blockDim.x) {
        out_row[selected * params.compress_ratio + token] = index_row[tail_start + token];
      }
      __syncthreads();
      continue;
    }

    float previous_score = 0.0f;
    int previous_index = -1;
    int emitted = 0;
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
        out_row[rank * params.compress_ratio + t] = index_row[previous_index * params.compress_ratio + t];
      }
      emitted = rank + 1;
      __syncthreads();
    }

    const int tail_start = block_count * params.compress_ratio;
    for (int t = threadIdx.x; t < visible - tail_start; t += blockDim.x) {
      out_row[emitted * params.compress_ratio + t] = index_row[tail_start + t];
    }
    __syncthreads();
  }
}

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
    const int visible = visible_count[row];
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    if (selected == 0) {
      continue;
    }
    const float* scores = block_scores + row * params.max_block_count;
    bool has_nan = false;
    for (int index = threadIdx.x; index < block_count; index += blockDim.x) {
      has_nan = has_nan || isnan(scores[index]);
    }
    if (__syncthreads_or(has_nan)) {
      if (threadIdx.x == 0) {
        topk_indices[row * params.block_topk] = -1;
      }
      __syncthreads();
      continue;
    }

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
        for (int warp_index = 0; warp_index < kThreads / kWarpSize; ++warp_index) {
          const int warp_count = equal_warp_offsets[warp_index];
          equal_warp_offsets[warp_index] = tile_total;
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
                                  T* present_compressed_key, float* new_compressed_key,
                                  SparseAttentionIndexerParams params) {
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
      pooled[d] = denominator > 0.0f && isfinite(denominator) ? accumulator / denominator : 0.0f;
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
        (static_cast<int64_t>(batch) * params.max_rotary_length + position) * params.rotary_width;
    const int64_t out_base =
        (static_cast<int64_t>(batch) * params.compressed_cache_capacity + entry) * params.head_size;
    const int64_t workspace_base =
        (static_cast<int64_t>(batch) * params.new_window_count + window) * params.head_size;
    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      const float rotated = TrailingRope<T>(
          pooled, params.head_size, params.rotary_width, cos_cache + cache_offset, sin_cache + cache_offset, d);
      new_compressed_key[workspace_base + d] = rotated;
      present_compressed_key[out_base + d] = from_float<T>(rotated);
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
                               const float* new_compressed_key,
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
    const bool is_new_entry = entry >= params.past_compressed_length;
    const int64_t new_key_base =
        (static_cast<int64_t>(batch) * params.new_window_count + entry - params.past_compressed_length) *
        params.head_size;
    float total_score = 0.0f;
    for (int head = 0; head < params.num_heads; ++head) {
      const float* query_head = query_rotated + (row * params.num_heads + head) * params.head_size;
      float dot = 0.0f;
      for (int d = 0; d < params.head_size; ++d) {
        const float key_value =
            is_new_entry ? new_compressed_key[new_key_base + d] : to_float<T>(present_compressed_key[key_base + d]);
        dot += query_head[d] * key_value;
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
    for (int64_t position = threadIdx.x; position < params.capacity; position += blockDim.x) {
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
  const size_t pooled_keys = static_cast<size_t>(params.batch_size) * std::max(params.max_block_count, 1) * params.head_size;
  return rows * params.num_heads * params.head_size + rows * std::max(params.max_block_count, 1) + pooled_keys;
}

size_t GetQsaWorkspaceIntCount(const SparseAttentionIndexerParams& params) {
  const SafeInt<size_t> rows = SafeInt<size_t>(params.batch_size) * params.sequence_length;
  const int topk_width = params.block_topk > 0 && params.block_topk <= kBoundedTopKMax ? params.block_topk : 0;
  const SafeInt<size_t> count = rows * std::max(params.total_sequence_length, 1) + rows + rows * topk_width;
  return SafeInt<size_t>(count) * sizeof(int32_t) / sizeof(int32_t);
}

size_t GetCsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  return rows * params.num_heads * params.head_size +
         rows * std::max(params.present_compressed_length, 1) +
         static_cast<size_t>(params.batch_size) * params.new_window_count * params.head_size;
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
  const int64_t new_key_elements =
      static_cast<int64_t>(params.batch_size) * params.sequence_length * params.head_size;
  if (new_key_elements > 0) {
    AppendQsaKeyKernel<T><<<GridForElements(new_key_elements), kThreads, 0, stream>>>(key, present_key, params);
  }

  if (rows == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* block_scores = query_rotated + rows * params.num_heads * params.head_size;
  int32_t* visible_indices = int_workspace;
  int32_t* visible_count = int_workspace + rows * params.total_sequence_length;
  int32_t* topk_indices = params.block_topk > 0 && params.block_topk <= kBoundedTopKMax
                              ? visible_count + rows
                              : nullptr;

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);
  const size_t rotate_shared_bytes = value_bytes + kThreads * sizeof(float);
  const size_t score_shared_bytes = 2 * value_bytes + kThreads * sizeof(float);
  ORT_RETURN_IF_ERROR(
      ConfigureDynamicSharedMemory(RotateQueryKernel<T, true>, rotate_shared_bytes, "RotateQueryKernel"));

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  CompactVisibleKernel<<<row_blocks, kThreads, kThreads * sizeof(int32_t), stream>>>(
      mask, visible_indices, visible_count, params);

  const int64_t query_rows = rows * params.num_heads;
  const int rotate_blocks = static_cast<int>(std::min<int64_t>(query_rows, kMaxGridDimX));
  RotateQueryKernel<T, true><<<rotate_blocks, kThreads, rotate_shared_bytes, stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, nullptr, query_rotated, params);

  static const bool tiled_scoring =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_QSA_TILED_SCORING", 1) != 0;
  if (params.max_block_count > 0 && tiled_scoring && params.sequence_length >= 16 && params.head_size % kScoreChunk == 0) {
    float* pooled_keys = block_scores + rows * params.max_block_count;
    ORT_RETURN_IF_ERROR(
        ConfigureDynamicSharedMemory(QsaPoolBlockKeysKernel<T>, score_shared_bytes, "QsaPoolBlockKeysKernel"));
    const dim3 pool_grid(static_cast<unsigned>(std::min(params.max_block_count, 4096)),
                         static_cast<unsigned>(params.batch_size));
    QsaPoolBlockKeysKernel<T><<<pool_grid, kThreads, score_shared_bytes, stream>>>(
        present_key, key_norm_weight, cos_cache, sin_cache, visible_indices, visible_count, pooled_keys, params);
    const dim3 score_grid(static_cast<unsigned>((params.max_block_count + kScoreTileBlocks - 1) / kScoreTileBlocks),
                          static_cast<unsigned>((params.sequence_length + kScoreTileRows - 1) / kScoreTileRows),
                          static_cast<unsigned>(params.batch_size));
    QsaTiledBlockScoreKernel<<<score_grid, 256, 0, stream>>>(query_rotated, pooled_keys, visible_count,
                                                             block_scores, params);
  } else if (params.max_block_count > 0) {
    ORT_RETURN_IF_ERROR(
        ConfigureDynamicSharedMemory(QsaBlockScoreKernel<T>, score_shared_bytes, "QsaBlockScoreKernel"));
    const int64_t block_work = rows * params.max_block_count;
    const int score_blocks = static_cast<int>(std::min<int64_t>(block_work, kMaxGridDimX));
    QsaBlockScoreKernel<T><<<score_blocks, kThreads, score_shared_bytes, stream>>>(
        query_rotated, present_key, key_norm_weight, cos_cache, sin_cache, visible_indices, visible_count,
        block_scores, params);
  }

  if (topk_indices != nullptr) {
    QsaPartialTopKKernel<<<row_blocks, kThreads, 0, stream>>>(block_scores, visible_count, topk_indices, params);
  }
  QsaSelectKernel<<<row_blocks, kThreads, kThreads * (sizeof(float) + sizeof(int)), stream>>>(
      block_scores, visible_indices, visible_count, topk_indices, selected_indices, params);

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
  const size_t value_shared_bytes = value_bytes + kThreads * sizeof(float);

  float* query_rotated = float_workspace;
  float* scores = query_rotated + rows * params.num_heads * params.head_size;
  float* new_compressed_key = scores + rows * std::max(params.present_compressed_length, 1);

  // The present state is produced even when there is no query row to score, so that a zero-length step still
  // forwards the incoming cache unchanged.
  const int64_t past_compressed_elements =
      static_cast<int64_t>(params.batch_size) * params.past_compressed_capacity * params.head_size;
  if (past_compressed_elements > 0 && past_compressed_key != present_compressed_key) {
    CsaCopyPastCompressedKernel<T><<<GridForElements(past_compressed_elements), kThreads, 0, stream>>>(
        past_compressed_key, present_compressed_key, params);
  }

  if (params.batch_size > 0 && params.new_window_count > 0) {
    ORT_RETURN_IF_ERROR(
        ConfigureDynamicSharedMemory(CsaCompressKernel<T>, value_shared_bytes, "CsaCompressKernel"));
    const int compress_blocks = static_cast<int>(
        std::min<int64_t>(static_cast<int64_t>(params.batch_size) * params.new_window_count, kMaxGridDimX));
    CsaCompressKernel<T><<<compress_blocks, kThreads, value_shared_bytes, stream>>>(
        key, gate, past_kv_buffer, past_gate_buffer, position_bias, key_norm_weight, cos_cache, sin_cache,
        present_compressed_key, new_compressed_key, params);
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

  ORT_RETURN_IF_ERROR(
      ConfigureDynamicSharedMemory(RotateQueryKernel<T, false>, value_shared_bytes, "RotateQueryKernel"));
  const int rotate_blocks = static_cast<int>(std::min<int64_t>(rows * params.num_heads, kMaxGridDimX));
  RotateQueryKernel<T, false><<<rotate_blocks, kThreads, value_shared_bytes, stream>>>(
      query, query_norm_weight, cos_cache, sin_cache, position_ids, query_rotated, params);

  if (params.present_compressed_length > 0) {
    CsaScoreKernel<T><<<GridForElements(rows * params.present_compressed_length), kThreads, 0, stream>>>(
        query_rotated, present_compressed_key, new_compressed_key, head_weights, position_ids, scores, params);
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
