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
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <limits>

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_device_math.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// The block reductions below halve the active thread count, so this must stay a power of two.
constexpr int kThreads = 128;
constexpr int kWarpSize = 32;
constexpr int64_t kMaxGridDimX = kSaiMaxGridDimX;
constexpr int kRadixSortMinBlockCount = 512;

// Thin, same-signature aliases over the device math shared with the packed indexer implementation
// (sparse_attention_indexer_device_math.cuh), so the kernels below are unchanged.
__device__ __forceinline__ float NegativeInfinity() { return SaiNegativeInfinity(); }

int GridForElements(int64_t count) { return SaiGridForElements(count, kThreads); }

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
__global__ void RotateQueryKernel(const T* query, const T* cos_cache, const T* sin_cache,
                                  const int64_t* position_ids, float* query_rotated,
                                  SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length * params.num_heads;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int token = static_cast<int>((row / params.num_heads) % params.sequence_length);
    const int batch = static_cast<int>(row / (static_cast<int64_t>(params.num_heads) * params.sequence_length));
    const int64_t base = row * params.head_size;

    for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
      shared[d] = to_float<T>(query[base + d]);
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
                d] = key[index];
  }
}

// One block per query row; compacts the visible key positions of that row into visible_indices.
__global__ void CompactVisibleKernel(const bool* mask, int32_t* visible_indices, int32_t* visible_count,
                                     SparseAttentionIndexerParams params) {
  __shared__ int32_t warp_offsets[kThreads / kWarpSize + 1];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const bool* mask_row = mask + row * params.total_sequence_length;
    int32_t* out_row = visible_indices + row * params.total_sequence_length;
    int32_t offset = 0;
    for (int base = 0; base < params.total_sequence_length; base += blockDim.x) {
      const int position = base + static_cast<int>(threadIdx.x);
      const bool visible = position < params.total_sequence_length && mask_row[position];
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
    if (threadIdx.x == 0) {
      visible_count[row] = offset;
    }
    __syncthreads();
  }
}

// One block per (query row, block index). Pools compress_ratio visible keys, normalizes, rotates
// and scores the result against every query head.
template <typename T>
__global__ void QsaBlockScoreKernel(const T* query, const T* present_key, const T* key_norm_weight,
                                    const T* cos_cache, const T* sin_cache,
                                    const int32_t* visible_indices, const int32_t* visible_count,
                                    float* block_scores, SparseAttentionIndexerParams params) {
  extern __shared__ float shared[];
  float* pooled = shared;
  float* rotated = shared + params.head_size;
  float* query_head = pooled;
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
      for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
        query_head[d] = to_float<T>(query[query_base + d]);
      }
      __syncthreads();

      const int query_position = ClampPosition(
          static_cast<int64_t>(params.past_sequence_length) + row % params.sequence_length,
          params.max_rotary_length);
      const int64_t query_cache_offset =
          (static_cast<int64_t>(batch) * params.max_rotary_length + query_position) * params.rotary_width;
      float partial = 0.0f;
      for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
        partial += LeadingRope<T>(query_head, params.rotary_width, cos_cache + query_cache_offset,
                                  sin_cache + query_cache_offset, d) *
                   rotated[d];
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

    const int visible = visible_count[row];
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    const float* scores_row = block_scores + row * params.max_block_count;
    const int32_t* index_row = visible_indices + row * params.total_sequence_length;

    int emitted = selected;
    if (topk_indices != nullptr) {
      const int32_t* topk_row = topk_indices + row * params.max_block_count;
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = static_cast<int>(topk_row[rank]);
        for (int t = threadIdx.x; t < params.compress_ratio; t += blockDim.x) {
          out_row[rank * params.compress_ratio + t] =
              index_row[selected_block * params.compress_ratio + t];
        }
      }
      __syncthreads();
    } else if (selected > 0 && use_fast_topk) {
      SaiBlockTopK(scores_row, block_count, selected, shared_value, shared_index);
      for (int rank = 0; rank < selected; ++rank) {
        const int selected_block = shared_index[rank];
        for (int t = threadIdx.x; t < params.compress_ratio; t += blockDim.x) {
          out_row[rank * params.compress_ratio + t] =
              index_row[selected_block * params.compress_ratio + t];
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
          out_row[rank * params.compress_ratio + t] = index_row[previous_index * params.compress_ratio + t];
        }
        emitted = rank + 1;
        __syncthreads();
      }
    }

    const int tail_start = block_count * params.compress_ratio;
    for (int t = threadIdx.x; t < visible - tail_start; t += blockDim.x) {
      out_row[emitted * params.compress_ratio + t] = index_row[tail_start + t];
    }
    __syncthreads();
  }
}

__global__ void SetupQsaSortKernel(int32_t* indices, int32_t* offsets, int num_items,
                                   int rows, int dimension) {
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_items; index += static_cast<int>(gridDim.x) * blockDim.x) {
    indices[index] = index % dimension;
  }
  for (int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
       index <= rows; index += static_cast<int>(gridDim.x) * blockDim.x) {
    offsets[index] = index * dimension;
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
        (static_cast<int64_t>(batch) * params.max_rotary_length + position) * params.rotary_width;
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
  return rows * std::max(params.max_block_count, 1);
}

size_t GetQsaWorkspaceIntCount(const SparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  return rows * std::max(params.total_sequence_length, 1) + rows;
}

size_t GetCsaWorkspaceFloatCount(const SparseAttentionIndexerParams& params) {
  const size_t rows = static_cast<size_t>(params.batch_size) * params.sequence_length;
  return rows * params.num_heads * params.head_size + rows * std::max(params.present_compressed_length, 1);
}

template <typename T>
Status LaunchQsaSparseAttentionIndexer(const onnxruntime::cuda::CudaKernel* kernel, cudaStream_t stream,
                                       void* alloc_stream,
                                       const SparseAttentionIndexerParams& params,
                                       const T* query, const T* key, const T* key_norm_weight,
                                       const T* cos_cache, const T* sin_cache, const bool* mask,
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

  float* block_scores = float_workspace;
  int32_t* visible_indices = int_workspace;
  int32_t* visible_count = int_workspace + rows * params.total_sequence_length;

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  CompactVisibleKernel<<<row_blocks, kThreads, 0, stream>>>(
      mask, visible_indices, visible_count, params);

  if (params.max_block_count > 0) {
    const int64_t block_work = rows * params.max_block_count;
    const int score_blocks = static_cast<int>(std::min<int64_t>(block_work, kMaxGridDimX));
    QsaBlockScoreKernel<T><<<score_blocks, kThreads, 2 * value_bytes + kThreads * sizeof(float), stream>>>(
        query, present_key, key_norm_weight, cos_cache, sin_cache, visible_indices, visible_count,
        block_scores, params);
  }

  const int64_t sort_items_64 = rows * params.max_block_count;
  const bool use_radix_sort =
      params.max_block_count >= kRadixSortMinBlockCount &&
      sort_items_64 <= std::numeric_limits<int>::max();
  IAllocatorUniquePtr<float> sorted_scores;
  IAllocatorUniquePtr<int32_t> sort_indices_in;
  IAllocatorUniquePtr<int32_t> sort_indices_out;
  IAllocatorUniquePtr<int32_t> sort_offsets;
  if (use_radix_sort) {
    const int sort_items = static_cast<int>(sort_items_64);
    const int sort_rows = static_cast<int>(rows);
    sorted_scores = kernel->GetScratchBuffer<float>(sort_items, alloc_stream);
    sort_indices_in = kernel->GetScratchBuffer<int32_t>(sort_items, alloc_stream);
    sort_indices_out = kernel->GetScratchBuffer<int32_t>(sort_items, alloc_stream);
    sort_offsets = kernel->GetScratchBuffer<int32_t>(static_cast<size_t>(sort_rows) + 1, alloc_stream);
    SetupQsaSortKernel<<<GridForElements(std::max(sort_items, sort_rows + 1)), kThreads, 0, stream>>>(
        sort_indices_in.get(), sort_offsets.get(), sort_items, sort_rows, params.max_block_count);

    size_t temp_storage_bytes = 0;
    ORT_RETURN_IF_ERROR(CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, block_scores, sorted_scores.get(),
        sort_indices_in.get(), sort_indices_out.get(), sort_items, sort_rows,
        sort_offsets.get(), sort_offsets.get() + 1, 0, sizeof(float) * 8, stream)));
    auto temp_storage = kernel->GetScratchBuffer<void>(temp_storage_bytes, alloc_stream);
    ORT_RETURN_IF_ERROR(CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_storage.get(), temp_storage_bytes, block_scores, sorted_scores.get(),
        sort_indices_in.get(), sort_indices_out.get(), sort_items, sort_rows,
        sort_offsets.get(), sort_offsets.get() + 1, 0, sizeof(float) * 8, stream)));
  }

  const int topk_shared_entries = !use_radix_sort && params.block_topk <= kSaiFastTopKMax
                                      ? kThreads * params.block_topk
                                      : kThreads;
  QsaSelectKernel<<<row_blocks, kThreads,
                    static_cast<size_t>(topk_shared_entries) * (sizeof(float) + sizeof(int)), stream>>>(
      block_scores, visible_indices, visible_count, sort_indices_out.get(), selected_indices, params);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchCsaSparseAttentionIndexer(cudaStream_t stream, const SparseAttentionIndexerParams& params,
                                       const T* query, const T* key, const T* key_norm_weight,
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
  RotateQueryKernel<T, false><<<rotate_blocks, kThreads, value_bytes, stream>>>(
      query, cos_cache, sin_cache, position_ids, query_rotated, params);

  if (params.present_compressed_length > 0) {
    CsaScoreKernel<T><<<GridForElements(rows * params.present_compressed_length), kThreads, 0, stream>>>(
        query_rotated, present_compressed_key, head_weights, position_ids, scores, params);
  }

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  CsaSelectKernel<<<row_blocks, kThreads, kThreads * (sizeof(float) + sizeof(int)), stream>>>(
      scores, position_ids, selected_indices, params);

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_SPARSE_ATTENTION_INDEXER(T)                                                                \
  template Status LaunchQsaSparseAttentionIndexer<T>(const onnxruntime::cuda::CudaKernel*, cudaStream_t, void*, \
                                                     const SparseAttentionIndexerParams&,                      \
                                                     const T*, const T*, const T*, const T*, const T*,         \
                                                     const bool*, const T*, int32_t*, T*, float*, int32_t*);   \
  template Status LaunchCsaSparseAttentionIndexer<T>(                                                        \
      cudaStream_t, const SparseAttentionIndexerParams&, const T*, const T*, const T*, const T*, const T*,   \
      const T*, const T*, const T*, const int64_t*, const T*, const T*, const T*, int32_t*, T*, T*, T*, float*);

INSTANTIATE_SPARSE_ATTENTION_INDEXER(float)
INSTANTIATE_SPARSE_ATTENTION_INDEXER(half)
INSTANTIATE_SPARSE_ATTENTION_INDEXER(__nv_bfloat16)

#undef INSTANTIATE_SPARSE_ATTENTION_INDEXER

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
