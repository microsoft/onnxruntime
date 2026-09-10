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

#include <algorithm>

#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// The block reductions below halve the active thread count, so this must stay a power of two.
constexpr int kThreads = 128;
constexpr int64_t kMaxGridDimX = 2147483647;

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
  for (int candidate = static_cast<int>(threadIdx.x); candidate < count;
       candidate += static_cast<int>(blockDim.x)) {
    const float value = scores[candidate];
    if (previous_index >= 0 &&
        !(value < previous_score || (value == previous_score && candidate > previous_index))) {
      continue;
    }
    if (*best_index < 0 || value > *best_value ||
        (value == *best_value && candidate < *best_index)) {
      *best_value = value;
      *best_index = candidate;
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
  return position < 0 ? 0 : (position + 1) / compress_ratio;
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
__global__ void ConcatPastKeyKernel(const T* past_key, const T* key, T* present_key,
                                    SparseAttentionIndexerParams params) {
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.total_sequence_length * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(index % params.head_size);
    const int64_t token_row = index / params.head_size;
    const int token = static_cast<int>(token_row % params.total_sequence_length);
    const int batch = static_cast<int>(token_row / params.total_sequence_length);
    present_key[index] =
        token < params.past_sequence_length
            ? past_key[(static_cast<int64_t>(batch) * params.past_sequence_length + token) * params.head_size + d]
            : key[(static_cast<int64_t>(batch) * params.sequence_length + (token - params.past_sequence_length)) *
                      params.head_size +
                  d];
  }
}

// One block per query row; compacts the visible key positions of that row into visible_indices.
__global__ void CompactVisibleKernel(const bool* mask, int32_t* visible_indices, int32_t* visible_count,
                                     SparseAttentionIndexerParams params) {
  extern __shared__ int32_t shared_scan[];
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const bool* mask_row = mask + row * params.total_sequence_length;
    int32_t* out_row = visible_indices + row * params.total_sequence_length;
    int32_t offset = 0;
    for (int base = 0; base < params.total_sequence_length; base += blockDim.x) {
      const int position = base + static_cast<int>(threadIdx.x);
      const int32_t flag = (position < params.total_sequence_length && mask_row[position]) ? 1 : 0;
      shared_scan[threadIdx.x] = flag;
      __syncthreads();
      for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        const int32_t addend = (threadIdx.x >= static_cast<unsigned>(stride)) ? shared_scan[threadIdx.x - stride] : 0;
        __syncthreads();
        shared_scan[threadIdx.x] += addend;
        __syncthreads();
      }
      if (flag != 0) {
        out_row[offset + shared_scan[threadIdx.x] - 1] = position;
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
__global__ void QsaBlockScoreKernel(const T* present_key, const T* key_norm_weight, const T* cos_cache,
                                    const T* sin_cache, const float* query_rotated,
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
    const int64_t key_base = static_cast<int64_t>(batch) * params.total_sequence_length * params.head_size;

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
      const float* query_head = query_rotated + (row * params.num_heads + head) * params.head_size;
      float partial = 0.0f;
      for (int d = threadIdx.x; d < params.head_size; d += blockDim.x) {
        partial += query_head[d] * rotated[d];
      }
      score += fmaxf(BlockSum(partial, reduction), 0.0f);
    }

    if (threadIdx.x == 0) {
      block_scores[row * params.max_block_count + block_index] = score * params.scale;
    }
    __syncthreads();
  }
}

// One block per query row. Emits the token indices of the highest scoring blocks followed by the
// visible tokens of the trailing incomplete block.
__global__ void QsaSelectKernel(const float* block_scores, const int32_t* visible_indices,
                                const int32_t* visible_count, int32_t* selected_indices,
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

    const int visible = visible_count[row];
    const int block_count = visible / params.compress_ratio;
    const int selected = min(params.block_topk, block_count);
    const float* scores_row = block_scores + row * params.max_block_count;
    const int32_t* index_row = visible_indices + row * params.total_sequence_length;

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
        (static_cast<int64_t>(batch) * params.present_compressed_length + entry) * params.head_size;
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
  const int64_t total = static_cast<int64_t>(params.batch_size) * params.past_compressed_length * params.head_size;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t entry_row = index / params.head_size;
    const int entry = static_cast<int>(entry_row % params.past_compressed_length);
    const int batch = static_cast<int>(entry_row / params.past_compressed_length);
    const int d = static_cast<int>(index % params.head_size);
    present_compressed_key[(static_cast<int64_t>(batch) * params.present_compressed_length + entry) * params.head_size +
                           d] = past_compressed_key[index];
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
        (static_cast<int64_t>(batch) * params.present_compressed_length + entry) * params.head_size;
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
  return rows * params.num_heads * params.head_size + rows * std::max(params.max_block_count, 1);
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
Status LaunchQsaSparseAttentionIndexer(cudaStream_t stream, const SparseAttentionIndexerParams& params,
                                       const T* query, const T* key, const T* key_norm_weight,
                                       const T* cos_cache, const T* sin_cache, const bool* mask,
                                       const T* past_key, int32_t* selected_indices, T* present_key,
                                       float* float_workspace, int32_t* int_workspace) {
  const int64_t rows = static_cast<int64_t>(params.batch_size) * params.sequence_length;

  // The present state is produced even when there is no query row to score, so that a zero-length step still
  // forwards the incoming cache unchanged.
  const int64_t present_key_elements =
      static_cast<int64_t>(params.batch_size) * params.total_sequence_length * params.head_size;
  if (present_key_elements > 0) {
    ConcatPastKeyKernel<T><<<GridForElements(present_key_elements), kThreads, 0, stream>>>(past_key, key, present_key,
                                                                                           params);
  }

  if (rows == 0) {
    return CUDA_CALL(cudaGetLastError());
  }

  float* query_rotated = float_workspace;
  float* block_scores = float_workspace + rows * params.num_heads * params.head_size;
  int32_t* visible_indices = int_workspace;
  int32_t* visible_count = int_workspace + rows * params.total_sequence_length;

  const size_t value_bytes = static_cast<size_t>(params.head_size) * sizeof(float);

  const int rotate_blocks = static_cast<int>(std::min<int64_t>(rows * params.num_heads, kMaxGridDimX));
  RotateQueryKernel<T, true><<<rotate_blocks, kThreads, value_bytes, stream>>>(
      query, cos_cache, sin_cache, nullptr, query_rotated, params);

  const int row_blocks = static_cast<int>(std::min<int64_t>(rows, kMaxGridDimX));
  CompactVisibleKernel<<<row_blocks, kThreads, kThreads * sizeof(int32_t), stream>>>(
      mask, visible_indices, visible_count, params);

  if (params.max_block_count > 0) {
    const int64_t block_work = rows * params.max_block_count;
    const int score_blocks = static_cast<int>(std::min<int64_t>(block_work, kMaxGridDimX));
    QsaBlockScoreKernel<T><<<score_blocks, kThreads, 2 * value_bytes + kThreads * sizeof(float), stream>>>(
        present_key, key_norm_weight, cos_cache, sin_cache, query_rotated, visible_indices, visible_count,
        block_scores, params);
  }

  QsaSelectKernel<<<row_blocks, kThreads, kThreads * (sizeof(float) + sizeof(int)), stream>>>(
      block_scores, visible_indices, visible_count, selected_indices, params);

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
      static_cast<int64_t>(params.batch_size) * params.past_compressed_length * params.head_size;
  if (past_compressed_elements > 0) {
    CsaCopyPastCompressedKernel<T><<<GridForElements(past_compressed_elements), kThreads, 0, stream>>>(
        past_compressed_key, present_compressed_key, params);
  }

  if (params.new_window_count > 0) {
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

#define INSTANTIATE_SPARSE_ATTENTION_INDEXER(T)                                                              \
  template Status LaunchQsaSparseAttentionIndexer<T>(cudaStream_t, const SparseAttentionIndexerParams&,      \
                                                     const T*, const T*, const T*, const T*, const T*,       \
                                                     const bool*, const T*, int32_t*, T*, float*, int32_t*); \
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
