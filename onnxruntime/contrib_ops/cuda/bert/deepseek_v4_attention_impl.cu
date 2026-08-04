// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/deepseek_v4_attention_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ---------------------------------------------------------------------------
// Helpers: convert any supported type to/from float in device code.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float CvtToFloat(T v) {
  return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float CvtToFloat<half>(half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T CvtFromFloat(float v) {
  return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half CvtFromFloat<half>(float v) {
  return __float2half(v);
}

// ---------------------------------------------------------------------------
// SplitKVRowKernel
//
// Input : kv_row [batch_seq, kv_width]
// Output: key    [batch_seq, head_size]  ← kv_row[:, :head_size]
//         value  [batch_seq, head_size]  ← kv_row[:, head_size:2*head_size]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void SplitKVRowKernel(T* key, T* value, const T* kv_row,
                                  int batch_seq, int head_size, int kv_width) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_seq * head_size) return;
  int row = idx / head_size;
  int col = idx % head_size;
  key[idx] = kv_row[row * kv_width + col];
  value[idx] = kv_row[row * kv_width + head_size + col];
}

template <typename T>
__device__ __forceinline__ T ReadCombinedToken(
    const T* pending, const T* current,
    int batch, int token, int dim,
    int pending_tokens, int sequence_length, int width) {
  if (token < pending_tokens) {
    return pending[(batch * pending_tokens + token) * width + dim];
  }
  return current[(batch * sequence_length + token - pending_tokens) * width + dim];
}

template <typename T>
__global__ void WriteCompressorPendingKernel(
    T* pending_kv_out, T* pending_gate_out,
    const T* current_kv, const T* current_gate,
    const T* past_pending_kv, const T* past_pending_gate,
    int batch_size, int sequence_length, int past_pending_tokens,
    int usable_tokens, int pending_tokens, int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = batch_size * pending_tokens * width;
  if (index >= count) {
    return;
  }

  const int dim = index % width;
  const int row = index / width;
  const int token = row % pending_tokens;
  const int batch = row / pending_tokens;
  const int source_token = usable_tokens + token;
  pending_kv_out[index] = ReadCombinedToken(
      past_pending_kv, current_kv, batch, source_token, dim,
      past_pending_tokens, sequence_length, width);
  pending_gate_out[index] = ReadCombinedToken(
      past_pending_gate, current_gate, batch, source_token, dim,
      past_pending_tokens, sequence_length, width);
}

template <typename T>
__global__ void DeepSeekV4CompressorKernel(
    T* entries, T* overlap_kv_out, T* overlap_gate_out,
    const T* current_kv, const T* current_gate,
    const T* past_pending_kv, const T* past_pending_gate,
    const T* past_overlap_kv, const T* past_overlap_gate,
    const T* position_bias, const T* norm_weight,
    const T* cos_cache, const T* sin_cache,
    int sequence_length, int past_pending_tokens,
    int old_entry_count, int new_entry_count,
    int width, int head_size, int compress_rate,
    int rotary_dim, int cos_cache_width, float epsilon, bool is_csa) {
  const int batch = blockIdx.x / new_entry_count;
  const int window = blockIdx.x % new_entry_count;
  const int tid = threadIdx.x;
  const int slots = is_csa ? 2 * compress_rate : compress_rate;
  extern __shared__ float scratch[];
  float* row = scratch;
  float* reduction = scratch + head_size;

  float local_squares = 0.0f;
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    float max_logit = -FLT_MAX;
    for (int slot = 0; slot < slots; ++slot) {
      float logit;
      if (!is_csa) {
        const int token = window * compress_rate + slot;
        logit = CvtToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, dim,
            past_pending_tokens, sequence_length, width));
        logit += CvtToFloat(position_bias[(slot % compress_rate) * width + dim]);
      } else if (slot < compress_rate && window == 0) {
        logit = CvtToFloat(past_overlap_gate[(batch * compress_rate + slot) * head_size + dim]);
      } else {
        const int token = slot < compress_rate
                              ? (window - 1) * compress_rate + slot
                              : window * compress_rate + slot - compress_rate;
        const int source_dim = (slot < compress_rate ? 0 : head_size) + dim;
        logit = CvtToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, source_dim,
            past_pending_tokens, sequence_length, width));
        logit += CvtToFloat(position_bias[(token % compress_rate) * width + source_dim]);
      }
      max_logit = fmaxf(max_logit, logit);
    }

    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    for (int slot = 0; slot < slots; ++slot) {
      float value;
      float logit;
      if (!is_csa) {
        const int token = window * compress_rate + slot;
        value = CvtToFloat(ReadCombinedToken(
            past_pending_kv, current_kv, batch, token, dim,
            past_pending_tokens, sequence_length, width));
        logit = CvtToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, dim,
            past_pending_tokens, sequence_length, width));
        logit += CvtToFloat(position_bias[(slot % compress_rate) * width + dim]);
      } else if (slot < compress_rate && window == 0) {
        value = CvtToFloat(past_overlap_kv[(batch * compress_rate + slot) * head_size + dim]);
        logit = CvtToFloat(past_overlap_gate[(batch * compress_rate + slot) * head_size + dim]);
      } else {
        const int token = slot < compress_rate
                              ? (window - 1) * compress_rate + slot
                              : window * compress_rate + slot - compress_rate;
        const int source_dim = (slot < compress_rate ? 0 : head_size) + dim;
        value = CvtToFloat(ReadCombinedToken(
            past_pending_kv, current_kv, batch, token, source_dim,
            past_pending_tokens, sequence_length, width));
        logit = CvtToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, source_dim,
            past_pending_tokens, sequence_length, width));
        logit += CvtToFloat(position_bias[(token % compress_rate) * width + source_dim]);
      }
      const float weight = expf(logit - max_logit);
      weighted_sum += weight * value;
      weight_sum += weight;
    }
    row[dim] = weight_sum > 0.0f ? weighted_sum / weight_sum : 0.0f;
    local_squares += row[dim] * row[dim];
  }

  reduction[tid] = local_squares;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(reduction[0] / static_cast<float>(head_size) + epsilon);
  T* output_entry = entries + (batch * (old_entry_count + new_entry_count) + old_entry_count + window) * head_size;
  for (int dim = tid; dim < head_size; dim += blockDim.x) {
    output_entry[dim] = CvtFromFloat<T>(row[dim] * inv_rms * CvtToFloat(norm_weight[dim]));
  }
  __syncthreads();

  const int rotary_start = head_size - rotary_dim;
  const int position = (old_entry_count + window) * compress_rate;
  for (int pair = tid; pair < rotary_dim / 2; pair += blockDim.x) {
    const int dim = rotary_start + 2 * pair;
    const float x0 = CvtToFloat(output_entry[dim]);
    const float x1 = CvtToFloat(output_entry[dim + 1]);
    const float cosine = CvtToFloat(cos_cache[position * cos_cache_width + pair]);
    const float sine = CvtToFloat(sin_cache[position * cos_cache_width + pair]);
    output_entry[dim] = CvtFromFloat<T>(x0 * cosine - x1 * sine);
    output_entry[dim + 1] = CvtFromFloat<T>(x0 * sine + x1 * cosine);
  }

  if (is_csa && window == new_entry_count - 1) {
    for (int dim = tid; dim < head_size; dim += blockDim.x) {
      for (int slot = 0; slot < compress_rate; ++slot) {
        const int token = window * compress_rate + slot;
        const int output_index = (batch * compress_rate + slot) * head_size + dim;
        overlap_kv_out[output_index] = ReadCombinedToken(
            past_pending_kv, current_kv, batch, token, dim,
            past_pending_tokens, sequence_length, width);
        const float gate = CvtToFloat(ReadCombinedToken(
            past_pending_gate, current_gate, batch, token, dim,
            past_pending_tokens, sequence_length, width));
        overlap_gate_out[output_index] = CvtFromFloat<T>(
            gate + CvtToFloat(position_bias[slot * width + dim]));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// DeepSeekV4CacheAndAttentionKernel
//
// Grid : (batch_size,)
// Block: (thread_count,)  where thread_count <= head_size, power-of-2
//
// Each block processes all S sequence steps and all NH heads for one
// batch element b.  Steps are serialised to maintain the correct
// sliding-window KV-cache state between steps.
//
// Shared-memory layout (all float):
//   [0 .. C-1]           logits   (C = cache_capacity)
//   [C .. 2C-1]          exp_vals
//   [2C .. 2C+BDX-1]     reduction scratch (BDX = blockDim.x)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void DeepSeekV4CacheAndAttentionKernel(
    T* context,            // [B, S, NH, HS]
  float* attention_workspace,
    T* cache_k,            // [B, C, HS]  (kv_num_heads=1)
    T* cache_v,            // [B, C, HS]
    const T* q_full,       // [B, S, NH, HS]
    const T* new_k,        // [B, S, HS]
    const T* new_v,        // [B, S, HS]
    const T* compressed_entries,  // [B, E, HS] or nullptr
    const T* bias,         // [bb, bh, bq, bk] or nullptr
    int64_t bb, int64_t bh, int64_t bq, int64_t bk,
    const T* sink,         // [1 or NH] sink logit per head
    const int32_t* seqlens_k,
    const int64_t* position_ids,
    int S, int NH, int HS, int C, int W,
    int E, int compress_rate, int index_topk, int attention_mode,
    float scale, int sink_count) {
  const int b = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int bdx = static_cast<int>(blockDim.x);

  const int max_selected_entries = attention_mode == 1 ? min(index_topk, E) : E;
  const int score_capacity = C + max_selected_entries;
  float* workspace = attention_workspace +
                     static_cast<size_t>(b) * (2 * score_capacity + bdx);
  float* logits = workspace;
  float* exp_vals = workspace + score_capacity;
  float* scratch = workspace + 2 * score_capacity;

  int seq_len_k = static_cast<int>(seqlens_k[b]);

  for (int s = 0; s < S; ++s) {
    // =========================================================================
    // Stage 1 — Update KV cache for token (b, s).
    // =========================================================================
    const T* nk = new_k + (b * S + s) * HS;
    const T* nv = new_v + (b * S + s) * HS;

    if (seq_len_k >= C) {
      // Shift entire cache left by one slot (all (C-1)*HS elements).
      int base = b * C * HS;
      for (int idx = tid; idx < (C - 1) * HS; idx += bdx) {
        cache_k[base + idx] = cache_k[base + idx + HS];
        cache_v[base + idx] = cache_v[base + idx + HS];
      }
      __syncthreads();
      seq_len_k = C - 1;
    }

    // Write new token at seq_len_k.
    {
      int base = (b * C + seq_len_k) * HS;
      for (int d = tid; d < HS; d += bdx) {
        cache_k[base + d] = nk[d];
        cache_v[base + d] = nv[d];
      }
      __syncthreads();
    }

    // =========================================================================
    // Stage 2 — Sliding-window attention for each head.
    // =========================================================================
    const int available_len = min(seq_len_k + 1, C);
    const int attended_len  = min(W, available_len);
    const int cache_start   = available_len - attended_len;
    // key_total_start: absolute sequence index of the first attended cache slot.
    const int key_total_start = seq_len_k - available_len + 1 + cache_start;
    const int visible_entries = attention_mode == 0
                    ? 0
                    : min(static_cast<int>((position_ids[b * S + s] + 1) / compress_rate), E);
    const int selected_entries = attention_mode == 1 ? min(index_topk, visible_entries) : visible_entries;
    const int first_entry = attention_mode == 1 ? visible_entries - selected_entries : 0;

    for (int h = 0; h < NH; ++h) {
      const T* q_head = q_full + ((b * S + s) * NH + h) * HS;
      const float sink_val = CvtToFloat<T>(sink[sink_count == 1 ? 0 : h]);

      // --- Compute dot products and record per-slot logits ---
      float max_logit = -FLT_MAX;

      for (int i = 0; i < attended_len; ++i) {
        const T* k = cache_k + (b * C + cache_start + i) * HS;

        float partial = 0.0f;
        for (int d = tid; d < HS; d += bdx) {
          partial += CvtToFloat<T>(q_head[d]) * CvtToFloat<T>(k[d]);
        }
        scratch[tid] = partial;
        __syncthreads();

        // Tree reduction within block to get full dot product in scratch[0].
        for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
          if (tid < stride) scratch[tid] += scratch[tid + stride];
          __syncthreads();
        }

        if (tid == 0) {
          float logit = scratch[0] * scale;

          if (bias != nullptr) {
            int64_t key_total_idx = static_cast<int64_t>(key_total_start + i);
            if (key_total_idx >= 0 && key_total_idx < bk) {
              int64_t b_idx = (bb == 1) ? 0 : static_cast<int64_t>(b);
              int64_t h_idx = (bh == 1) ? 0 : static_cast<int64_t>(h);
              int64_t s_idx = (bq > static_cast<int64_t>(s)) ? static_cast<int64_t>(s) : bq - 1;
              int64_t bias_idx = ((b_idx * bh + h_idx) * bq + s_idx) * bk + key_total_idx;
              logit += CvtToFloat<T>(bias[bias_idx]);
            }
          }

          logits[i] = logit;
          if (logit > max_logit) max_logit = logit;
          // Stash updated max_logit for all iterations (serial, only tid==0).
        }
        __syncthreads();
      }

      for (int i = 0; i < selected_entries; ++i) {
        const T* entry = compressed_entries + (b * E + first_entry + i) * HS;
        float partial = 0.0f;
        for (int d = tid; d < HS; d += bdx) {
          partial += CvtToFloat<T>(q_head[d]) * CvtToFloat<T>(entry[d]);
        }
        scratch[tid] = partial;
        __syncthreads();
        for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
          if (tid < stride) scratch[tid] += scratch[tid + stride];
          __syncthreads();
        }
        if (tid == 0) {
          const float logit = scratch[0] * scale;
          logits[attended_len + i] = logit;
          if (logit > max_logit) max_logit = logit;
        }
        __syncthreads();
      }

      // Broadcast max_logit from thread 0 to the whole block.
      if (tid == 0) scratch[0] = max_logit;
      __syncthreads();
      max_logit = scratch[0];
      max_logit = fmaxf(max_logit, sink_val);

      // --- Softmax with sink (serial in thread 0, broadcast via smem) ---
      if (tid == 0) {
        float exp_sum = expf(sink_val - max_logit);
        for (int i = 0; i < attended_len + selected_entries; ++i) {
          float e = expf(logits[i] - max_logit);
          exp_vals[i] = e;
          exp_sum += e;
        }
        // Store reciprocal exp_sum for the weighted sum below.
        scratch[0] = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
      }
      __syncthreads();
      const float inv_exp_sum = scratch[0];

      // --- Weighted value accumulation (parallel across head dimensions) ---
      T* out_head = context + ((b * S + s) * NH + h) * HS;
      for (int d = tid; d < HS; d += bdx) {
        float c = 0.0f;
        for (int i = 0; i < attended_len; ++i) {
          float w = exp_vals[i] * inv_exp_sum;
          c += w * CvtToFloat<T>(cache_v[(b * C + cache_start + i) * HS + d]);
        }
        for (int i = 0; i < selected_entries; ++i) {
          const float w = exp_vals[attended_len + i] * inv_exp_sum;
          c += w * CvtToFloat<T>(compressed_entries[(b * E + first_entry + i) * HS + d]);
        }
        out_head[d] = CvtFromFloat<T>(c);
      }
      __syncthreads();
    }  // end heads loop

    ++seq_len_k;
  }  // end sequence-step loop
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

template <typename T>
Status LaunchSplitKVRowKernel(
    cudaStream_t stream,
    T* key, T* value, const T* kv_row,
    int batch_seq, int head_size, int kv_width,
    int max_threads_per_block) {
  const int total = batch_seq * head_size;
  const int block = std::min(max_threads_per_block, 256);
  const int grid  = (total + block - 1) / block;
  SplitKVRowKernel<T><<<grid, block, 0, stream>>>(key, value, kv_row, batch_seq, head_size, kv_width);
  return CUDA_CALL(cudaGetLastError());
}

// Choose the largest power-of-2 thread count that is <= min(head_size, max_threads).
static int ChooseBlockSize(int head_size, int max_threads) {
  int t = std::min(head_size, max_threads);
  // round down to power of 2
  int p = 1;
  while (p * 2 <= t) p *= 2;
  return p;
}

template <typename T>
Status LaunchDeepSeekV4CompressorKernel(
    cudaStream_t stream,
    T* entries, T* pending_kv_out, T* pending_gate_out,
    T* overlap_kv_out, T* overlap_gate_out,
    const T* current_kv, const T* current_gate,
    const T* past_pending_kv, const T* past_pending_gate,
    const T* past_overlap_kv, const T* past_overlap_gate,
    const T* position_bias, const T* norm_weight,
    const T* cos_cache, const T* sin_cache,
    int batch_size, int sequence_length, int pending_token_count,
    int old_entry_count, int new_entry_count, int width, int head_size,
    int compress_rate, int rotary_dim, int cos_cache_width,
    float epsilon, bool is_csa, int max_threads_per_block) {
  const int total_tokens = pending_token_count + sequence_length;
  const int usable_tokens = new_entry_count * compress_rate;
  const int output_pending_tokens = total_tokens - usable_tokens;
  if (output_pending_tokens > 0 && pending_kv_out != nullptr && pending_gate_out != nullptr) {
    const int count = batch_size * output_pending_tokens * width;
    const int block = std::min(max_threads_per_block, 256);
    WriteCompressorPendingKernel<T><<<(count + block - 1) / block, block, 0, stream>>>(
        pending_kv_out, pending_gate_out, current_kv, current_gate,
        past_pending_kv, past_pending_gate, batch_size, sequence_length,
        pending_token_count, usable_tokens, output_pending_tokens, width);
  }

  if (new_entry_count > 0) {
    const int block = ChooseBlockSize(head_size, max_threads_per_block);
    const size_t shared_memory = static_cast<size_t>(head_size + block) * sizeof(float);
    DeepSeekV4CompressorKernel<T><<<batch_size * new_entry_count, block, shared_memory, stream>>>(
        entries, overlap_kv_out, overlap_gate_out,
        current_kv, current_gate, past_pending_kv, past_pending_gate,
        past_overlap_kv, past_overlap_gate, position_bias, norm_weight,
        cos_cache, sin_cache, sequence_length, pending_token_count,
        old_entry_count, new_entry_count, width, head_size, compress_rate,
        rotary_dim, cos_cache_width, epsilon, is_csa);
  }
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchDeepSeekV4CacheAndAttentionKernel(
    cudaStream_t stream,
  T* context, float* attention_workspace, T* present_key, T* present_value,
    const T* q_full, const T* new_key, const T* new_value,
    const T* compressed_entries,
    const T* attention_bias,
    int64_t bias_b_dim, int64_t bias_h_dim, int64_t bias_q_dim, int64_t bias_k_dim,
    const T* head_sink, const int32_t* seqlens_k,
    const int64_t* position_ids,
    int batch_size, int sequence_length,
    int num_heads, int head_size, int cache_capacity,
    int local_window_size, int compressed_entry_count,
    int compress_rate, int index_topk, int attention_mode,
    float scale, int head_sink_count,
    int max_threads_per_block) {
  const int bdx = ChooseBlockSize(head_size, max_threads_per_block);
  const int grid = batch_size;
  const int max_selected_entries = attention_mode == 1
                                       ? std::min(index_topk, compressed_entry_count)
                                       : compressed_entry_count;
  DeepSeekV4CacheAndAttentionKernel<T><<<grid, bdx, 0, stream>>>(
      context, attention_workspace, present_key, present_value,
      q_full, new_key, new_value, compressed_entries,
      attention_bias, bias_b_dim, bias_h_dim, bias_q_dim, bias_k_dim,
      head_sink, seqlens_k, position_ids,
      sequence_length, num_heads, head_size, cache_capacity,
      local_window_size, compressed_entry_count, compress_rate, index_topk,
      attention_mode, scale, head_sink_count);
  return CUDA_CALL(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                                    \
  template Status LaunchSplitKVRowKernel<T>(                                             \
      cudaStream_t, T*, T*, const T*, int, int, int, int);                               \
  template Status LaunchDeepSeekV4CompressorKernel<T>(                                   \
      cudaStream_t, T*, T*, T*, T*, T*, const T*, const T*, const T*, const T*,          \
      const T*, const T*, const T*, const T*, const T*, const T*, int, int, int, int,     \
      int, int, int, int, int, int, float, bool, int);                                    \
  template Status LaunchDeepSeekV4CacheAndAttentionKernel<T>(                            \
      cudaStream_t, T*, float*, T*, T*, const T*, const T*, const T*, const T*, const T*, \
      int64_t, int64_t, int64_t, int64_t, const T*, const int32_t*, const int64_t*,       \
      int, int, int, int, int, int, int, int, int, int, float, int, int);

INSTANTIATE(half)
INSTANTIATE(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
