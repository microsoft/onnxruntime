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
    T* cache_k,            // [B, C, HS]  (kv_num_heads=1)
    T* cache_v,            // [B, C, HS]
    const T* q_full,       // [B, S, NH, HS]
    const T* new_k,        // [B, S, HS]
    const T* new_v,        // [B, S, HS]
    const T* bias,         // [bb, bh, bq, bk] or nullptr
    int64_t bb, int64_t bh, int64_t bq, int64_t bk,
    const T* sink,         // [1 or NH] sink logit per head
    const int32_t* seqlens_k,
    int S, int NH, int HS, int C, int W, float scale, int sink_count) {
  const int b = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int bdx = static_cast<int>(blockDim.x);

  // Dynamic shared memory: logits[C] + exp_vals[C] + scratch[bdx]
  extern __shared__ float smem[];
  float* logits   = smem;
  float* exp_vals = smem + C;
  float* scratch  = smem + 2 * C;

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

      // Broadcast max_logit from thread 0 to the whole block.
      if (tid == 0) scratch[0] = max_logit;
      __syncthreads();
      max_logit = scratch[0];
      max_logit = fmaxf(max_logit, sink_val);

      // --- Softmax with sink (serial in thread 0, broadcast via smem) ---
      if (tid == 0) {
        float exp_sum = expf(sink_val - max_logit);
        for (int i = 0; i < attended_len; ++i) {
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
Status LaunchDeepSeekV4CacheAndAttentionKernel(
    cudaStream_t stream,
    T* context, T* present_key, T* present_value,
    const T* q_full, const T* new_key, const T* new_value,
    const T* attention_bias,
    int64_t bias_b_dim, int64_t bias_h_dim, int64_t bias_q_dim, int64_t bias_k_dim,
    const T* head_sink, const int32_t* seqlens_k,
    int batch_size, int sequence_length,
    int num_heads, int head_size, int cache_capacity,
    int local_window_size, float scale, int head_sink_count,
    int max_threads_per_block) {
  const int bdx = ChooseBlockSize(head_size, max_threads_per_block);
  const int grid = batch_size;
  // Shared memory: logits[C] + exp_vals[C] + scratch[bdx]
  const size_t smem = static_cast<size_t>(2 * cache_capacity + bdx) * sizeof(float);

  DeepSeekV4CacheAndAttentionKernel<T><<<grid, bdx, smem, stream>>>(
      context, present_key, present_value,
      q_full, new_key, new_value,
      attention_bias, bias_b_dim, bias_h_dim, bias_q_dim, bias_k_dim,
      head_sink, seqlens_k,
      sequence_length, num_heads, head_size, cache_capacity,
      local_window_size, scale, head_sink_count);
  return CUDA_CALL(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE(T)                                                                    \
  template Status LaunchSplitKVRowKernel<T>(                                             \
      cudaStream_t, T*, T*, const T*, int, int, int, int);                               \
  template Status LaunchDeepSeekV4CacheAndAttentionKernel<T>(                            \
      cudaStream_t, T*, T*, T*, const T*, const T*, const T*, const T*,                  \
      int64_t, int64_t, int64_t, int64_t, const T*, const int32_t*,                      \
      int, int, int, int, int, int, float, int, int);

INSTANTIATE(half)
INSTANTIATE(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
