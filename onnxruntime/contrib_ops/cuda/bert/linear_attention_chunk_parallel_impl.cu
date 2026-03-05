// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/linear_attention_chunk_parallel_impl.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }

#if __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }
#endif

template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }

#if __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16(val); }
#endif

// ============================================================================
// Intra-chunk causal attention kernel (linear mode)
//
// For each chunk of size C, computes the causal linear attention:
//   For position t within the chunk:
//     intra_output[t] = q[t]^T * (sum_{s<=t} k[s] ⊗ v[s])
//
// This is the parallel component — within each chunk, all positions can
// share computation through cumulative outer products.
//
// Each block handles one (batch, head, chunk) triple.
// ============================================================================
template <typename T, int UPDATE_RULE>
__global__ void IntraChunkKernel(
    const T* __restrict__ query,    // (B, H, T, d_k)
    const T* __restrict__ key,      // (B, H, T, d_k)
    const T* __restrict__ value,    // (B, H, T, d_v)
    const T* __restrict__ decay,    // (B, H, T, d_k) or nullptr
    const T* __restrict__ beta,     // (B, H, T, 1) or nullptr
    T* __restrict__ output,         // (B, H, T, d_v) — intra-chunk contribution
    T* __restrict__ chunk_states,   // (B, H, num_chunks, d_k, d_v) — accumulated state per chunk
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    int num_chunks,
    bool decay_broadcasted) {
  const int bh = blockIdx.x;
  const int chunk_idx = blockIdx.y;
  const int tid = threadIdx.x;

  const int chunk_start = chunk_idx * chunk_size;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);
  const int actual_chunk_size = chunk_end - chunk_start;

  const int bh_offset_qk = bh * seq_len * d_k;
  const int bh_offset_v = bh * seq_len * d_v;
  const int chunk_state_offset = (bh * num_chunks + chunk_idx) * d_k * d_v;

  // For the "linear" update rule, intra-chunk attention is:
  //   output[t] = sum_{s=0..t} (q[t] · k[s]) * v[s]
  //
  // We compute this sequentially within the chunk, accumulating k⊗v.
  // For delta rules, we accumulate the local state within the chunk.

  // Process each position in the chunk
  // Each thread handles a subset of d_v columns
  for (int j = tid; j < d_v; j += blockDim.x) {
    // Local state accumulator for this column
    float local_state[128];  // d_k <= 128 enforced by host validation
    for (int i = 0; i < d_k; i++) {
      local_state[i] = 0.0f;
    }

    for (int t = 0; t < actual_chunk_size; t++) {
      int global_t = chunk_start + t;
      int qk_idx = bh_offset_qk + global_t * d_k;
      int v_idx = bh_offset_v + global_t * d_v;

      float v_val = to_float(value[v_idx + j]);
      float out_val = 0.0f;

      if constexpr (UPDATE_RULE == 0) {
        // linear: state += k ⊗ v, output = q^T state
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          local_state[i] += k_val * v_val;
          float q_val = to_float(query[qk_idx + i]);
          out_val += q_val * local_state[i];
        }
      } else if constexpr (UPDATE_RULE == 1) {
        // gated: state = decay * state + k ⊗ v
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          float decay_val = 1.0f;
          if (decay != nullptr) {
            if (decay_broadcasted) {
              decay_val = expf(to_float(decay[qk_idx + i]));
            } else {
              decay_val = expf(to_float(decay[bh * seq_len + global_t]));
            }
          }
          local_state[i] = decay_val * local_state[i] + k_val * v_val;
          float q_val = to_float(query[qk_idx + i]);
          out_val += q_val * local_state[i];
        }
      } else if constexpr (UPDATE_RULE == 2) {
        // delta: state += beta * k ⊗ (v - state^T k)
        float beta_val = to_float(beta[bh * seq_len + global_t]);
        // Compute retrieved = state^T k for this column j
        float retrieved = 0.0f;
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          retrieved += local_state[i] * k_val;
        }
        float delta = v_val - retrieved;
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          local_state[i] += beta_val * k_val * delta;
          float q_val = to_float(query[qk_idx + i]);
          out_val += q_val * local_state[i];
        }
      } else {
        // gated_delta: state = decay * state + beta * k ⊗ (v - decay * state^T k)
        float beta_val = to_float(beta[bh * seq_len + global_t]);
        // Apply decay first, then compute retrieved
        float retrieved = 0.0f;
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          float decay_val = 1.0f;
          if (decay_broadcasted) {
            decay_val = expf(to_float(decay[qk_idx + i]));
          } else {
            decay_val = expf(to_float(decay[bh * seq_len + global_t]));
          }
          local_state[i] *= decay_val;
          retrieved += local_state[i] * k_val;
        }
        float delta = v_val - retrieved;
        for (int i = 0; i < d_k; i++) {
          float k_val = to_float(key[qk_idx + i]);
          local_state[i] += beta_val * k_val * delta;
          float q_val = to_float(query[qk_idx + i]);
          out_val += q_val * local_state[i];
        }
      }

      // Write intra-chunk output (will add inter-chunk contribution later)
      output[bh_offset_v + global_t * d_v + j] = from_float<T>(out_val);
    }

    // Save final local state for this chunk
    for (int i = 0; i < d_k; i++) {
      chunk_states[chunk_state_offset + i * d_v + j] = from_float<T>(local_state[i]);
    }
  }
}

// ============================================================================
// Inter-chunk state propagation kernel
//
// Sequentially accumulates chunk states:
//   cumulative_state[c] = f(cumulative_state[c-1], chunk_state[c-1])
// where f depends on the update rule (for gated modes, includes cumulative decay).
//
// Then adds the inter-chunk contribution to each position's output:
//   output[t] += scale * q[t]^T cumulative_state[chunk_of(t)]
//
// Each block handles one (batch, head) pair.
// ============================================================================
template <typename T>
__global__ void InterChunkKernel(
    const T* __restrict__ query,         // (B, H, T, d_k)
    const T* __restrict__ initial_state, // (B, H, d_k, d_v) or nullptr
    const T* __restrict__ chunk_states,  // (B, H, num_chunks, d_k, d_v)
    T* __restrict__ output,              // (B, H, T, d_v) — add inter-chunk contribution
    T* __restrict__ final_state,         // (B, H, d_k, d_v)
    float scale,
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    int num_chunks) {
  const int bh = blockIdx.x;
  const int tid = threadIdx.x;

  const int bh_offset_qk = bh * seq_len * d_k;
  const int bh_offset_v = bh * seq_len * d_v;
  const int state_size = d_k * d_v;

  // Process each d_v column
  for (int j = tid; j < d_v; j += blockDim.x) {
    // Running cumulative state
    float cum_state[128];  // d_k <= 128 enforced by host validation
    for (int i = 0; i < d_k; i++) {
      cum_state[i] = 0.0f;
    }

    // Initialize from initial_state if provided
    if (initial_state != nullptr) {
      for (int i = 0; i < d_k; i++) {
        cum_state[i] = to_float(initial_state[bh * state_size + i * d_v + j]);
      }
    }

    for (int c = 0; c < num_chunks; c++) {
      int chunk_start = c * chunk_size;
      int chunk_end = min(chunk_start + chunk_size, seq_len);

      // Add inter-chunk contribution to each position in this chunk
      // output[t] += scale * q[t]^T * cum_state
      for (int t = chunk_start; t < chunk_end; t++) {
        float inter_val = 0.0f;
        for (int i = 0; i < d_k; i++) {
          float q_val = to_float(query[bh_offset_qk + t * d_k + i]);
          inter_val += q_val * cum_state[i];
        }
        // Add to existing intra-chunk output and apply scale
        float existing = to_float(output[bh_offset_v + t * d_v + j]);
        output[bh_offset_v + t * d_v + j] = from_float<T>((existing + inter_val) * scale);
      }

      // Accumulate this chunk's state into cumulative state
      int cs_offset = (bh * num_chunks + c) * state_size;
      for (int i = 0; i < d_k; i++) {
        cum_state[i] += to_float(chunk_states[cs_offset + i * d_v + j]);
      }
    }

    // Write final state
    for (int i = 0; i < d_k; i++) {
      final_state[bh * state_size + i * d_v + j] = from_float<T>(cum_state[i]);
    }
  }
}

}  // namespace

size_t GetLinearAttentionChunkParallelWorkspaceSize(
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    size_t element_size) {
  int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
  // chunk_states: (B, H, num_chunks, d_k, d_v)
  return static_cast<size_t>(batch_size) * num_heads * num_chunks * d_k * d_v * element_size;
}

template <typename T>
Status LaunchLinearAttentionChunkParallelKernel(
    cudaStream_t stream,
    LinearAttentionUpdateRule update_rule,
    const T* query,
    const T* key,
    const T* value,
    const T* initial_state,
    const T* decay,
    const T* beta,
    T* output,
    T* final_state,
    void* workspace,
    size_t workspace_size,
    float scale,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_k,
    int d_v,
    int chunk_size,
    bool decay_broadcasted) {
  int num_chunks = (seq_len + chunk_size - 1) / chunk_size;
  int BH = batch_size * num_heads;
  T* chunk_states = reinterpret_cast<T*>(workspace);

  // Phase 1: Intra-chunk computation
  // Grid: (BH, num_chunks), Block: min(256, d_v) threads
  dim3 grid1(BH, num_chunks);
  int threads = min(256, d_v);

  switch (update_rule) {
    case LinearAttentionUpdateRule::kLinear:
      IntraChunkKernel<T, 0><<<grid1, threads, 0, stream>>>(
          query, key, value, decay, beta, output, chunk_states,
          seq_len, d_k, d_v, chunk_size, num_chunks, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kGated:
      IntraChunkKernel<T, 1><<<grid1, threads, 0, stream>>>(
          query, key, value, decay, beta, output, chunk_states,
          seq_len, d_k, d_v, chunk_size, num_chunks, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kDelta:
      IntraChunkKernel<T, 2><<<grid1, threads, 0, stream>>>(
          query, key, value, decay, beta, output, chunk_states,
          seq_len, d_k, d_v, chunk_size, num_chunks, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kGatedDelta:
      IntraChunkKernel<T, 3><<<grid1, threads, 0, stream>>>(
          query, key, value, decay, beta, output, chunk_states,
          seq_len, d_k, d_v, chunk_size, num_chunks, decay_broadcasted);
      break;
  }

  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  // Phase 2: Inter-chunk state propagation and output correction
  // Grid: (BH), Block: min(256, d_v) threads
  InterChunkKernel<T><<<BH, threads, 0, stream>>>(
      query, initial_state, chunk_states, output, final_state,
      scale, seq_len, d_k, d_v, chunk_size, num_chunks);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionChunkParallelKernel<float>(
    cudaStream_t, LinearAttentionUpdateRule,
    const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    void*, size_t, float, int, int, int, int, int, int, bool);

template Status LaunchLinearAttentionChunkParallelKernel<half>(
    cudaStream_t, LinearAttentionUpdateRule,
    const half*, const half*, const half*, const half*,
    const half*, const half*, half*, half*,
    void*, size_t, float, int, int, int, int, int, int, bool);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionChunkParallelKernel<nv_bfloat16>(
    cudaStream_t, LinearAttentionUpdateRule,
    const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*,
    const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*, nv_bfloat16*,
    void*, size_t, float, int, int, int, int, int, int, bool);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
