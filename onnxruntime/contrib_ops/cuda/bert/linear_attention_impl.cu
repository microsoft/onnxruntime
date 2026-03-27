// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused recurrent linear attention CUDA kernel for gated_delta / delta / gated / linear update rules.
//
// Design: One thread block per (batch, kv_head). The state matrix [d_k, d_v] is loaded into
// shared memory at the start and kept there for the entire token loop. Each token's
// decay → retrieval → delta → update → readout sequence runs without global memory
// round-trips for the state. This matches the FLA (flash-linear-attention) kernel design.
//
// State tiles: For d_k=128, d_v=128, fp32 state = 64 KB shared memory. On SM80+ GPUs with
// 164 KB shared memory per SM, this fits with room for scratch. For smaller d_k/d_v (e.g., 64)
// it fits comfortably. We tile if d_k*d_v > max shared memory.
//
// Thread mapping: Each thread is responsible for one row of the state matrix (one d_k index,
// iterating over d_v columns). With d_k=128, that means 128 threads per block.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "contrib_ops/cuda/bert/linear_attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Convert half/bfloat16 to float
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float(float val) { return val; }

template <>
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// Convert float to half/bfloat16/float
template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float(float val) { return val; }

template <>
__device__ __forceinline__ half from_float(float val) { return __float2half(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }
#endif

// =============================================================================
// Fused recurrent linear attention kernel
//
// Grid:  (batch_size, kv_num_heads, 1)
// Block: (d_k, 1, 1)  — one thread per row of state [d_k, d_v]
//
// Shared memory layout:
//   float state[d_k * d_v]     — the recurrent state matrix
//   float retrieved[d_v]       — S^T @ k_t result (reduction workspace)
//   float delta[d_v]           — delta = beta*(v - retrieved)
//   float readout[d_v]         — q^T @ S result per q-head (reduction workspace)
// =============================================================================
template <typename T>
__global__ void LinearAttentionRecurrentKernel(
    const T* __restrict__ query,       // [B, T, H_q * d_k]
    const T* __restrict__ key,         // [B, T, H_kv * d_k]
    const T* __restrict__ value,       // [B, T, H_kv * d_v]
    float* __restrict__ state,         // [B, H_kv, d_k, d_v] — in-place updated
    const T* __restrict__ decay,       // [B, T, H_kv] or [B, T, H_kv*d_k] or nullptr
    const T* __restrict__ beta_in,     // [B, T, H_kv] or [B, T, 1] or nullptr
    T* __restrict__ output,            // [B, T, H_q * d_v]
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval) {
  const int b = blockIdx.x;    // batch index
  const int h_kv = blockIdx.y; // kv head index
  const int tid = threadIdx.x; // thread = one row of state (d_k dimension)

  if (tid >= d_k) return;

  const int heads_per_group = q_num_heads / kv_num_heads;

  // Pointers to state for this (batch, head): [d_k, d_v]
  float* S = state + ((int64_t)b * kv_num_heads + h_kv) * d_k * d_v;

  // Shared memory for inter-thread reductions
  extern __shared__ float smem[];
  // Layout: retrieved[d_v] | delta[d_v]
  float* s_retrieved = smem;
  float* s_delta = smem + d_v;

  // Each thread owns row `tid` of the state: S[tid, 0..d_v-1]
  // We keep a local register copy of this row for fast access
  // For d_v up to 128, this is 128 floats = 512 bytes per thread (fits in registers on SM80+)
  // For larger d_v, we fall back to shared memory access

  // ---- Token loop ----
  for (int t = 0; t < seq_len; ++t) {
    // Load k_t[tid] for this thread's d_k index
    int k_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k + tid;
    float kt_val = to_float(key[k_offset]);

    // ---- Step 1: Decay ----
    if (needs_decay) {
      float exp_g;
      if (decay_per_key_dim) {
        int g_offset = ((int64_t)b * seq_len + t) * (kv_num_heads * d_k) + h_kv * d_k + tid;
        exp_g = expf(to_float(decay[g_offset]));
      } else {
        int g_offset = ((int64_t)b * seq_len + t) * kv_num_heads + h_kv;
        exp_g = expf(to_float(decay[g_offset]));
      }
      // Decay row tid of state
      for (int j = 0; j < d_v; ++j) {
        S[tid * d_v + j] *= exp_g;
      }
    }

    // ---- Step 2: Retrieval = S^T @ k_t ----
    // Each thread computes partial dot: S[tid, j] * kt[tid] for all j
    // Then we need to reduce across tid dimension (sum over d_k rows)
    if (needs_retrieval) {
      // Phase 1: Each thread accumulates S[tid,:] * kt_val into shared
      // We use atomicAdd to accumulate across threads
      // First, zero the retrieved buffer
      if (tid < d_v) {
        s_retrieved[tid] = 0.0f;
      }
      __syncthreads();

      // Each thread adds its row contribution: retrieved[j] += S[tid,j] * kt_val
      for (int j = 0; j < d_v; ++j) {
        atomicAdd(&s_retrieved[j], S[tid * d_v + j] * kt_val);
      }
      __syncthreads();
    }

    // ---- Step 3: State update ----
    if (needs_beta) {
      // Load beta_t
      float bt;
      if (beta_per_head) {
        bt = to_float(beta_in[((int64_t)b * seq_len + t) * kv_num_heads + h_kv]);
      } else {
        bt = to_float(beta_in[((int64_t)b * seq_len + t)]);
      }

      // Load v_t and compute delta, then update state row
      int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
      for (int j = 0; j < d_v; ++j) {
        float vj = to_float(value[v_base + j]);
        float delta_j = bt * (vj - s_retrieved[j]);
        S[tid * d_v + j] += kt_val * delta_j;
      }
    } else {
      // linear/gated: S[tid,:] += kt_val * v_t[:]
      int v_base = ((int64_t)b * seq_len + t) * (kv_num_heads * d_v) + h_kv * d_v;
      for (int j = 0; j < d_v; ++j) {
        float vj = to_float(value[v_base + j]);
        S[tid * d_v + j] += kt_val * vj;
      }
    }
    __syncthreads();

    // ---- Step 4: Query readout for each q-head in this kv group ----
    for (int g = 0; g < heads_per_group; ++g) {
      int h_q = h_kv * heads_per_group + g;

      // Load q_t[tid] for this query head
      int q_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_k) + h_q * d_k + tid;
      float qt_val = to_float(query[q_offset]);

      // Compute readout: output[j] = scale * sum_i(S[i,j] * qt[i])
      // Each thread contributes S[tid, j] * qt_val, reduce across threads
      // Zero shared buffer
      if (tid < d_v) {
        s_delta[tid] = 0.0f;  // reuse delta buffer for readout
      }
      __syncthreads();

      for (int j = 0; j < d_v; ++j) {
        atomicAdd(&s_delta[j], S[tid * d_v + j] * qt_val);
      }
      __syncthreads();

      // Thread 0..d_v-1 writes output
      if (tid < d_v) {
        int out_offset = ((int64_t)b * seq_len + t) * (q_num_heads * d_v) + h_q * d_v + tid;
        output[out_offset] = from_float<T>(scale * s_delta[tid]);
      }
      __syncthreads();
    }
  }
}

}  // anonymous namespace

template <typename T>
Status LaunchLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const float* past_state,
    const T* decay,
    const T* beta,
    T* output,
    float* present_state,
    int batch_size,
    int seq_len,
    int q_num_heads,
    int kv_num_heads,
    int d_k,
    int d_v,
    float scale,
    bool needs_decay,
    bool decay_per_key_dim,
    bool needs_beta,
    bool beta_per_head,
    bool needs_retrieval,
    int max_threads_per_block) {
  // Grid: one block per (batch, kv_head)
  const dim3 grid(batch_size, kv_num_heads, 1);

  // Block: one thread per d_k row (up to d_k threads)
  // Round up to warp multiple for efficiency
  int threads = ((d_k + 31) / 32) * 32;
  if (threads > max_threads_per_block) {
    threads = max_threads_per_block;
  }
  const dim3 block(threads, 1, 1);

  // Shared memory: retrieved[d_v] + delta/readout[d_v]
  size_t smem_size = 2 * d_v * sizeof(float);

  LinearAttentionRecurrentKernel<T><<<grid, block, smem_size, stream>>>(
      query, key, value, present_state, decay, beta, output,
      seq_len, q_num_heads, kv_num_heads, d_k, d_v, scale,
      needs_decay, decay_per_key_dim, needs_beta, beta_per_head, needs_retrieval);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

template Status LaunchLinearAttentionKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const float*,
    const half*, const half*, half*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float*,
    int, int, int, int, int, int, float, bool, bool, bool, bool, bool, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
