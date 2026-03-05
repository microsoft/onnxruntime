// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/linear_attention_recurrent_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Convert type to float for computation
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

// Convert float to output type
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
// Fused Linear Attention Recurrent Kernel
//
// Each thread block handles one (batch, head) pair.
// The state matrix S[d_k][d_v] is kept in registers/shared memory.
//
// Thread layout: blockDim.x threads iterate over d_v columns.
// Each thread handles ceil(d_v / blockDim.x) columns of the state.
// The d_k rows are iterated sequentially within each thread.
//
// For typical dimensions (d_k=d_v=128 or 256), this keeps the full
// state in fast memory, avoiding global memory round-trips.
// ============================================================================

template <typename T, int UPDATE_RULE>
__global__ void LinearAttentionRecurrentKernel(
    const T* __restrict__ query,       // (B, H, 1, d_k)
    const T* __restrict__ key,         // (B, H, 1, d_k)
    const T* __restrict__ value,       // (B, H, 1, d_v)
    const T* __restrict__ past_state,  // (B, H, d_k, d_v)
    const T* __restrict__ decay,       // (B, H, 1, d_k) or (B,H,1,1) or nullptr
    const T* __restrict__ beta,        // (B, H, 1, 1) or nullptr
    T* __restrict__ output,            // (B, H, 1, d_v)
    T* __restrict__ present_state,     // (B, H, d_k, d_v)
    float scale,
    int d_k,
    int d_v,
    bool decay_broadcasted) {
  // Each block handles one (batch, head) pair
  const int bh = blockIdx.x;  // batch * num_heads + head
  const int tid = threadIdx.x;

  // Pointers for this (batch, head)
  const int qkv_offset = bh * d_k;           // offset into (B*H, 1, d_k)
  const int v_offset = bh * d_v;             // offset into (B*H, 1, d_v)
  const int state_offset = bh * d_k * d_v;  // offset into (B*H, d_k, d_v)

  // Load q, k vectors into shared memory
  extern __shared__ float shared_mem[];
  float* s_q = shared_mem;            // d_k floats
  float* s_k = s_q + d_k;            // d_k floats
  float* s_v = s_k + d_k;            // d_v floats
  float* s_decay = s_v + d_v;        // d_k floats (for gated modes)
  float* s_retrieved = s_decay + d_k; // d_v floats (for delta modes)

  // Load q, k into shared memory
  for (int i = tid; i < d_k; i += blockDim.x) {
    s_q[i] = to_float(query[qkv_offset + i]);
    s_k[i] = to_float(key[qkv_offset + i]);
  }
  // Load v into shared memory
  for (int i = tid; i < d_v; i += blockDim.x) {
    s_v[i] = to_float(value[v_offset + i]);
  }

  // Load decay if needed
  float beta_val = 0.0f;
  if constexpr (UPDATE_RULE == 1 || UPDATE_RULE == 3) {  // gated or gated_delta
    if (decay_broadcasted) {
      for (int i = tid; i < d_k; i += blockDim.x) {
        s_decay[i] = expf(to_float(decay[qkv_offset + i]));
      }
    } else {
      // Scalar decay — broadcast to all d_k
      float decay_scalar = expf(to_float(decay[bh]));
      for (int i = tid; i < d_k; i += blockDim.x) {
        s_decay[i] = decay_scalar;
      }
    }
  }

  // Load beta if needed
  if constexpr (UPDATE_RULE == 2 || UPDATE_RULE == 3) {  // delta or gated_delta
    if (tid == 0) {
      beta_val = to_float(beta[bh]);
    }
  }

  __syncthreads();

  // Broadcast beta
  if constexpr (UPDATE_RULE == 2 || UPDATE_RULE == 3) {
    beta_val = __shfl_sync(0xffffffff, beta_val, 0);
    // For blocks > 32 threads, use shared memory
    if (tid == 0) s_retrieved[0] = beta_val;
    __syncthreads();
    beta_val = s_retrieved[0];
    __syncthreads();
  }

  // For delta modes: compute retrieved = S^T k (dot product of state columns with k)
  // retrieved[j] = sum_i(state[i][j] * k[i])
  if constexpr (UPDATE_RULE == 2 || UPDATE_RULE == 3) {
    // Initialize retrieved to 0
    for (int j = tid; j < d_v; j += blockDim.x) {
      s_retrieved[j] = 0.0f;
    }
    __syncthreads();

    // Accumulate state^T * k
    for (int j = tid; j < d_v; j += blockDim.x) {
      float sum = 0.0f;
      for (int i = 0; i < d_k; i++) {
        float state_val = to_float(past_state[state_offset + i * d_v + j]);
        float gate = 1.0f;
        if constexpr (UPDATE_RULE == 3) {  // gated_delta: multiply state by decay first
          gate = s_decay[i];
        }
        sum += gate * state_val * s_k[i];
      }
      s_retrieved[j] = sum;
    }
    __syncthreads();
  }

  // Now update state and compute output
  // Each thread handles a subset of d_v columns
  // output[j] = scale * sum_i(q[i] * state_new[i][j])
  for (int j = tid; j < d_v; j += blockDim.x) {
    float out_val = 0.0f;

    for (int i = 0; i < d_k; i++) {
      float state_val = to_float(past_state[state_offset + i * d_v + j]);

      float new_state;
      if constexpr (UPDATE_RULE == 0) {
        // linear: S_t = S_{t-1} + k_t ⊗ v_t
        new_state = state_val + s_k[i] * s_v[j];
      } else if constexpr (UPDATE_RULE == 1) {
        // gated: S_t = exp(g_t) · S_{t-1} + k_t ⊗ v_t
        new_state = s_decay[i] * state_val + s_k[i] * s_v[j];
      } else if constexpr (UPDATE_RULE == 2) {
        // delta: S_t = S_{t-1} + β_t · k_t ⊗ (v_t − S_{t-1}^T k_t)
        float delta = s_v[j] - s_retrieved[j];
        new_state = state_val + beta_val * s_k[i] * delta;
      } else {
        // gated_delta: S_t = exp(g) · S_{t-1} + β · k ⊗ (v − exp(g) · S_{t-1}^T k)
        float delta = s_v[j] - s_retrieved[j];
        new_state = s_decay[i] * state_val + beta_val * s_k[i] * delta;
      }

      // Write updated state
      present_state[state_offset + i * d_v + j] = from_float<T>(new_state);

      // Accumulate output: o_t = scale · q^T S_t
      out_val += s_q[i] * new_state;
    }

    output[v_offset + j] = from_float<T>(out_val * scale);
  }
}

}  // namespace

template <typename T>
Status LaunchLinearAttentionRecurrentKernel(
    cudaStream_t stream,
    LinearAttentionUpdateRule update_rule,
    const T* query,
    const T* key,
    const T* value,
    const T* past_state,
    const T* decay,
    const T* beta,
    T* output,
    T* present_state,
    float scale,
    int batch_size,
    int num_heads,
    int d_k,
    int d_v,
    bool decay_broadcasted) {
  int num_blocks = batch_size * num_heads;
  int threads_per_block = min(256, d_v);  // One thread per d_v column, capped at 256

  // Shared memory: q(d_k) + k(d_k) + v(d_v) + decay(d_k) + retrieved(d_v)
  size_t shared_mem_size = (d_k * 3 + d_v * 2) * sizeof(float);

  // Dispatch based on update rule (compile-time specialization)
  switch (update_rule) {
    case LinearAttentionUpdateRule::kLinear:
      LinearAttentionRecurrentKernel<T, 0><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          query, key, value, past_state, decay, beta, output, present_state,
          scale, d_k, d_v, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kGated:
      LinearAttentionRecurrentKernel<T, 1><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          query, key, value, past_state, decay, beta, output, present_state,
          scale, d_k, d_v, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kDelta:
      LinearAttentionRecurrentKernel<T, 2><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          query, key, value, past_state, decay, beta, output, present_state,
          scale, d_k, d_v, decay_broadcasted);
      break;
    case LinearAttentionUpdateRule::kGatedDelta:
      LinearAttentionRecurrentKernel<T, 3><<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
          query, key, value, past_state, decay, beta, output, present_state,
          scale, d_k, d_v, decay_broadcasted);
      break;
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchLinearAttentionRecurrentKernel<float>(
    cudaStream_t, LinearAttentionUpdateRule,
    const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    float, int, int, int, int, bool);

template Status LaunchLinearAttentionRecurrentKernel<half>(
    cudaStream_t, LinearAttentionUpdateRule,
    const half*, const half*, const half*, const half*,
    const half*, const half*, half*, half*,
    float, int, int, int, int, bool);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchLinearAttentionRecurrentKernel<nv_bfloat16>(
    cudaStream_t, LinearAttentionUpdateRule,
    const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*,
    const nv_bfloat16*, const nv_bfloat16*, nv_bfloat16*, nv_bfloat16*,
    float, int, int, int, int, bool);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
