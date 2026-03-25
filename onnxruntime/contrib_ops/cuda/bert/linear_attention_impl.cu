// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/linear_attention_impl.h"

#include <cuda_fp16.h>
#include <cmath>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Helper to convert type to float for arithmetic
template <typename T>
__device__ __forceinline__ float ToFloat(T val);

template <>
__device__ __forceinline__ float ToFloat(float val) { return val; }

template <>
__device__ __forceinline__ float ToFloat(half val) { return __half2float(val); }

template <typename T>
__device__ __forceinline__ T FromFloat(float val);

template <>
__device__ __forceinline__ float FromFloat(float val) { return val; }

template <>
__device__ __forceinline__ half FromFloat(float val) { return __float2half(val); }

// Kernel for recurrent step (T=1) -- one block per (batch, head)
// Each thread handles one element of the state update and output computation.
// For T>1 we call this iteratively from the host side.
template <typename T>
__global__ void LinearAttentionRecurrentKernel(
    const T* __restrict__ query,       // (B, H, 1, d_k)
    const T* __restrict__ key,         // (B, H, 1, d_k)
    const T* __restrict__ value,       // (B, H, 1, d_v)
    const T* __restrict__ decay,       // (B, H, 1, decay_key_dim) or nullptr
    const T* __restrict__ beta,        // (B, H, 1, 1) or nullptr
    T* __restrict__ state,             // (B, H, d_k, d_v) - in/out
    T* __restrict__ output,            // (B, H, 1, d_v)
    int key_dim,
    int value_dim,
    int decay_key_dim,                 // 0, 1, or key_dim
    float scale,
    int update_rule,                   // 0=linear, 1=gated, 2=delta, 3=gated_delta
    int q_seq_offset,                  // offset into q/k/v for current timestep
    int state_size) {                  // key_dim * value_dim

  // Each block handles one (batch, head) pair
  int bh = blockIdx.x;

  // State for this (b, h): S[d_k, d_v]
  T* S = state + bh * state_size;

  // Current token q, k, v
  const T* q_t = query + bh * q_seq_offset * key_dim;
  const T* k_t = key + bh * q_seq_offset * key_dim;
  const T* v_t = value + bh * q_seq_offset * value_dim;
  T* o_t = output + bh * q_seq_offset * value_dim;

  bool is_gated = (update_rule == 1 || update_rule == 3);
  bool is_delta = (update_rule == 2 || update_rule == 3);

  // Thread index maps to (dk, dv) in the state matrix
  // We use a grid-stride loop for large state matrices
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  // Step 1: Load decay factors into shared memory
  extern __shared__ float shared_mem[];
  // Layout: [key_dim floats for decay] [key_dim floats for k] [value_dim floats for v]
  //         [value_dim floats for retrieved] [1 float for beta]
  float* s_g = shared_mem;                          // key_dim
  float* s_k = s_g + key_dim;                       // key_dim
  float* s_v = s_k + key_dim;                       // value_dim
  float* s_retrieved = s_v + value_dim;              // value_dim (for delta mode)
  float* s_beta = s_retrieved + value_dim;           // 1

  // Load k, v into shared memory
  for (int i = tid; i < key_dim; i += block_size) {
    s_k[i] = ToFloat(k_t[i]);
  }
  for (int i = tid; i < value_dim; i += block_size) {
    s_v[i] = ToFloat(v_t[i]);
  }

  // Load decay
  if (is_gated && decay != nullptr) {
    const T* decay_t = decay + bh * q_seq_offset * decay_key_dim;
    for (int i = tid; i < key_dim; i += block_size) {
      int decay_idx = (decay_key_dim == 1) ? 0 : i;
      s_g[i] = expf(ToFloat(decay_t[decay_idx]));
    }
  } else {
    for (int i = tid; i < key_dim; i += block_size) {
      s_g[i] = 1.0f;
    }
  }

  // Load beta
  if (is_delta && beta != nullptr) {
    if (tid == 0) {
      const T* beta_t = beta + bh * q_seq_offset;
      s_beta[0] = ToFloat(*beta_t);
    }
  }

  // Initialize retrieved to zero
  if (is_delta) {
    for (int i = tid; i < value_dim; i += block_size) {
      s_retrieved[i] = 0.0f;
    }
  }
  __syncthreads();

  // Step 2: Apply decay to state
  if (is_gated) {
    for (int idx = tid; idx < key_dim * value_dim; idx += block_size) {
      int dk = idx / value_dim;
      S[idx] = FromFloat<T>(ToFloat(S[idx]) * s_g[dk]);
    }
    __syncthreads();
  }

  // Step 3 (Delta): Compute retrieved = S^T k
  if (is_delta) {
    // Each thread accumulates partial sums for its assigned dv values
    for (int dv = tid; dv < value_dim; dv += block_size) {
      float sum = 0.0f;
      for (int dk = 0; dk < key_dim; ++dk) {
        sum += ToFloat(S[dk * value_dim + dv]) * s_k[dk];
      }
      s_retrieved[dv] = sum;
    }
    __syncthreads();
  }

  // Step 4: Update state
  if (is_delta) {
    float beta_val = s_beta[0];
    for (int idx = tid; idx < key_dim * value_dim; idx += block_size) {
      int dk = idx / value_dim;
      int dv = idx % value_dim;
      float delta = beta_val * (s_v[dv] - s_retrieved[dv]);
      S[idx] = FromFloat<T>(ToFloat(S[idx]) + s_k[dk] * delta);
    }
  } else {
    for (int idx = tid; idx < key_dim * value_dim; idx += block_size) {
      int dk = idx / value_dim;
      int dv = idx % value_dim;
      S[idx] = FromFloat<T>(ToFloat(S[idx]) + s_k[dk] * s_v[dv]);
    }
  }
  __syncthreads();

  // Step 5: Compute output = scale * q^T S
  // Load q into shared memory (reuse s_k since we're done with it and both are key_dim sized)
  float* s_q = s_k;  // Reuse s_k space for q
  for (int i = tid; i < key_dim; i += block_size) {
    s_q[i] = ToFloat(q_t[i]);
  }
  __syncthreads();

  for (int dv = tid; dv < value_dim; dv += block_size) {
    float sum = 0.0f;
    for (int dk = 0; dk < key_dim; ++dk) {
      sum += s_q[dk] * ToFloat(S[dk * value_dim + dv]);
    }
    o_t[dv] = FromFloat<T>(sum * scale);
  }
}

}  // namespace

template <typename T>
void LaunchLinearAttentionKernel(
    cudaStream_t stream,
    const T* query,
    const T* key,
    const T* value,
    const T* past_state,
    const T* decay,
    const T* beta,
    T* output,
    T* present_state,
    int batch_size,
    int num_heads,
    int seq_len,
    int key_dim,
    int value_dim,
    int decay_key_dim,
    float scale,
    LinearAttentionUpdateRuleCuda update_rule) {

  int BH = batch_size * num_heads;
  int state_size = key_dim * value_dim;

  // Copy past_state to present_state (or zero-initialize)
  if (past_state != nullptr) {
    cudaMemcpyAsync(present_state, past_state,
                    static_cast<size_t>(BH) * state_size * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
  } else {
    cudaMemsetAsync(present_state, 0,
                    static_cast<size_t>(BH) * state_size * sizeof(T), stream);
  }

  // Shared memory: key_dim (decay) + key_dim (k) + value_dim (v) + value_dim (retrieved) + 1 (beta)
  int shared_mem_size = static_cast<int>((2 * key_dim + 2 * value_dim + 1) * sizeof(float));

  // Block size: use up to 256 threads
  int block_size = 256;
  if (state_size < block_size) {
    block_size = (state_size + 31) / 32 * 32;  // round up to multiple of 32
    if (block_size < 32) block_size = 32;
  }

  int rule = static_cast<int>(update_rule);

  // Process each timestep sequentially.
  // Note: For multi-token sequences (T > 1), a chunk-parallel algorithm using
  // WY decomposition could provide better GPU utilization. The current sequential
  // approach is functionally correct for all sequence lengths.
  for (int t = 0; t < seq_len; ++t) {
    // Offset q/k/v/decay/beta pointers for timestep t
    const T* q_t = query + static_cast<int64_t>(t) * key_dim;
    const T* k_t = key + static_cast<int64_t>(t) * key_dim;
    const T* v_t = value + static_cast<int64_t>(t) * value_dim;
    T* o_t = output + static_cast<int64_t>(t) * value_dim;

    const T* decay_t = nullptr;
    const T* beta_t = nullptr;
    if (decay != nullptr) {
      decay_t = decay + static_cast<int64_t>(t) * decay_key_dim;
    }
    if (beta != nullptr) {
      beta_t = beta + static_cast<int64_t>(t);
    }

    // For the kernel, we set q_seq_offset=1 since we're processing one timestep
    // But we need to adjust the base pointers per (b,h) properly
    // The data layout is (B, H, T, d), so for timestep t:
    // q[b,h,t,:] is at offset ((b*H+h)*T + t) * d_k

    // We'll launch a kernel that processes all (b,h) pairs for timestep t
    LinearAttentionRecurrentKernel<T><<<BH, block_size, shared_mem_size, stream>>>(
        q_t,       // effectively shifted by t * key_dim for one element stride
        k_t,
        v_t,
        decay_t,
        beta_t,
        present_state,
        o_t,
        key_dim,
        value_dim,
        decay_key_dim,
        scale,
        rule,
        seq_len,      // q_seq_offset = seq_len (stride between heads in seq dim)
        state_size);
  }
}

// Explicit template instantiations
template void LaunchLinearAttentionKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    const float*, const float*, float*, float*,
    int, int, int, int, int, int, float, LinearAttentionUpdateRuleCuda);

template void LaunchLinearAttentionKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    const half*, const half*, half*, half*,
    int, int, int, int, int, int, float, LinearAttentionUpdateRuleCuda);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
