// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused causal depthwise conv1d CUDA kernel with stateful carry and optional SiLU activation.
//
// Design: One thread block per (batch, channel). Two execution paths:
//
// 1. Decode (L=1):  The convolution window is [past_state(K-1), input(1)].
//    Load K values into registers, compute a single dot product, shift state.
//    One thread block does the entire operation — zero shared memory needed.
//
// 2. Prefill (L>1): Load past_state + input into shared memory as a padded buffer,
//    then each thread computes one output position's convolution.
//
// State is stored in fp32 for cross-call stability (matching the CPU kernel).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include "contrib_ops/cuda/bert/causal_conv_with_state_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

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

__device__ __forceinline__ float silu_fn(float x) {
  return x / (1.0f + expf(-x));
}

// =============================================================================
// Decode kernel: L=1, one dot product per (batch, channel)
// Grid:  (batch_size * channels, 1, 1)
// Block: (1, 1, 1) — one thread per (batch, channel)
// No shared memory needed.
// =============================================================================
template <typename T>
__global__ void CausalConvDecodeKernel(
    const T* __restrict__ input,        // [B, C, 1]
    const T* __restrict__ weight,       // [C, 1, K]
    const T* __restrict__ bias,         // [C] or nullptr
    const float* __restrict__ past_state,  // [B, C, K-1] or nullptr
    T* __restrict__ output,             // [B, C, 1]
    float* __restrict__ present_state,  // [B, C, K-1]
    int channels,
    int kernel_size,
    bool apply_silu) {
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  const int B_times_C = gridDim.x * blockDim.x;  // total (batch, channel) count
  // Guard check computed from actual dims passed
  const int b = bc / channels;
  const int c = bc % channels;

  // Bounds check
  if (bc >= B_times_C) return;

  const int pad = kernel_size - 1;
  const float* w = nullptr;

  // Load weight for this channel: [K] values
  // weight layout: [C, 1, K], so channel c starts at c * K
  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;

  // Convolution window: [past_state[0..K-2], input[0]]
  for (int k = 0; k < pad; ++k) {
    float wk = to_float(weight[c * kernel_size + k]);
    float xk = (past_state != nullptr)
                   ? past_state[(int64_t)b * channels * pad + (int64_t)c * pad + k]
                   : 0.0f;
    sum += wk * xk;
  }
  // Last element of window is current input
  sum += to_float(weight[c * kernel_size + pad]) *
         to_float(input[(int64_t)b * channels + c]);

  if (apply_silu) {
    sum = silu_fn(sum);
  }
  output[(int64_t)b * channels + c] = from_float<T>(sum);

  // Update present_state: shift left by 1, append input
  // present_state[b, c, 0..K-3] = past_state[b, c, 1..K-2]
  // present_state[b, c, K-2]    = input[b, c, 0]
  float* ps = present_state + (int64_t)b * channels * pad + (int64_t)c * pad;
  for (int k = 0; k < pad - 1; ++k) {
    ps[k] = (past_state != nullptr)
                ? past_state[(int64_t)b * channels * pad + (int64_t)c * pad + k + 1]
                : 0.0f;
  }
  if (pad > 0) {
    ps[pad - 1] = to_float(input[(int64_t)b * channels + c]);
  }
}

// =============================================================================
// Prefill kernel: L>1, one thread per output position within a (batch, channel)
// Grid:  (batch_size, channels, 1)
// Block: (min(L, max_threads), 1, 1)
// Shared memory: padded input buffer [K-1 + L] floats
// =============================================================================
template <typename T>
__global__ void CausalConvPrefillKernel(
    const T* __restrict__ input,        // [B, C, L]
    const T* __restrict__ weight,       // [C, 1, K]
    const T* __restrict__ bias,         // [C] or nullptr
    const float* __restrict__ past_state,  // [B, C, K-1] or nullptr
    T* __restrict__ output,             // [B, C, L]
    float* __restrict__ present_state,  // [B, C, K-1]
    int seq_len,
    int channels,
    int kernel_size,
    bool apply_silu) {
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  const int tid = threadIdx.x;

  const int pad = kernel_size - 1;
  const int padded_len = pad + seq_len;

  // Shared memory: padded input [pad + L] floats + weight [K] floats
  extern __shared__ float smem[];
  float* s_padded = smem;
  float* s_weight = smem + padded_len;

  // Cooperatively load padded input into shared memory
  // Past state portion: [0..pad-1]
  for (int i = tid; i < pad; i += blockDim.x) {
    if (past_state != nullptr) {
      s_padded[i] = past_state[(int64_t)b * channels * pad + (int64_t)c * pad + i];
    } else {
      s_padded[i] = 0.0f;
    }
  }
  // Current input portion: [pad..pad+L-1]
  for (int i = tid; i < seq_len; i += blockDim.x) {
    s_padded[pad + i] = to_float(input[((int64_t)b * channels + c) * seq_len + i]);
  }
  // Load weight into shared memory
  for (int i = tid; i < kernel_size; i += blockDim.x) {
    s_weight[i] = to_float(weight[(int64_t)c * kernel_size + i]);
  }
  __syncthreads();

  // Each thread computes one output position
  float bias_val = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
  for (int l = tid; l < seq_len; l += blockDim.x) {
    float sum = bias_val;
    for (int k = 0; k < kernel_size; ++k) {
      sum += s_weight[k] * s_padded[l + k];
    }
    if (apply_silu) {
      sum = silu_fn(sum);
    }
    output[((int64_t)b * channels + c) * seq_len + l] = from_float<T>(sum);
  }

  // Save present_state: last K-1 elements of padded input
  __syncthreads();
  float* ps = present_state + (int64_t)b * channels * pad + (int64_t)c * pad;
  for (int i = tid; i < pad; i += blockDim.x) {
    ps[i] = s_padded[padded_len - pad + i];
  }
}

}  // anonymous namespace

template <typename T>
Status LaunchCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* bias,
    const float* past_state,
    T* output,
    float* present_state,
    int batch_size,
    int channels,
    int seq_len,
    int kernel_size,
    bool apply_silu,
    int max_threads_per_block) {
  if (seq_len == 1) {
    // Decode fast-path: one thread per (batch, channel)
    int total = batch_size * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    CausalConvDecodeKernel<T><<<blocks, threads, 0, stream>>>(
        input, weight, bias, past_state, output, present_state,
        channels, kernel_size, apply_silu);
  } else {
    // Prefill: one block per (batch, channel), threads handle output positions
    const dim3 grid(batch_size, channels, 1);
    int threads = std::min(seq_len, max_threads_per_block);
    threads = ((threads + 31) / 32) * 32;  // round to warp
    if (threads > max_threads_per_block) threads = max_threads_per_block;
    const dim3 block(threads, 1, 1);

    int pad = kernel_size - 1;
    size_t smem_size = (static_cast<size_t>(pad + seq_len) + kernel_size) * sizeof(float);

    CausalConvPrefillKernel<T><<<grid, block, smem_size, stream>>>(
        input, weight, bias, past_state, output, present_state,
        seq_len, channels, kernel_size, apply_silu);
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchCausalConvWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int, bool, int);

template Status LaunchCausalConvWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const float*,
    half*, float*, int, int, int, int, bool, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchCausalConvWithStateKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*,
    __nv_bfloat16*, float*, int, int, int, int, bool, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
