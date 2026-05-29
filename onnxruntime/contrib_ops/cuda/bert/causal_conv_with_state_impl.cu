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
// State is stored in type T to match the op schema convention.

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
    const T* __restrict__ input,       // [B, C, 1]
    const T* __restrict__ weight,      // [C, 1, K]
    const T* __restrict__ bias,        // [C] or nullptr
    const T* __restrict__ past_state,  // [B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, 1]
    T* __restrict__ present_state,     // [B, C, K-1]
    int batch_channels,                // = batch_size * channels (actual element count)
    int channels,
    int kernel_size,
    bool apply_silu) {
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) return;
  const int b = bc / channels;
  const int c = bc % channels;

  const int pad = kernel_size - 1;

  // Cache input value in register — avoids redundant global reads
  const float input_val = to_float(input[(int64_t)b * channels + c]);

  // Cache past_state base pointer for this (b, c)
  const T* ps_in = (past_state != nullptr)
                       ? past_state + (int64_t)b * channels * pad + (int64_t)c * pad
                       : nullptr;

  // Load weight for this channel: [K] values
  // weight layout: [C, 1, K], so channel c starts at c * K
  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;

  // Convolution window: [past_state[0..K-2], input[0]]
  for (int k = 0; k < pad; ++k) {
    float wk = to_float(weight[c * kernel_size + k]);
    float xk = (ps_in != nullptr) ? to_float(ps_in[k]) : 0.0f;
    sum += wk * xk;
  }
  // Last element of window is current input
  sum += to_float(weight[c * kernel_size + pad]) * input_val;

  if (apply_silu) {
    sum = silu_fn(sum);
  }
  output[(int64_t)b * channels + c] = from_float<T>(sum);

  // Update present_state: shift left by 1, append input
  T* ps_out = present_state + (int64_t)b * channels * pad + (int64_t)c * pad;
  for (int k = 0; k < pad - 1; ++k) {
    ps_out[k] = (ps_in != nullptr) ? ps_in[k + 1] : from_float<T>(0.0f);
  }
  if (pad > 0) {
    ps_out[pad - 1] = from_float<T>(input_val);
  }
}

template <typename T, int K>
__global__ void CausalConvDecodeKernelFixedK(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const T* __restrict__ past_state,
    T* __restrict__ output,
    T* __restrict__ present_state,
    int batch_channels,
    int channels,
    bool apply_silu) {
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) return;

  const int b = bc / channels;
  const int c = bc % channels;
  constexpr int pad = K - 1;

  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
  const T* w = weight + static_cast<int64_t>(c) * K;
  const T* ps_in = (past_state != nullptr)
                       ? past_state + static_cast<int64_t>(b) * channels * pad + static_cast<int64_t>(c) * pad
                       : nullptr;

  if (ps_in != nullptr) {
#pragma unroll
    for (int k = 0; k < pad; ++k) {
      sum += to_float(w[k]) * to_float(ps_in[k]);
    }
  }
  sum += to_float(w[pad]) * to_float(input[static_cast<int64_t>(b) * channels + c]);

  if (apply_silu) {
    sum = silu_fn(sum);
  }
  output[static_cast<int64_t>(b) * channels + c] = from_float<T>(sum);

  T* ps_out = present_state + static_cast<int64_t>(b) * channels * pad + static_cast<int64_t>(c) * pad;
  if constexpr (pad > 0) {
#pragma unroll
    for (int k = 0; k < pad - 1; ++k) {
      ps_out[k] = (ps_in != nullptr) ? ps_in[k + 1] : from_float<T>(0.0f);
    }
    ps_out[pad - 1] = input[static_cast<int64_t>(b) * channels + c];
  }
}

// =============================================================================
// Prefill kernel: L>1, one thread per output position within a (batch, channel)
// Grid:  (batch_size, channels, 1)
// Block: (min(L, max_threads), 1, 1)
// Shared memory: padded input buffer [K-1 + L] floats + weight [K] floats
// =============================================================================
template <typename T>
__global__ void CausalConvPrefillKernel(
    const T* __restrict__ input,       // [B, C, L]
    const T* __restrict__ weight,      // [C, 1, K]
    const T* __restrict__ bias,        // [C] or nullptr
    const T* __restrict__ past_state,  // [B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, L]
    T* __restrict__ present_state,     // [B, C, K-1]
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
      s_padded[i] = to_float(past_state[(int64_t)b * channels * pad + (int64_t)c * pad + i]);
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
  T* ps = present_state + (int64_t)b * channels * pad + (int64_t)c * pad;
  for (int i = tid; i < pad; i += blockDim.x) {
    ps[i] = from_float<T>(s_padded[padded_len - pad + i]);
  }
}

// =============================================================================
// Batched prefill kernel: processes CHANNELS_PER_BLOCK channels per block
// to improve occupancy when per-channel work is small (short sequences).
//
// Grid:  (batch_size, ceil(channels / CPB), 1)
// Block: (threads, 1, 1) — threads are split across CPB channels
//
// Each channel gets (blockDim.x / CPB) threads. Weight is loaded into
// registers (small K), input+state goes through shared memory.
// =============================================================================
template <typename T, int CPB>
__global__ void CausalConvPrefillKernelBatched(
    const T* __restrict__ input,       // [B, C, L]
    const T* __restrict__ weight,      // [C, 1, K]
    const T* __restrict__ bias,        // [C] or nullptr
    const T* __restrict__ past_state,  // [B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, L]
    T* __restrict__ present_state,     // [B, C, K-1]
    int seq_len,
    int channels,
    int kernel_size,
    bool apply_silu) {
  const int b = blockIdx.x;
  const int c_base = blockIdx.y * CPB;
  const int tid = threadIdx.x;

  const int pad = kernel_size - 1;
  const int padded_len = pad + seq_len;

  // Which channel within this block's CPB group does this thread serve?
  const int threads_per_channel = blockDim.x / CPB;
  const int local_ch = tid / threads_per_channel;   // 0..CPB-1
  const int local_tid = tid % threads_per_channel;  // thread index within channel
  const int c = c_base + local_ch;

  // Shared memory: CPB * (padded_len + kernel_size) floats
  extern __shared__ float smem[];
  const int smem_per_ch = padded_len + kernel_size;
  float* s_padded = smem + local_ch * smem_per_ch;
  float* s_weight = s_padded + padded_len;

  if (c < channels) {
    // Load past state
    for (int i = local_tid; i < pad; i += threads_per_channel) {
      if (past_state != nullptr) {
        s_padded[i] = to_float(past_state[(int64_t)b * channels * pad + (int64_t)c * pad + i]);
      } else {
        s_padded[i] = 0.0f;
      }
    }
    // Load input
    for (int i = local_tid; i < seq_len; i += threads_per_channel) {
      s_padded[pad + i] = to_float(input[((int64_t)b * channels + c) * seq_len + i]);
    }
    // Load weight
    for (int i = local_tid; i < kernel_size; i += threads_per_channel) {
      s_weight[i] = to_float(weight[(int64_t)c * kernel_size + i]);
    }
  }
  __syncthreads();

  if (c < channels) {
    float bias_val = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
    for (int l = local_tid; l < seq_len; l += threads_per_channel) {
      float sum = bias_val;
      for (int k = 0; k < kernel_size; ++k) {
        sum += s_weight[k] * s_padded[l + k];
      }
      if (apply_silu) {
        sum = silu_fn(sum);
      }
      output[((int64_t)b * channels + c) * seq_len + l] = from_float<T>(sum);
    }
  }

  // Unconditional barrier — s_padded is read-only after the cooperative load,
  // so this is safe even when c >= channels.  Hoisted out of the conditional
  // to avoid divergent __syncthreads() (undefined behavior in CUDA).
  __syncthreads();

  if (c < channels) {
    // Save present state
    T* ps = present_state + (int64_t)b * channels * pad + (int64_t)c * pad;
    for (int i = local_tid; i < pad; i += threads_per_channel) {
      ps[i] = from_float<T>(s_padded[padded_len - pad + i]);
    }
  }
}

}  // anonymous namespace

template <typename T>
Status LaunchCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* bias,
    const T* past_state,
    T* output,
    T* present_state,
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
    switch (kernel_size) {
      case 2:
        CausalConvDecodeKernelFixedK<T, 2><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu);
        break;
      case 3:
        CausalConvDecodeKernelFixedK<T, 3><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu);
        break;
      case 4:
        CausalConvDecodeKernelFixedK<T, 4><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu);
        break;
      case 5:
        CausalConvDecodeKernelFixedK<T, 5><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu);
        break;
      default:
        CausalConvDecodeKernel<T><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, kernel_size, apply_silu);
        break;
    }
  } else {
    // Prefill path: choose between batched (short seq) or single-channel (long seq) kernel
    int pad = kernel_size - 1;

    // For short sequences, batch multiple channels per block to improve occupancy.
    // CPB=4: each block handles 4 channels, reducing block count by 4x.
    // Threshold: use batched when seq_len <= 128 (small per-channel work).
    constexpr int CPB = 4;
    if (seq_len <= 128 && channels >= CPB) {
      int channel_blocks = (channels + CPB - 1) / CPB;
      const dim3 grid(batch_size, channel_blocks, 1);
      // Each channel gets threads/CPB threads
      int threads_per_ch = std::min(seq_len, max_threads_per_block / CPB);
      threads_per_ch = ((threads_per_ch + 31) / 32) * 32;
      if (threads_per_ch < 32) threads_per_ch = 32;
      int total_threads = threads_per_ch * CPB;
      if (total_threads > max_threads_per_block) {
        total_threads = (max_threads_per_block / CPB) * CPB;  // round down to multiple of CPB
      }
      const dim3 block(total_threads, 1, 1);
      size_t smem_size = static_cast<size_t>(CPB) * (static_cast<size_t>(pad + seq_len) + kernel_size) * sizeof(float);

      // Request extended shared memory if needed (default limit is 48 KB)
      if (smem_size > 48 * 1024) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            CausalConvPrefillKernelBatched<T, CPB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
        if (attr_err != cudaSuccess) {
          return CUDA_CALL(attr_err);
        }
      }

      CausalConvPrefillKernelBatched<T, CPB><<<grid, block, smem_size, stream>>>(
          input, weight, bias, past_state, output, present_state,
          seq_len, channels, kernel_size, apply_silu);
    } else {
      // Original single-channel-per-block path for long sequences
      const dim3 grid(batch_size, channels, 1);
      int threads = std::min(seq_len, max_threads_per_block);
      threads = ((threads + 31) / 32) * 32;  // round to warp
      if (threads > max_threads_per_block) threads = max_threads_per_block;
      const dim3 block(threads, 1, 1);

      size_t smem_size = (static_cast<size_t>(pad + seq_len) + kernel_size) * sizeof(float);

      // Request extended shared memory if needed (default limit is 48 KB)
      if (smem_size > 48 * 1024) {
        cudaError_t attr_err = cudaFuncSetAttribute(
            CausalConvPrefillKernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
        if (attr_err != cudaSuccess) {
          return CUDA_CALL(attr_err);
        }
      }

      CausalConvPrefillKernel<T><<<grid, block, smem_size, stream>>>(
          input, weight, bias, past_state, output, present_state,
          seq_len, channels, kernel_size, apply_silu);
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchCausalConvWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int, bool, int);

template Status LaunchCausalConvWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    half*, half*, int, int, int, int, bool, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchCausalConvWithStateKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, bool, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
