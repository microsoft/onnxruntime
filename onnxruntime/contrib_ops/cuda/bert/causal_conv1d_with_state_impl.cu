// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/bert/causal_conv1d_with_state_impl.h"

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

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// ============================================================================
// Fused Causal 1D Depthwise Convolution Kernel
//
// Each thread handles one (batch, channel) pair across all time steps.
// The convolution is causal (left-padded): output[t] depends only on
// input[t-K+1 .. t].
//
// During decode (T=1), the conv_state provides the history.
// During prefill (T>1), zero-padding or conv_state provides initial context.
//
// The carry state is always updated: present_state holds the last K-1
// input values for use in the next step.
// ============================================================================
template <typename T, int ACTIVATION>
__global__ void CausalConv1DWithStateKernel(
    const T* __restrict__ input,       // (B, D, T)
    const T* __restrict__ weight,      // (D, 1, K)
    const T* __restrict__ bias,        // (D) or nullptr
    const T* __restrict__ conv_state,  // (B, D, K-1) or nullptr
    T* __restrict__ output,            // (B, D, T)
    T* __restrict__ present_state,     // (B, D, K-1)
    int batch_size,
    int channels,
    int seq_len,
    int kernel_size) {
  // Each thread handles one (batch, channel) pair
  const int bd = blockIdx.x * blockDim.x + threadIdx.x;
  if (bd >= batch_size * channels) return;

  const int b = bd / channels;
  const int d = bd % channels;

  const int state_len = kernel_size - 1;

  // Weight for this channel: weight[d, 0, 0..K-1]
  // Load weights into registers
  float w[32];  // kernel_size <= 32 enforced by host validation
  for (int k = 0; k < kernel_size; k++) {
    w[k] = to_float(weight[d * kernel_size + k]);
  }

  float bias_val = 0.0f;
  if (bias != nullptr) {
    bias_val = to_float(bias[d]);
  }

  // Process each time step
  for (int t = 0; t < seq_len; t++) {
    float sum = bias_val;

    // Convolution: sum over kernel positions
    // output[b,d,t] = bias[d] + sum_{k=0}^{K-1} weight[d,0,k] * input_padded[b,d,t-K+1+k]
    for (int k = 0; k < kernel_size; k++) {
      int src_t = t - state_len + k;  // position in padded input

      float input_val;
      if (src_t >= 0) {
        // Read from input
        input_val = to_float(input[(b * channels + d) * seq_len + src_t]);
      } else {
        // Read from conv_state (padding region)
        int state_idx = state_len + src_t;  // maps -state_len...-1 to 0...state_len-1
        if (conv_state != nullptr && state_idx >= 0) {
          input_val = to_float(conv_state[(b * channels + d) * state_len + state_idx]);
        } else {
          input_val = 0.0f;
        }
      }

      sum += w[k] * input_val;
    }

    // Apply activation
    if constexpr (ACTIVATION == 1) {  // SiLU
      sum = silu(sum);
    }

    output[(b * channels + d) * seq_len + t] = from_float<T>(sum);
  }

  // Update carry state: last K-1 input values
  for (int k = 0; k < state_len; k++) {
    int src_t = seq_len - state_len + k;
    float val;
    if (src_t >= 0) {
      val = to_float(input[(b * channels + d) * seq_len + src_t]);
    } else {
      // Still in the padding region — carry from old state
      int state_idx = state_len + src_t;  // into old conv_state
      if (conv_state != nullptr && state_idx >= 0) {
        val = to_float(conv_state[(b * channels + d) * state_len + state_idx]);
      } else {
        val = 0.0f;
      }
    }
    present_state[(b * channels + d) * state_len + k] = from_float<T>(val);
  }
}

}  // namespace

template <typename T>
Status LaunchCausalConv1DWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* bias,
    const T* conv_state,
    T* output,
    T* present_state,
    CausalConv1DActivation activation,
    int batch_size,
    int channels,
    int seq_len,
    int kernel_size) {
  int total_threads = batch_size * channels;
  int threads_per_block = 256;
  int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  switch (activation) {
    case CausalConv1DActivation::kNone:
      CausalConv1DWithStateKernel<T, 0><<<num_blocks, threads_per_block, 0, stream>>>(
          input, weight, bias, conv_state, output, present_state,
          batch_size, channels, seq_len, kernel_size);
      break;
    case CausalConv1DActivation::kSiLU:
      CausalConv1DWithStateKernel<T, 1><<<num_blocks, threads_per_block, 0, stream>>>(
          input, weight, bias, conv_state, output, present_state,
          batch_size, channels, seq_len, kernel_size);
      break;
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchCausalConv1DWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, CausalConv1DActivation, int, int, int, int);

template Status LaunchCausalConv1DWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    half*, half*, CausalConv1DActivation, int, int, int, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchCausalConv1DWithStateKernel<nv_bfloat16>(
    cudaStream_t, const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*, const nv_bfloat16*,
    nv_bfloat16*, nv_bfloat16*, CausalConv1DActivation, int, int, int, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
