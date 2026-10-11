// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused causal depthwise conv1d CUDA kernel with stateful carry and optional SiLU activation.
//
// Design: One thread block per (batch, channel). Two execution paths:
//
// 1. Decode (L=1):  The convolution window is [past_state(pad), input(1)] with
//    pad = (K-1)*dilation. Load K values into registers, compute a single dot product,
//    shift state.
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
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

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
    const T* __restrict__ past_state,  // [W, B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, 1]
    T* __restrict__ present_state,     // [W, B, C, K-1]
    int batch_channels,                // = batch_size * channels (actual element count)
    int channels,
    int kernel_size,
    int dilation,
    CausalConvLayout act_layout,
    CausalConvLayout state_layout,
    bool apply_silu,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) return;
  const int b = bc / channels;
  const int c = bc % channels;

  const int pad = (kernel_size - 1) * dilation;
  const int64_t state_pos_stride = state_layout.pos_stride;

  // Cache input value in register — avoids redundant global reads
  const int64_t act_offset = act_layout.Offset(b, 0, c);
  const float input_val = to_float(input[act_offset]);

  // seq_len == 1, so the single position is window slot W-1 for both the read and the write.
  // Window-major: one slot is a whole [B, ...] block of batch_channels*pad elements.
  const int64_t state_offset =
      (int64_t)(state_window - 1) * batch_channels * pad + state_layout.Offset(b, 0, c);

  // Cache past_state base pointer for this (b, c)
  const T* ps_in = (past_state != nullptr) ? past_state + state_offset : nullptr;

  // Load weight for this channel: [K] values
  // weight layout: [C, 1, K], so channel c starts at c * K
  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;

  // Convolution window: [past_state(pad), input[0]]. Tap k reads window slot k*dilation.
  for (int k = 0; k < kernel_size - 1; ++k) {
    float wk = to_float(weight[c * kernel_size + k]);
    float xk = (ps_in != nullptr) ? to_float(ps_in[k * dilation * state_pos_stride]) : 0.0f;
    sum += wk * xk;
  }
  // Last tap is the current input
  sum += to_float(weight[c * kernel_size + kernel_size - 1]) * input_val;

  if (apply_silu) {
    sum = silu_fn(sum);
  }
  output[act_offset] = from_float<T>(sum);

  // Update present_state: shift left by 1, append input
  T* ps_out = present_state + state_offset;
  for (int k = 0; k < pad - 1; ++k) {
    ps_out[k * state_pos_stride] =
        (ps_in != nullptr) ? ps_in[(k + 1) * state_pos_stride] : from_float<T>(0.0f);
  }
  if (pad > 0) {
    ps_out[(pad - 1) * state_pos_stride] = from_float<T>(input_val);
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
    bool apply_silu,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) return;

  const int b = bc / channels;
  const int c = bc % channels;
  constexpr int pad = K - 1;

  // seq_len == 1, so the single position is window slot W-1 for both the read and the write.
  // Window-major [W, B, C, K-1]: slot stride is batch_channels*pad and (b, c) flattens to bc.
  const int64_t state_offset =
      static_cast<int64_t>(state_window - 1) * batch_channels * pad + static_cast<int64_t>(bc) * pad;

  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
  const T* w = weight + static_cast<int64_t>(c) * K;
  const T* ps_in = (past_state != nullptr) ? past_state + state_offset : nullptr;

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

  T* ps_out = present_state + state_offset;
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
    const T* __restrict__ past_state,  // [W, B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, L]
    T* __restrict__ present_state,     // [W, B, C, K-1]
    int seq_len,
    int channels,
    int kernel_size,
    int dilation,
    CausalConvLayout act_layout,
    CausalConvLayout state_layout,
    bool apply_silu,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int b = blockIdx.x;
  const int c = blockIdx.y;
  const int tid = threadIdx.x;

  const int pad = (kernel_size - 1) * dilation;
  const int padded_len = pad + seq_len;
  const int64_t state_pos_stride = state_layout.pos_stride;

  // Slot W-1 holds the state after the last token; that is what past_state is read from.
  // Window-major, so one slot spans the whole batch.
  const int64_t slot_stride = (int64_t)batch_size * channels * pad;
  const int64_t last_slot_offset =
      (int64_t)(state_window - 1) * slot_stride + state_layout.Offset(b, 0, c);

  // Shared memory: padded input [pad + L] floats + weight [K] floats
  extern __shared__ float smem[];
  float* s_padded = smem;
  float* s_weight = smem + padded_len;

  // Cooperatively load padded input into shared memory
  // Past state portion: [0..pad-1]
  for (int i = tid; i < pad; i += blockDim.x) {
    if (past_state != nullptr) {
      s_padded[i] = to_float(past_state[last_slot_offset + (int64_t)i * state_pos_stride]);
    } else {
      s_padded[i] = 0.0f;
    }
  }
  // Current input portion: [pad..pad+L-1]
  for (int i = tid; i < seq_len; i += blockDim.x) {
    s_padded[pad + i] = to_float(input[act_layout.Offset(b, i, c)]);
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
      sum += s_weight[k] * s_padded[l + k * dilation];
    }
    if (apply_silu) {
      sum = silu_fn(sum);
    }
    output[act_layout.Offset(b, l, c)] = from_float<T>(sum);
  }

  // Save present_state. The carry state after token t is the pad-length window ending at position
  // t in the [past_state, input] stream, i.e. s_padded[t + 1 .. t + pad]; it goes into the
  // right-aligned slot t + W - seq_len, and earlier tokens fall outside the window. The last
  // token always maps to slot W-1.
  __syncthreads();
  const int first = seq_len > state_window ? seq_len - state_window : 0;
  for (int t = first + tid; t < seq_len; t += blockDim.x) {
    T* ps = present_state + (int64_t)(t + state_window - seq_len) * slot_stride +
            state_layout.Offset(b, 0, c);
    for (int p = 0; p < pad; ++p) {
      ps[(int64_t)p * state_pos_stride] = from_float<T>(s_padded[t + 1 + p]);
    }
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
    const T* __restrict__ past_state,  // [W, B, C, K-1] or nullptr
    T* __restrict__ output,            // [B, C, L]
    T* __restrict__ present_state,     // [W, B, C, K-1]
    int seq_len,
    int channels,
    int kernel_size,
    int dilation,
    CausalConvLayout act_layout,
    CausalConvLayout state_layout,
    bool apply_silu,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int b = blockIdx.x;
  const int c_base = blockIdx.y * CPB;
  const int tid = threadIdx.x;

  const int pad = (kernel_size - 1) * dilation;
  const int padded_len = pad + seq_len;
  const int64_t state_pos_stride = state_layout.pos_stride;

  // Window-major: one slot spans the whole batch.
  const int64_t slot_stride = (int64_t)batch_size * channels * pad;

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
    // Load past state from window slot W-1 (the state after the last token of the previous step)
    const int64_t last_slot_offset =
        (int64_t)(state_window - 1) * slot_stride + state_layout.Offset(b, 0, c);
    for (int i = local_tid; i < pad; i += threads_per_channel) {
      if (past_state != nullptr) {
        s_padded[i] = to_float(past_state[last_slot_offset + (int64_t)i * state_pos_stride]);
      } else {
        s_padded[i] = 0.0f;
      }
    }
    // Load input
    for (int i = local_tid; i < seq_len; i += threads_per_channel) {
      s_padded[pad + i] = to_float(input[act_layout.Offset(b, i, c)]);
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
        sum += s_weight[k] * s_padded[l + k * dilation];
      }
      if (apply_silu) {
        sum = silu_fn(sum);
      }
      output[act_layout.Offset(b, l, c)] = from_float<T>(sum);
    }
  }

  // Unconditional barrier — s_padded is read-only after the cooperative load,
  // so this is safe even when c >= channels.  Hoisted out of the conditional
  // to avoid divergent __syncthreads() (undefined behavior in CUDA).
  __syncthreads();

  if (c < channels) {
    // Save the carry state after token t (window s_padded[t+1 .. t+pad]) into the right-aligned
    // slot t + W - seq_len; earlier tokens fall outside the window. The last token maps to W-1.
    const int first = seq_len > state_window ? seq_len - state_window : 0;
    for (int t = first + local_tid; t < seq_len; t += threads_per_channel) {
      T* ps = present_state + (int64_t)(t + state_window - seq_len) * slot_stride +
              state_layout.Offset(b, 0, c);
      for (int p = 0; p < pad; ++p) {
        ps[(int64_t)p * state_pos_stride] = from_float<T>(s_padded[t + 1 + p]);
      }
    }
  }
}

// =============================================================================
// Channels-last prefill kernel: L>1, one thread per (batch, position, channel)
// with the channel as the fastest-moving thread axis.
//
// Grid:  (ceil(channels / threads), seq_len, batch_size)
// Block: (threads, 1, 1)
// Shared memory: none
//
// The shared-memory kernels above stage one channel per block and walk positions, which is
// coalesced only when positions are contiguous. Under channels_last the contiguous axis is the
// channel, so that access pattern turns every load and store into a strided gather. Here adjacent
// threads hold adjacent channels, so each convolution tap, each state read and every store is a
// single contiguous transaction. Staging is unnecessary because the overlapping taps of
// neighbouring positions are served by L1/L2 rather than shared memory.
// =============================================================================
template <typename T>
__global__ void CausalConvPrefillKernelChannelsLast(
    const T* __restrict__ input,       // [B, L, C]
    const T* __restrict__ weight,      // [C, 1, K]
    const T* __restrict__ bias,        // [C] or nullptr
    const T* __restrict__ past_state,  // [W, B, K-1, C] or nullptr
    T* __restrict__ output,            // [B, L, C]
    T* __restrict__ present_state,     // [W, B, K-1, C]
    int seq_len,
    int channels,
    int kernel_size,
    int dilation,
    CausalConvLayout act_layout,
    CausalConvLayout state_layout,
    bool apply_silu,
    int batch_size,
    int state_window) {  // W: axis-0 extent of past_state / present_state (>= 1)
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) {
    return;
  }
  const int l = blockIdx.y;
  const int b = blockIdx.z;

  const int pad = (kernel_size - 1) * dilation;
  const int64_t state_pos_stride = state_layout.pos_stride;
  // Window-major: one slot spans the whole batch. Slot W-1 holds the state after the last token of
  // the previous step, which is the only slot past_state is read from.
  const int64_t slot_stride = (int64_t)batch_size * channels * pad;
  const int64_t last_slot_offset =
      (int64_t)(state_window - 1) * slot_stride + state_layout.Offset(b, 0, c);

  // Reads the virtual stream [past_state (pad samples), input (seq_len samples)] at index `vp`.
  auto sample = [&](int vp) -> float {
    if (vp >= pad) {
      return to_float(input[act_layout.Offset(b, vp - pad, c)]);
    }
    return past_state != nullptr
               ? to_float(past_state[last_slot_offset + (int64_t)vp * state_pos_stride])
               : 0.0f;
  };

  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    sum += to_float(weight[(int64_t)c * kernel_size + k]) * sample(l + k * dilation);
  }
  if (apply_silu) {
    sum = silu_fn(sum);
  }
  output[act_layout.Offset(b, l, c)] = from_float<T>(sum);

  // The carry state after token l is the pad-length window ending at that token, i.e. virtual
  // stream positions [l + 1, l + pad]. It goes into the right-aligned slot l + W - seq_len;
  // earlier tokens fall outside the window. The last token always maps to slot W-1.
  const int first = seq_len > state_window ? seq_len - state_window : 0;
  if (l >= first) {
    T* ps = present_state + (int64_t)(l + state_window - seq_len) * slot_stride +
            state_layout.Offset(b, 0, c);
    for (int p = 0; p < pad; ++p) {
      ps[(int64_t)p * state_pos_stride] = from_float<T>(sample(l + 1 + p));
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
    int dilation,
    CausalConvLayout act_layout,
    CausalConvLayout state_layout,
    bool apply_silu,
    int max_threads_per_block,
    int state_window) {
  if (seq_len == 1) {
    // Decode fast-path: one thread per (batch, channel)
    int total = batch_size * channels;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // The fixed-K decode kernels hard-code pad == K - 1 and a contiguous state row, so they only
    // apply to the undilated channels-first case (pos_stride is 1 there, and `channels` in the
    // channels-last one).
    switch ((dilation == 1 && state_layout.pos_stride == 1) ? kernel_size : 0) {
      case 2:
        CausalConvDecodeKernelFixedK<T, 2><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu, state_window);
        break;
      case 3:
        CausalConvDecodeKernelFixedK<T, 3><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu, state_window);
        break;
      case 4:
        CausalConvDecodeKernelFixedK<T, 4><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu, state_window);
        break;
      case 5:
        CausalConvDecodeKernelFixedK<T, 5><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, apply_silu, state_window);
        break;
      default:
        CausalConvDecodeKernel<T><<<blocks, threads, 0, stream>>>(
            input, weight, bias, past_state, output, present_state,
            total, channels, kernel_size, dilation, act_layout, state_layout, apply_silu,
            state_window);
        break;
    }
  } else {
    // Prefill path: choose between batched (short seq) or single-channel (long seq) kernel
    int pad = (kernel_size - 1) * dilation;

    // Under channels_last the contiguous axis is the channel, so use the kernel whose fastest
    // thread axis is the channel; the shared-memory kernels below would gather every access.
    // gridDim.y is capped at 65535; longer sequences fall through to the shared-memory kernels,
    // which handle either layout correctly (just less efficiently).
    constexpr int kMaxGridDimY = 65535;
    if (act_layout.chan_stride == 1 && seq_len <= kMaxGridDimY) {
      int threads = channels >= 256 ? 256 : ((channels + 31) / 32) * 32;
      threads = std::min(threads, max_threads_per_block);
      const dim3 grid((channels + threads - 1) / threads, seq_len, batch_size);
      CausalConvPrefillKernelChannelsLast<T><<<grid, threads, 0, stream>>>(
          input, weight, bias, past_state, output, present_state,
          seq_len, channels, kernel_size, dilation, act_layout, state_layout, apply_silu,
          batch_size, state_window);
      return CUDA_CALL(cudaGetLastError());
    }

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
          seq_len, channels, kernel_size, dilation, act_layout, state_layout, apply_silu,
          batch_size, state_window);
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
          seq_len, channels, kernel_size, dilation, act_layout, state_layout, apply_silu,
          batch_size, state_window);
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchCausalConvWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, int, int, int, int, int, CausalConvLayout, CausalConvLayout, bool, int, int);

template Status LaunchCausalConvWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    half*, half*, int, int, int, int, int, CausalConvLayout, CausalConvLayout, bool, int, int);

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template Status LaunchCausalConvWithStateKernel<__nv_bfloat16>(
    cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, CausalConvLayout, CausalConvLayout,
    bool, int, int);
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
