// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Packed (ragged) variant of the causal depthwise conv1d + state kernel: every sequence's tokens
// are packed back to back along a single token axis instead of padded into a [batch, channels,
// seq_len] tensor, and cu_seqlens gives each sequence's token range.
//
// Unlike the dense kernel, the ragged path here never buffers a padded [pad + seq_len] window in
// shared memory: seq_len is not known on the host per request (only cu_seqlens is, and only on
// device), so a shared-memory buffer cannot be sized ahead of the launch. Instead every tap read
// -- for both the convolution sum and the state-window write -- is resolved directly against
// global memory through one branch: a tap at a negative local position comes from that request's
// past_state (or zero); a tap at a non-negative local position comes from the packed input at
// that request's own token range and never from a neighboring request.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include "contrib_ops/cuda/bert/varlen_causal_conv_with_state.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

__device__ __forceinline__ float varlen_silu_fn(float x) {
  return x / (1.0f + expf(-x));
}

// Reads a single tap at local position `local_pos` (relative to a request's own first token).
// local_pos < 0 refers to a position before the request started: it comes from that request's
// own past_state slot (ps_last, already offset to slot W-1; never a neighboring request's
// tokens, and never a shared "left edge" buffer). local_pos >= 0 reads the packed input at that
// request's own token range, starting at `start`.
template <typename T>
__device__ __forceinline__ float ReadVarlenCausalConvTap(
    const T* __restrict__ input,
    const T* ps_last,
    int64_t start,
    int channels,
    int c,
    int pad,
    int local_pos) {
  if (local_pos < 0) {
    // local_pos in [-pad, -1] maps to past_state slot (local_pos + pad) in [0, pad - 1].
    return (ps_last != nullptr) ? to_float(ps_last[local_pos + pad]) : 0.0f;
  }
  return to_float(input[(start + local_pos) * channels + c]);
}

// All-ones decode fast path: every sequence has exactly one token, so token index == sequence
// index directly and cu_seqlens is never read. Identical math to the dense decode kernel, just
// addressed through the packed [total_tokens, channels] == [batch_size, channels] layout.
//
// Grid:  (ceil(batch_size * channels / threads), 1, 1)
// Block: (threads, 1, 1)
template <typename T>
__global__ void VarlenCausalConvDecodeKernel(
    const T* __restrict__ input,       // [batch_size, channels]
    const T* __restrict__ weight,      // [channels, 1, kernel_size]
    const T* __restrict__ bias,        // [channels] or nullptr
    const T* __restrict__ past_state,  // [state_window, batch_size, channels, kernel_size - 1] or nullptr
    T* __restrict__ output,            // [batch_size, channels]
    T* __restrict__ present_state,     // [state_window, batch_size, channels, kernel_size - 1]
    int batch_channels,                // = batch_size * channels
    int channels,
    int kernel_size,
    bool apply_silu,
    int state_window) {
  const int bc = blockIdx.x * blockDim.x + threadIdx.x;
  if (bc >= batch_channels) return;
  const int c = bc % channels;
  const int pad = kernel_size - 1;

  const float input_val = to_float(input[bc]);

  // Slot W-1 is both where past_state is read from and where the fresh state is written.
  const int64_t state_offset = (int64_t)(state_window - 1) * batch_channels * pad + (int64_t)bc * pad;
  const T* ps_in = (pad > 0 && past_state != nullptr) ? past_state + state_offset : nullptr;
  const int64_t weight_offset = static_cast<int64_t>(c) * kernel_size;

  float sum = (bias != nullptr) ? to_float(bias[c]) : 0.0f;
  for (int k = 0; k < pad; ++k) {
    const float wk = to_float(weight[weight_offset + k]);
    const float xk = (ps_in != nullptr) ? to_float(ps_in[k]) : 0.0f;
    sum += wk * xk;
  }
  sum += to_float(weight[weight_offset + pad]) * input_val;

  if (apply_silu) {
    sum = varlen_silu_fn(sum);
  }
  output[bc] = from_float<T>(sum);

  if (pad > 0) {
    T* ps_out = present_state + state_offset;
    for (int k = 0; k < pad - 1; ++k) {
      ps_out[k] = (ps_in != nullptr) ? ps_in[k + 1] : from_float<T>(0.0f);
    }
    ps_out[pad - 1] = from_float<T>(input_val);
  }
}

// General ragged path: one block per (sequence, channel). No shared-memory padded buffer -- see
// the file header. Handles every kernel_size and every combination of ragged sequence lengths.
//
// Grid:  (batch_size, channels, 1)
// Block: (threads, 1, 1)
template <typename T>
__global__ void VarlenCausalConvKernel(
    const T* __restrict__ input,   // [total_tokens, channels]
    const T* __restrict__ weight,  // [channels, 1, kernel_size]
    const T* __restrict__ bias,    // [channels] or nullptr
    const T* past_state,           // [state_window, batch_size, channels, kernel_size - 1] or nullptr
    T* present_state,              // [state_window, batch_size, channels, kernel_size - 1]
    T* __restrict__ output,        // [total_tokens, channels]
    const int32_t* __restrict__ cu_seqlens,
    int channels,
    int kernel_size,
    bool apply_silu,
    int batch_size,
    int state_window) {
  const int bc = blockIdx.x;
  const int r = bc / channels;
  const int c = bc % channels;
  const int tid = threadIdx.x;
  const int num_threads = blockDim.x;

  const int pad = kernel_size - 1;
  const int64_t start = cu_seqlens[r];
  const int local_len = static_cast<int>(cu_seqlens[r + 1] - cu_seqlens[r]);

  const int64_t slot_stride = (int64_t)batch_size * channels * pad;
  const int64_t last_slot_offset =
      (int64_t)(state_window - 1) * slot_stride + ((int64_t)r * channels + c) * pad;
  const T* ps_last = (pad > 0 && past_state != nullptr) ? past_state + last_slot_offset : nullptr;

  const T* w = weight + (int64_t)c * kernel_size;
  const float bias_val = (bias != nullptr) ? to_float(bias[c]) : 0.0f;

  for (int t = tid; t < local_len; t += num_threads) {
    float sum = bias_val;
    for (int k = 0; k < kernel_size; ++k) {
      sum += to_float(w[k]) * ReadVarlenCausalConvTap(input, ps_last, start, channels, c, pad, t - pad + k);
    }
    if (apply_silu) {
      sum = varlen_silu_fn(sum);
    }
    output[(start + t) * channels + c] = from_float<T>(sum);
  }

  // present_state slot t + W - local_len holds the pad most recent raw inputs ending at local
  // position t (positions t - pad + 1 .. t), resolved through the same past-state-vs-input
  // branch. Positions before `first` fall entirely outside the window and are left at the zero
  // the caller already memset the whole buffer to.
  if (pad > 0) {
    const int first = local_len > state_window ? local_len - state_window : 0;
    for (int t = first + tid; t < local_len; t += num_threads) {
      const int state_slot = t + state_window - local_len;
      T* ps_out = present_state + (int64_t)state_slot * slot_stride + ((int64_t)r * channels + c) * pad;
      for (int p = 0; p < pad; ++p) {
        ps_out[p] =
            from_float<T>(ReadVarlenCausalConvTap(input, ps_last, start, channels, c, pad, t - pad + 1 + p));
      }
    }
  }
}

}  // namespace

template <typename T>
Status LaunchVarlenCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* bias,
    const T* past_state,
    T* output,
    T* present_state,
    const int32_t* cu_seqlens,
    int batch_size,
    bool all_ones,
    int channels,
    int kernel_size,
    bool apply_silu,
    int max_threads_per_block,
    int state_window) {
  const int64_t batch_channels = static_cast<int64_t>(batch_size) * channels;
  ORT_RETURN_IF_NOT(batch_channels <= std::numeric_limits<int>::max(),
                    "VarlenCausalConvWithState: batch_size * channels exceeds INT_MAX");

  if (all_ones) {
    const int total = static_cast<int>(batch_channels);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    VarlenCausalConvDecodeKernel<T><<<blocks, threads, 0, stream>>>(
        input, weight, bias, past_state, output, present_state,
        total, channels, kernel_size, apply_silu, state_window);
    return CUDA_CALL(cudaGetLastError());
  }

  // One block per (sequence, channel) already gives batch_size * channels blocks of parallelism,
  // which is ample for this op (unlike LinearAttention's recurrent kernel, this never collapses
  // to a handful of blocks for a small packed batch), so no column-split occupancy strategy is
  // needed here. Each block's thread count is fixed at launch time -- independent of any
  // particular request's length, which is only known on the device -- and threads stride over
  // local positions, so it stays correct for every ragged length.
  const dim3 grid(static_cast<unsigned int>(batch_channels), 1, 1);
  int threads = std::min(256, max_threads_per_block);
  threads = std::max(32, (threads / 32) * 32);
  const dim3 block(threads, 1, 1);

  VarlenCausalConvKernel<T><<<grid, block, 0, stream>>>(
      input, weight, bias, past_state, present_state, output, cu_seqlens,
      channels, kernel_size, apply_silu, batch_size, state_window);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations
template Status LaunchVarlenCausalConvWithStateKernel<float>(
    cudaStream_t, const float*, const float*, const float*, const float*,
    float*, float*, const int32_t*, int, bool, int, int, bool, int, int);

template Status LaunchVarlenCausalConvWithStateKernel<half>(
    cudaStream_t, const half*, const half*, const half*, const half*,
    half*, half*, const int32_t*, int, bool, int, int, bool, int, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
