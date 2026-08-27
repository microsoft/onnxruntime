// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/short_conv_with_state_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "contrib_ops/cuda/bert/kernel_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Pass 1: Compute inverse RMS per (batch, time, hc_mult) row and write normalized values.
// One block per row.
template <typename T>
__global__ void ShortConvWithStateNormKernel(
    const T* input,
    const T* norm_scale,
    T* normed,
    int64_t total_rows,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t channels,
    int64_t sequence_length,
    float epsilon) {
  extern __shared__ float shared[];

  for (int64_t row = blockIdx.x; row < total_rows; row += gridDim.x) {
    const T* input_row = input + row * hidden_size;

    // Compute sum of squares for this row.
    float sum_sq = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float value = to_float<T>(input_row[i]);
      sum_sq += value * value;
    }
    sum_sq = kernel_helper::BlockSum(sum_sq, shared);

    float inv_rms_val;
    if (threadIdx.x == 0) {
      inv_rms_val = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
    }
    // Broadcast inv_rms from thread 0.
    __shared__ float inv_rms_shared;
    if (threadIdx.x == 0) {
      inv_rms_shared = inv_rms_val;
    }
    __syncthreads();
    inv_rms_val = inv_rms_shared;

    // Write normalized values. Row index encodes (b, t, g).
    const int64_t g = row % hc_mult;
    // base offset in the [B, S, C] normed buffer
    const int64_t base = (row / hc_mult) * channels + g * hidden_size;
    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
      const float scale = to_float<T>(norm_scale[g * hidden_size + i]);
      normed[base + i] = from_float<T>(to_float<T>(input_row[i]) * inv_rms_val * scale);
    }
  }
}

// Pass 2: Dilated causal convolution using past_state + normed values.
// One thread per (batch, time, channel) output element.
template <typename T>
__global__ void ShortConvWithStateConvKernel(
    const T* normed,
    const T* past_state,
    const T* weight,
    const T* bias,
    T* output,
    int64_t total,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    int64_t state_len,
    int64_t channels,
    bool apply_silu) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t c = linear % hidden_size;
    const int64_t g = (linear / hidden_size) % hc_mult;
    const int64_t t = (linear / channels) % sequence_length;
    const int64_t b = linear / (sequence_length * channels);
    const int64_t flat_channel = g * hidden_size + c;

    const float bias_val = bias == nullptr ? 0.0f : to_float<T>(bias[flat_channel]);
    float sum = bias_val;

    for (int64_t k = 0; k < kernel_size; ++k) {
      // Source position in the combined [past_state | normed_current] timeline.
      const int64_t src = (state_len + t) - (kernel_size - 1 - k) * dilation;
      float src_val = 0.0f;
      if (src >= 0 && src < state_len) {
        // From past_state[b, flat_channel, src]
        if (past_state != nullptr) {
          src_val = to_float<T>(past_state[(b * channels + flat_channel) * state_len + src]);
        }
      } else if (src >= state_len && src < state_len + sequence_length) {
        // From normed[b, src - state_len, flat_channel]
        src_val = to_float<T>(normed[(b * sequence_length + (src - state_len)) * channels + flat_channel]);
      }
      sum += src_val * to_float<T>(weight[flat_channel * kernel_size + k]);
    }

    if (apply_silu) {
      sum = kernel_helper::SiluFloat(sum);
    }
    output[linear] = from_float<T>(sum);
  }
}

// Pass 3: Update present_state from the tail of the combined timeline.
// One thread per (batch, channel, state_position) element.
template <typename T>
__global__ void ShortConvWithStateUpdateKernel(
    const T* normed,
    const T* past_state,
    T* present_state,
    int64_t total_state_elements,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t channels,
    int64_t state_len) {
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total_state_elements;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t s = linear % state_len;
    const int64_t flat_c = (linear / state_len) % channels;
    const int64_t b = linear / (channels * state_len);

    // Position in the combined timeline: offset from the end.
    // Combined timeline has length state_len + sequence_length.
    // present_state stores the last state_len values.
    const int64_t timeline_pos = sequence_length + s;  // = (state_len + S) - state_len + s = S + s

    float val = 0.0f;
    if (timeline_pos < state_len) {
      // Still in past_state range.
      if (past_state != nullptr) {
        val = to_float<T>(past_state[(b * channels + flat_c) * state_len + timeline_pos]);
      }
    } else {
      // In the normed range: timeline_pos - state_len
      const int64_t normed_t = timeline_pos - state_len;
      if (normed_t < sequence_length) {
        val = to_float<T>(normed[(b * sequence_length + normed_t) * channels + flat_c]);
      }
    }
    present_state[(b * channels + flat_c) * state_len + s] = from_float<T>(val);
  }
}

}  // namespace

template <typename T>
Status LaunchShortConvWithStateKernel(
    cudaStream_t stream,
    const T* input,
    const T* past_state,
    const T* norm_scale,
    const T* weight,
    const T* bias,
    T* normed_workspace,
    T* output,
    T* present_state,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    int64_t state_len,
    float epsilon,
    bool apply_silu) {
  const int64_t channels = hc_mult * hidden_size;
  const int64_t rows = batch_size * sequence_length * hc_mult;
  const int64_t total = batch_size * sequence_length * channels;

  if (total == 0) {
    // Still need to copy/zero the state.
    const int64_t state_total = batch_size * channels * state_len;
    if (state_total > 0) {
      if (past_state != nullptr) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(present_state, past_state,
                                             static_cast<size_t>(state_total) * sizeof(T),
                                             cudaMemcpyDeviceToDevice, stream));
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(present_state, 0,
                                             static_cast<size_t>(state_total) * sizeof(T), stream));
      }
    }
    return Status::OK();
  }

  // Pass 1: Normalize input.
  {
    const int norm_blocks = static_cast<int>(std::min(rows, kernel_helper::kMaxGridDimX));
    const size_t shared_bytes = static_cast<size_t>(kernel_helper::kThreads) * sizeof(float);
    ShortConvWithStateNormKernel<T><<<norm_blocks, kernel_helper::kThreads, shared_bytes, stream>>>(
        input, norm_scale, normed_workspace, rows, hc_mult, hidden_size, channels, sequence_length, epsilon);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  // Pass 2: Convolution.
  {
    ShortConvWithStateConvKernel<T><<<kernel_helper::GridSize(total), kernel_helper::kThreads, 0, stream>>>(
        normed_workspace, past_state, weight, bias, output, total, batch_size, sequence_length,
        hc_mult, hidden_size, kernel_size, dilation, state_len, channels, apply_silu);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());
  }

  // Pass 3: Update present state.
  {
    const int64_t state_total = batch_size * channels * state_len;
    if (state_total > 0) {
      ShortConvWithStateUpdateKernel<T><<<kernel_helper::GridSize(state_total), kernel_helper::kThreads, 0, stream>>>(
          normed_workspace, past_state, present_state, state_total, batch_size, sequence_length,
          channels, state_len);
      CUDA_RETURN_IF_ERROR(cudaGetLastError());
    }
  }

  return Status::OK();
}

template Status LaunchShortConvWithStateKernel<float>(cudaStream_t, const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvWithStateKernel<half>(cudaStream_t, const half*, const half*, const half*, const half*, const half*, half*, half*, half*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);
template Status LaunchShortConvWithStateKernel<__nv_bfloat16>(cudaStream_t, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, float, bool);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
