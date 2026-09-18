// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Element strides of a (batch, position, channel) view over an activation or state tensor.
// Both supported layouts are dense, so a single strided view covers them and one kernel body
// serves the channels-first (batch, channels, length) and channels-last (batch, length, channels)
// layouts. For the state tensors these are the strides *within* one state_window slot.
struct CausalConvLayout {
  int64_t batch_stride;
  int64_t pos_stride;
  int64_t chan_stride;

  __host__ __device__ int64_t Offset(int b, int pos, int c) const {
    return static_cast<int64_t>(b) * batch_stride + static_cast<int64_t>(pos) * pos_stride +
           static_cast<int64_t>(c) * chan_stride;
  }
};

// `length` is the extent of the position axis: seq_len for activations, state_length for state.
inline CausalConvLayout MakeCausalConvLayout(bool channels_last, int64_t channels, int64_t length) {
  return channels_last ? CausalConvLayout{channels * length, channels, 1}
                       : CausalConvLayout{channels * length, 1, length};
}

// Fused causal depthwise conv1d + activation + state management.
// One thread block per (batch, channel). For decode (L=1), this is a simple
// dot product from shared memory. For prefill (L>1), each thread handles
// one output position.
template <typename T>
Status LaunchCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,       // [B, C, L]
    const T* weight,      // [C, 1, K]
    const T* bias,        // [C] or nullptr
    const T* past_state,  // [W, B, C, (K-1)*dilation] or nullptr
    T* output,            // [B, C, L]
    T* present_state,     // [W, B, C, (K-1)*dilation]
    int batch_size,
    int channels,
    int seq_len,
    int kernel_size,
    int dilation,                   // spacing between kernel taps along the causal axis (>= 1)
    CausalConvLayout act_layout,    // strides of input / output
    CausalConvLayout state_layout,  // strides within one past_state / present_state slot
    bool apply_silu,
    int max_threads_per_block,
    // Axis-0 extent W of past_state / present_state (>= 1). The window axis leads the batch axis
    // so that a slot is one contiguous [B, C, (K-1)*dilation] block. Right-aligned: token t writes slot
    // t + W - seq_len and negative slots are skipped, so slot W-1 always holds the state after the
    // last token and is the slot past_state is read from. Pass 1 for a plain single-state tensor
    // with no window axis.
    int state_window = 1);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
