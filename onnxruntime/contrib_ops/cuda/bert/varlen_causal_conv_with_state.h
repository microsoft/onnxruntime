// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <cstdint>
#include <string>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class VarlenCausalConvWithState final : public onnxruntime::cuda::CudaKernel {
 public:
  VarlenCausalConvWithState(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int ndim_;
  std::string activation_;
  // Leading (axis-0) extent of past_state / present_state; 0 means no window axis (single state).
  int state_window_;
};

// Launches the packed varlen causal-conv recurrence.
//
// input/output hold every sequence's tokens back to back along axis 0. cu_seqlens is a device
// int32 tensor of length (batch_size + 1): sequence r occupies the half-open token range
// [cu_seqlens[r], cu_seqlens[r + 1]). Offset *values* are a trusted producer precondition -- the
// caller only validates tensor shapes/types on the host and this kernel indexes tokens by the
// offsets as-is. The convolution never reads across a request boundary: a tap that lands before
// a request's first token comes from that request's own past_state (or zero), never from a
// neighboring request's tokens.
//
// all_ones is a host-known precondition -- total_tokens == batch_size established from tensor
// shapes -- that selects a dedicated one-token-per-sequence fast path without reading cu_seqlens
// at all (token i belongs to sequence i directly).
template <typename T>
Status LaunchVarlenCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,             // [total_tokens, channels]
    const T* weight,            // [channels, 1, kernel_size]
    const T* bias,              // [channels] or nullptr
    const T* past_state,        // [state_window, batch_size, channels, kernel_size - 1] or nullptr
    T* output,                  // [total_tokens, channels]
    T* present_state,           // [state_window, batch_size, channels, kernel_size - 1]
    const int32_t* cu_seqlens,  // [batch_size + 1], device-resident
    int batch_size,
    bool all_ones,
    int channels,
    int kernel_size,
    bool apply_silu,
    int max_threads_per_block,
    // Axis-0 extent W of past_state / present_state (>= 1). Right-aligned per sequence: for a
    // sequence of length L, local position t writes slot t + W - L and negative slots are
    // skipped, so slot W-1 always holds the state after that sequence's last token. Pass 1 for a
    // plain single-state tensor with no window axis.
    int state_window = 1);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
