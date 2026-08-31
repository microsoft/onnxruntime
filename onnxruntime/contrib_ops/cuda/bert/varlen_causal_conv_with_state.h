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
  std::string activation_;
  int state_update_capacity_;
};

// Launches the packed varlen causal-conv recurrence.
//
// input/output hold every sequence's tokens back to back along axis 0. cu_seqlens is a device
// int32 tensor of length (batch_size + 1): sequence r occupies the half-open token range
// [cu_seqlens[r], cu_seqlens[r + 1]). Offset *values* are a trusted producer precondition -- the
// device kernel validates the global endpoints and its local interval before any data access.
// Invalid offsets are contained by returning without data accesses; outputs are unspecified.
template <typename T>
Status LaunchVarlenCausalConvWithStateKernel(
    cudaStream_t stream,
    const T* input,                // [total_tokens, channels]
    const T* weight,               // [channels, 1, kernel_size]
    const T* bias,                 // [channels] or nullptr
    const T* initial_state,        // [batch_size, channels, kernel_size - 1], required
    T* output,                     // [total_tokens, channels]
    T* final_state,                // [batch_size, channels, kernel_size - 1]
    T* state_update,               // [batch_size, state_update_capacity, channels] or nullptr
    const int32_t* cu_seqlens,     // [batch_size + 1], device-resident
    const int32_t* capture_count,  // [batch_size] or nullptr
    int batch_size,
    int total_tokens,
    bool all_ones,
    int channels,
    int kernel_size,
    bool apply_silu,
    int max_threads_per_block,
    int state_update_capacity);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
