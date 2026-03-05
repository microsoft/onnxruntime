// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class CausalConv1DActivation {
  kNone,
  kSiLU,
};

// Launch the fused causal 1D depthwise convolution kernel.
//
// Parameters:
//   stream     - CUDA stream
//   input      - (B, D, T) input tensor
//   weight     - (D, 1, K) depthwise conv weights
//   bias       - (D) or nullptr
//   conv_state - (B, D, K-1) or nullptr (use zero padding for prefill)
//   output     - (B, D, T)
//   present_state - (B, D, K-1) updated carry state
//   activation - activation to fuse
//   batch_size, channels, seq_len, kernel_size
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
    int kernel_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
