// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
    const T* past_state,  // [B, C, K-1] or nullptr
    T* output,            // [B, C, L]
    T* present_state,     // [B, C, K-1]
    int batch_size,
    int channels,
    int seq_len,
    int kernel_size,
    bool apply_silu,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
