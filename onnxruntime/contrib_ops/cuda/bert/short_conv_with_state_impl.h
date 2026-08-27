// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
    bool apply_silu);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
