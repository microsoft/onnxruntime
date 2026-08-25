// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LaunchShortConvKernel(
    cudaStream_t stream,
    const T* input,
    const T* weight,
    const T* norm_scale,
    const T* bias,
    float* inv_rms_workspace,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t hc_mult,
    int64_t hidden_size,
    int64_t kernel_size,
    int64_t dilation,
    float epsilon,
    bool apply_silu);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
