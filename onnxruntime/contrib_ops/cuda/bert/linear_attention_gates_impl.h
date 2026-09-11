// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// decay = decay_scale * Softplus(a + dt_bias); beta = Sigmoid(b).
// `beta` and `b` may both be nullptr; the two per-head parameter vectors are float32.
template <typename T>
Status LaunchLinearAttentionGateKernel(
    cudaStream_t stream,
    T* decay,
    T* beta,
    const T* a,
    const T* b,
    const float* dt_bias,
    const float* decay_scale,
    int64_t num_tokens,
    int num_heads);

// Y = X * rsqrt(mean(X^2) + epsilon) * scale * SiLU(gate), reduced over groups of
// `norm_size` contiguous elements, with all arithmetic in float32.
template <typename T>
Status LaunchGatedRMSNormKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const T* scale,
    const T* gate,
    int64_t num_rows,
    int norm_size,
    float epsilon);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
