// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/fast_divmod.h"
namespace onnxruntime {
namespace cuda {

template <typename T>
void BatchNormImpl(
    const T* input_data,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const T epsilon,
    const fast_divmod& fdm_HWD,
    const fast_divmod& fdm_C,
    T* fused_alpha,
    T* fused_bias,
    T* output_data,
    size_t N);

}  // namespace cuda
}  // namespace onnxruntime
