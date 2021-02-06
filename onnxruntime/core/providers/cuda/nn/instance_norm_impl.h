// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/fast_divmod.h"
namespace onnxruntime {
namespace cuda {

template <typename T>
void InstanceNormImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* scale,
    const T* bias,
    const T* mean,
    const T* variance,
    const double variance_correction,
    const double epsilon,
    const fast_divmod& fdm_HW,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
