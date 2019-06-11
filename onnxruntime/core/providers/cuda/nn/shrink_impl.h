// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/fast_divmod.h"
namespace onnxruntime {
namespace cuda {

template <typename T>
void ShrinkImpl(
    const T* input_data,
    const T bias,
    const T lambda,
    T* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
