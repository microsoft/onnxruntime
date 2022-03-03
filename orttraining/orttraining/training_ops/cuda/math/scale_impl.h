// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

#include <cuda_runtime.h>

template <typename T>
void Impl_Scale(
    cudaStream_t stream,
    const T* input_data,
    const float scale_value,
    T* output_data,
    size_t count);
 }
}  // namespace onnxruntime
