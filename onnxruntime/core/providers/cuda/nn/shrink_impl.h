// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

template <typename T>
void ShrinkImpl(
    cudaStream_t stream,
    const T* input_data,
    const float bias,
    const float lambda,
    T* output_data,
    size_t count);

}  // namespace cuda
}  // namespace onnxruntime
