// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
bool LaunchSkipLayerNormKernel(
    hipStream_t stream,
    T* output,                // output tensor
    const T* input,           // input tensor
    const T* skip,            // skip tensor
    const T* gamma,           // Layer normalization gamma tensor
    const T* beta,            // Layer normalization beta tensor
    const T* bias,            // Layer normalization beta tensor
    const float epsilon,      // Layer normalization epsilon
    const int hidden_size,    // hidden size, it is the leading dimension (ld)
    const int element_count,  // number of elements in input tensor
    const bool tuning);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
