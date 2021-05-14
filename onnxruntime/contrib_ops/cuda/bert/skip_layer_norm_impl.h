// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchSkipLayerNormKernel(
    cudaStream_t stream,
    void* output,        // output tensor
    const void* input,   // input tensor
    const void* skip,    // skip tensor
    const void* gamma,   // Layer normalization gamma tensor
    const void* beta,    // Layer normalization beta tensor
    const void* bias,    // Layer normalization beta tensor
    float epsilon,      // Layer normalization epsilon
    int hidden_size,     // hidden size, it is the leading dimension (ld)
    int element_count,   // number of elements in input tensor
    size_t element_size  // element size of input tensor
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
