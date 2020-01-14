// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchSkipLayerNormKernel(
    void* output,              // output tensor
    const void* input,         // input tensor
    const void* skip,          // skip tensor
    const void* gamma,         // Layer normalization gamma tensor
    const void* beta,          // Layer normalization beta tensor
    const void* bias,          // Layer normalization beta tensor
    const int batch_size,      // batch size (B)
    const int hidden_size,     // hidden size, it is the leading dimension (ld)
    const int element_count,   // number of elements in input tensor
    const size_t element_size  // element size of input tensor
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
