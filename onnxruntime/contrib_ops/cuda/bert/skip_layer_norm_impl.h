// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, bool Simplified>
void LaunchSkipLayerNormKernel(
    cudaStream_t stream,
    T* output,                      // normalized output tensor
    T* skip_input_bias_add_output,  // sum of the input and skip (and bias if it exists) tensors output
    const T* input,                 // input tensor
    const T* skip,                  // skip tensor
    const T* gamma,                 // Layer normalization gamma tensor
    const T* beta,                  // Layer normalization beta tensor
    const T* bias,                  // Layer normalization beta tensor
    float epsilon,                  // Layer normalization epsilon
    int hidden_size,                // hidden size, it is the leading dimension (ld)
    int row_count,                  // number of rows. That is total number of elements divided by hidden size.
    bool skip_broadcasted,          // determines if broadcasting should be implemented
    int skip_size);                 // determines size of the skip tensor

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
