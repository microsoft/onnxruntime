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
    T* output,        // normalized output tensor
    T* sum_output,    // sum of the input and skip (and bias if it exists) tensors output
    const T* input,   // input tensor
    const T* skip,    // skip tensor
    const T* bias,    // bias tensor
    const T* gamma,   // Layer normalization gamma tensor
    const T* beta,    // Layer normalization beta tensor
    float epsilon,    // Layer normalization epsilon
    int hidden_size,  // hidden size, it is the leading dimension (ld)
    int row_count,    // number of rows. That is total number of elements divided by hidden size.
    int skip_size);   // number of elements of the skip tensor

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
