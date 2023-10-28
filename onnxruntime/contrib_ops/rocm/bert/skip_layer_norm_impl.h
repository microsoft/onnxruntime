// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T, typename U, typename V, bool Simplified>
Status LaunchSkipLayerNormKernel(
    RocmTuningContext* tuning,
    Stream* stream,
    V* output,                      // output tensor
    T* skip_input_bias_add_output,  // optional output tensor
    const T* input,                 // input tensor
    const T* skip,                  // skip tensor
    const V* gamma,                 // Layer normalization gamma tensor
    const V* beta,                  // Layer normalization beta tensor
    const T* bias,                  // Layer normalization beta tensor
    float epsilon,                  // Layer normalization epsilon
    int hidden_size,                // hidden size, it is the leading dimension (ld)
    int element_count               // number of elements in input tensor
);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
