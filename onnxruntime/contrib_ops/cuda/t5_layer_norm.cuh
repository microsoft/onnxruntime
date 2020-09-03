// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// T5 specific layer normalization
template <typename T, typename U>
void launch_t5_layer_norm(
    T* vals,
    const T* input,
    const T* gamma,
    const T* beta,
    U epsilon,
    int64_t n1,
    int64_t n2,
    U* invvars,
    U* means);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime