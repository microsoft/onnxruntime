// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

const int SUPPORTED_REDUCTION_DIM[] = {768, 512, 1024, 1536, 2048, 2560};

// Custom fused bias add with layer normalization
template <typename T, typename U>
void launch_custom_layer_norm(
    T* vals,
    const T* residual,
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