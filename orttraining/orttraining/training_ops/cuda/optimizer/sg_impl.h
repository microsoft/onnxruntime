// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {
template <typename T>
void SGDOptimizerImpl(
    cudaStream_t stream,
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weight_out,
    T* gradients_out,
    size_t count);
}
}  // namespace onnxruntime
