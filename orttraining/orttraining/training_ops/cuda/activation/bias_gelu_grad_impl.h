// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime {
namespace cuda {

// assumptions:
// - dY, X, dX have input_size elements
// - B has bias_size elements
// - input_size % bias_size == 0
template <typename T, typename GeluComputationMode>
void LaunchBiasGeluGradDxKernel(
    cudaStream_t stream,
    int64_t input_size, int64_t bias_size,
    const T* dY, const T* X, const T* B, T* dX);

}  // namespace cuda
}  // namespace onnxruntime
