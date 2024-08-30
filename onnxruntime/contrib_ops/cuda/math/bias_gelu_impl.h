// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
void LaunchBiasGeluKernel(cudaStream_t stream, int64_t input_size, int64_t bias_size, const T* X, const T* B, T* Y);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
