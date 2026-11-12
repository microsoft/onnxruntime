// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace onnxruntime::cuda {

template <typename T>
void CausalConvWithStateImpl(cudaStream_t stream, const T* input, const T* weight,
                             const T* bias, const T* past, T* output, T* present,
                             int64_t rows, int64_t channels, int64_t length,
                             int64_t kernel, int64_t dilation, int64_t history, bool silu);

}  // namespace onnxruntime::cuda
