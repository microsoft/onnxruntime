// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status LaunchGeluKernel(cudaStream_t stream, const T* input, T* output, size_t count);

template <typename T>
Status LaunchFastGeluKernel(const cudaDeviceProp& prop, cudaStream_t stream, int input_length, int bias_length,
                            const T* input, const T* bias, T* output, bool use_half2);

}  // namespace cuda
}  // namespace onnxruntime
