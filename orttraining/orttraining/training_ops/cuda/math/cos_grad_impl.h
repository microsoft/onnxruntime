// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

template <typename T>
Status CosGradImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* input, const T* output_grad, size_t N);

}
}  // namespace onnxruntime
