// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
Status SoftmaxDropoutGradImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* input_grad_data,
                              const T* output_grad_data, const bool* mask_data, const T* softmax_output_data,
                              int element_count, int batch_count, const float ratio);

}  // namespace cuda
}  // namespace onnxruntime
