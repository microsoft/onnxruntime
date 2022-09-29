// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
Status SoftmaxGradImpl(cudaStream_t stream, cudnnHandle_t cudnn_handle, T* input_grad, const T* output_grad,
                       const T* softmax_output, int element_count, int batch_count, bool is_log_softmax);
}
}  // namespace onnxruntime
