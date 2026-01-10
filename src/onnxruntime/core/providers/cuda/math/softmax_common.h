// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

Status SoftmaxForward(cudnnHandle_t cudnn_handle, const void* alpha, const cudnnTensorDescriptor_t input_tensor,
                      const void* input_data, const void* beta, const cudnnTensorDescriptor_t output_tensor,
                      void* output_data);

Status SoftmaxBackward(cudnnHandle_t cudnn_handle, bool is_log_softmax, const void* alpha,
                       const cudnnTensorDescriptor_t input_tensor, const void* output_data,
                       const void* output_grad_data, const void* beta, const cudnnTensorDescriptor_t output_tensor,
                       void* input_grad_data);

}  // namespace cuda
}  // namespace onnxruntime
