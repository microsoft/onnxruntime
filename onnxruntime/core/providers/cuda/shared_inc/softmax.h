// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

Status SoftmaxForward(cudnnHandle_t cudnn_handle, const void* alpha, const cudnnTensorDescriptor_t input_tensor,
                      const void* input_data, const void* beta, const cudnnTensorDescriptor_t output_tensor,
                      void* output_data) {
  CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, alpha,
                                            input_tensor, input_data, beta, output_tensor, output_data));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
