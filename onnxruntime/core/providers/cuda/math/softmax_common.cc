// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/softmax_common.h"

namespace onnxruntime {
namespace cuda {

Status SoftmaxForward(cudnnHandle_t cudnn_handle, const void* alpha, const cudnnTensorDescriptor_t input_tensor,
                      const void* input_data, const void* beta, const cudnnTensorDescriptor_t output_tensor,
                      void* output_data) {
  CUDNN_RETURN_IF_ERROR(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, alpha,
                                            input_tensor, input_data, beta, output_tensor, output_data));
  return Status::OK();
}

Status SoftmaxBackward(cudnnHandle_t cudnn_handle, bool is_log_softmax, const void* alpha,
                       const cudnnTensorDescriptor_t input_tensor, const void* output_data,
                       const void* output_grad_data, const void* beta, const cudnnTensorDescriptor_t output_tensor,
                       void* input_grad_data) {
  CUDNN_RETURN_IF_ERROR(cudnnSoftmaxBackward(cudnn_handle, is_log_softmax ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
                                             CUDNN_SOFTMAX_MODE_INSTANCE, alpha, input_tensor, output_data,
                                             input_tensor, output_grad_data, beta, output_tensor, input_grad_data));
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
