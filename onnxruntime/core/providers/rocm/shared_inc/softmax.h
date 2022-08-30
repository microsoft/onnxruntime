// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/rocm/miopen_common.h"

namespace onnxruntime {
namespace rocm {

Status SoftmaxForward(miopenHandle_t miopen_handle, const void* alpha, const miopenTensorDescriptor_t input_tensor,
                      const void* input_data, const void* beta, const miopenTensorDescriptor_t output_tensor,
                      void* output_data) {
  MIOPEN_RETURN_IF_ERROR(miopenSoftmaxForward_V2(miopen_handle, alpha, input_tensor, input_data, beta, output_tensor,
                                                 output_data, MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_INSTANCE));
  return Status::OK();
}

}  // namespace rocm
}  // namespace onnxruntime
