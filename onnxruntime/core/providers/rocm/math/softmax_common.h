// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/rocm/miopen_common.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {

template <typename input_t, typename output_t>
struct SoftmaxParams : onnxruntime::rocm::tunable::OpParams {
  SoftmaxParams(hipStream_t stream, output_t* output, const input_t* input, int softmax_elements,
                int input_stride, int output_stride, int batch_count, bool is_log_softmax)
      : OpParams(stream), output(output), input(input), softmax_elements(softmax_elements), input_stride(input_stride), output_stride(output_stride), batch_count(batch_count), is_log_softmax(is_log_softmax) {}

  std::string Signature() const override {
    std::string sig = std::to_string(batch_count) + "_" + std::to_string(softmax_elements);
    return sig;
  }

  output_t* output;
  const input_t* input;
  int softmax_elements;
  int input_stride;
  int output_stride;
  int batch_count;
  bool is_log_softmax;
};

Status SoftmaxForward(miopenHandle_t miopen_handle, const void* alpha, const miopenTensorDescriptor_t input_tensor,
                      const void* input_data, const void* beta, const miopenTensorDescriptor_t output_tensor,
                      void* output_data);

Status SoftmaxBackward(miopenHandle_t miopen_handle, bool is_log_softmax, const void* alpha,
                       const miopenTensorDescriptor_t input_tensor, const void* output_data,
                       const void* output_grad_data, const void* beta, const miopenTensorDescriptor_t output_tensor,
                       void* input_grad_data);

}  // namespace rocm
}  // namespace onnxruntime
