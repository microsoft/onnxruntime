// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#pragma once

#include "torch/torch.h"
#include "ATen/DLConvertor.h"
#include "core/providers/cuda/torch_wrapper/dlpack_convertor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_common.h"
// #include "include/onnxruntime/core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

at::Tensor ToTorchTensor(OrtValue& ort_value);

OrtValue FromTorchTensor(const at::Tensor& torch_tensor);

// Compute high-dimension matrix multiplication:
//   result = left * right
OrtValue matmul(OrtValue& left, OrtValue& right, const bool left_transpose, const bool right_transpose,
                const float alpha);

OrtValue convolution(OrtValue& input, OrtValue weight, const OrtValue* bias_ptr, const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups);

std::vector<OrtValue> convolution_backward(OrtValue& grad_output, OrtValue& input, OrtValue weight,
                                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, int64_t groups,
                                           bool should_compute_input_grad, bool should_compute_weight_grad,
                                           bool should_compute_bias_grad);

}  // namespace torch_wrapper
}  // namespace cuda
}  // namespace onnxruntime

#endif  // USE_TORCH
