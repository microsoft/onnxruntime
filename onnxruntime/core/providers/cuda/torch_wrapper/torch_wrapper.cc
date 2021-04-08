// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_TORCH

#include "core/providers/cuda/torch_wrapper/torch_wrapper.h"
#include <stdexcept>
#include <cstring>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

at::Tensor ToTorchTensor(OrtValue& ort_value) { return at::fromDLPack(OrtValueToDlpack(ort_value)); }

OrtValue FromTorchTensor(const at::Tensor& torch_tensor) {
  return DlpackToOrtValue(at::toDLPack(torch_tensor), torch_tensor.dtype() == at::kBool);
}

OrtValue matmul(OrtValue& left, OrtValue& right, const bool left_transpose, const bool right_transpose,
                const float alpha) {
  at::Tensor torch_left = ToTorchTensor(left);
  at::Tensor torch_right = ToTorchTensor(right);

  // Transpose left matrix if needed.
  if (left_transpose) {
    torch_left = at::transpose(torch_left, left.Get<Tensor>().Shape().NumDimensions() - 2,
                               left.Get<Tensor>().Shape().NumDimensions() - 1);
  }

  // Transpose right matrix if needed.
  if (right_transpose) {
    torch_right = at::transpose(torch_right, right.Get<Tensor>().Shape().NumDimensions() - 2,
                                right.Get<Tensor>().Shape().NumDimensions() - 1);
  }

  // Torch MatMul.
  auto torch_result = at::matmul(torch_left, torch_right);

  // Scaling MatMul result when needed.
  if (alpha != 1.0f) {
    torch_result = at::mul(torch_result, alpha);
  }

  return FromTorchTensor(torch_result);
}

OrtValue convolution(OrtValue& input, OrtValue weight, const OrtValue* bias_ptr, const std::vector<int64_t>& stride,
                     const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
  at::Tensor torch_input = ToTorchTensor(input);
  at::Tensor torch_weight = ToTorchTensor(weight);
  c10::optional<at::Tensor> torch_bias;
  if (bias_ptr) {
    OrtValue bias = *bias_ptr;
    torch_bias = ToTorchTensor(bias);
  }

  auto torch_result =
      at::convolution(torch_input, torch_weight, torch_bias, stride, padding, dilation, false, 0, groups);
  return FromTorchTensor(torch_result);
}

std::vector<OrtValue> convolution_backward(OrtValue& grad_output, OrtValue& input, OrtValue weight,
                                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, int64_t groups,
                                           bool should_compute_input_grad, bool should_compute_weight_grad,
                                           bool should_compute_bias_grad) {
  at::Tensor torch_grad_output = ToTorchTensor(grad_output);
  at::Tensor torch_input = ToTorchTensor(input);
  at::Tensor torch_weight = ToTorchTensor(weight);
  int64_t k = torch_weight.ndimension();

  std::vector<int64_t> new_stride = std::vector<int64_t>(stride);
  std::vector<int64_t> new_padding = std::vector<int64_t>(padding);
  std::vector<int64_t> new_dilation = std::vector<int64_t>(dilation);

  if (k - 2 > 1) {
    if (new_stride.size() == 1) new_stride = std::vector<int64_t>(k - 2, new_stride[0]);
    if (new_padding.size() == 1) new_padding = std::vector<int64_t>(k - 2, new_padding[0]);
    if (new_dilation.size() == 1) new_dilation = std::vector<int64_t>(k - 2, new_dilation[0]);
  }

  if (k == 3) {
    if (stride.size() == 1) {
      new_stride.insert(new_stride.begin(), 1);
      new_padding.insert(new_padding.begin(), 0);
      new_dilation.insert(new_dilation.begin(), 1);
    }

    torch_grad_output = torch_grad_output.unsqueeze(2);
    torch_input = torch_input.unsqueeze(2);
    torch_weight = torch_weight.unsqueeze(2);
  }

  auto torch_result = at::cudnn_convolution_backward(
      torch_input, torch_grad_output, torch_weight, new_padding, new_stride, new_dilation, groups, false, false, true,
      std::array<bool, 2>{should_compute_input_grad, should_compute_weight_grad});

  std::vector<OrtValue> ort_result;
  if (should_compute_input_grad) {
    at::Tensor grad_input = std::get<0>(torch_result);
    if (k == 3) {
      grad_input = grad_input.squeeze(2);
    }

    ort_result.emplace_back(FromTorchTensor(grad_input));
  } else {
    ort_result.emplace_back(OrtValue());
  }

  if (should_compute_weight_grad) {
    at::Tensor grad_weight = std::get<1>(torch_result);
    if (k == 3) {
      grad_weight = grad_weight.squeeze(2);
    }

    ort_result.emplace_back(FromTorchTensor(grad_weight));
  } else {
    ort_result.emplace_back(OrtValue());
  }

  if (should_compute_bias_grad) {
    std::vector<int64_t> dims;
    for (int64_t dim = 0; dim < torch_input.dim(); dim ++) {
      if (dim != 1) dims.emplace_back(dim);
    }

    at::Tensor grad_bias = torch_grad_output.sum(dims);
    ort_result.emplace_back(FromTorchTensor(grad_bias));
  } else {
    ort_result.emplace_back(OrtValue());
  }

  return ort_result;
}

}  // namespace torch_wrapper
}  // namespace cuda
}  // namespace onnxruntime

#endif  // USE_TORCH
