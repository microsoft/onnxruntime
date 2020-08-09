// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/torch_wrapper/torch_wrapper.h"
#include "torch/torch.h"
#include "core/framework/tensor.h"
#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

void matmul(
    const Tensor* left,
    const Tensor* right,
    /* output */ Tensor* result,
    const bool left_transpose,
    const bool right_transpose,
    const float alpha) {
  // This information is needed for creating Torch tensor on the right device.
  torch::Tensor torch_left = ConvertOrtTensorToTorchTensor(left);
  torch::Tensor torch_right = ConvertOrtTensorToTorchTensor(right);

  // Transpose left matrix if needed.
  if (left_transpose) {
    torch_left = at::transpose(torch_left, left->Shape().NumDimensions() - 2, left->Shape().NumDimensions() - 1);
  }

  // Transpose right matrix if needed.
  if (right_transpose) {
    torch_right = at::transpose(torch_right, right->Shape().NumDimensions() - 2, right->Shape().NumDimensions() - 1);
  }

  // Torch MatMul.
  auto torch_result = at::matmul(torch_left, torch_right);

  // Scaling MatMul result when needed.
  if (alpha != 1.0f) {
    torch_result = at::mul(torch_result, alpha);
  }

  // Copy torch result to ORT tensor.
  CopyTorchTensorToOrtTensor(torch_result, result);

  ORT_ENFORCE(cudaGetLastError() == cudaSuccess, "Torch Matmul fails.");
}

}  // namespace torch_wrapper
}  // namespace cuda
}  // namespace onnxruntime
