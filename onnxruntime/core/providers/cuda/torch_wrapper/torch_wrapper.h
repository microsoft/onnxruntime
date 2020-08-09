// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "torch/torch.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_common.h"
// #include "include/onnxruntime/core/framework/allocator.h"

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

// Convert ONNX tensor element type to the corresponding PyTorch type.
// Notice that not all ONNX types are supported. 
at::ScalarType GetTorchElementType(const ONNX_NAMESPACE::TensorProto_DataType type);

torch::DeviceType GetTorchDeviceType(OrtDevice::DeviceType device_type);

// Convert ORT tensor to Torch tensor.
torch::Tensor ConvertOrtTensorToTorchTensor(const Tensor* tensor);

// Copy content of Torch tensor to ORT tensor.
void CopyTorchTensorToOrtTensor(const torch::Tensor torch_tensor, Tensor* ort_tensor);

// Compute high-dimension matrix multiplication:
//   result = left * right
void matmul(
    const Tensor* left,
    const Tensor* right,
    Tensor* result,
    const bool left_transpose,
    const bool right_transpose,
    const float alpha);

}  // torch_wrapper
}  // namespace cuda
}  // namespace onnxruntime
