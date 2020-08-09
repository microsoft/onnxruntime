// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/torch_wrapper/torch_wrapper.h"
#include <stdexcept>
#include <cstring>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

at::ScalarType GetTorchElementType(const ONNX_NAMESPACE::TensorProto_DataType type) {
  // See the following header for supported types in Pytorch.
  // https://pytorch.org/cppdocs/api/program_listing_file_torch_csrc_api_include_torch_types.h.html
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      return at::kChar;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      return at::kByte;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      return at::kShort;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      return at::kInt;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      return at::kLong;
    default:
      throw std::invalid_argument("Unsupported ONNX-to-Torch type conversion");
  }
}

torch::DeviceType GetTorchDeviceType(OrtDevice::DeviceType device_type) {
  switch (device_type) {
    case OrtDevice::GPU:
      return torch::kCUDA;
    case OrtDevice::CPU:
      return torch::kCPU;
    default:
      throw std::invalid_argument("Unsupported ONNX-to-ORT device type conversion");
  }
}

OrtDevice::DeviceType GetOrtDeviceType(torch::DeviceType device_type) {
  switch (device_type) {
    case torch::kCUDA:
      return OrtDevice::GPU;
    case torch::kCPU:
      return OrtDevice::CPU;
    default:
      throw std::invalid_argument("Unsupported Torch-to-ORT device type conversion");
  }
}

torch::Tensor ConvertOrtTensorToTorchTensor(const Tensor* tensor) {
  const auto& element_type = tensor->GetElementType();
  const auto torch_element_type = GetTorchElementType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(element_type));

  const auto location = tensor->Location();
  const auto torch_device = GetTorchDeviceType(location.device.Type());
  const auto torch_device_id = location.device.Id();

  const auto torch_tensor_options = torch::TensorOptions().dtype(torch_element_type).device(torch_device, torch_device_id);

  torch::Tensor torch_tensor = torch::zeros(c10::IntArrayRef{tensor->Shape().GetDims()}, torch_tensor_options);

  switch (location.device.Type()) {
    case OrtDevice::GPU:
      cudaMemcpy(torch_tensor.data_ptr(), tensor->DataRaw(), tensor->SizeInBytes(), cudaMemcpyDeviceToDevice); break;
      break;
    case OrtDevice::CPU:
      std::memcpy(torch_tensor.data_ptr(), tensor->DataRaw(), tensor->SizeInBytes());
      break;
    default:
      throw std::invalid_argument("Unsupported memory transfer between devices");
  }

  return torch_tensor;
}

void CopyTorchTensorToOrtTensor(const torch::Tensor torch_tensor, Tensor* ort_tensor) {
  void* ort_data = ort_tensor->MutableDataRaw();
  switch (ort_tensor->Location().device.Type()) {
    case OrtDevice::GPU:
      cudaMemcpy(ort_data, torch_tensor.data_ptr(), ort_tensor->SizeInBytes(), cudaMemcpyDeviceToDevice);
      break;
    case OrtDevice::CPU:
      std::memcpy(ort_data, torch_tensor.data_ptr(), ort_tensor->SizeInBytes());
      break;
    default:
      throw std::invalid_argument("Unsupported memory transfer between devices");
  }
}

}  // torch_wrapper
}  // namespace cuda
}  // namespace onnxruntime
