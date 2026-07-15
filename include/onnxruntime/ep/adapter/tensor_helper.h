// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>
#include <sstream>

#include "core/framework/tensor.h"

namespace onnxruntime {
namespace ep {
namespace adapter {

/// <summary>
/// Create an unowned onnxruntime::Tensor from a tensor OrtValue from C API.
/// </summary>
inline onnxruntime::Tensor CreateTensorFromApiValue(OrtValue* ort_value) {
  Ort::UnownedValue value{ort_value};
  EP_ENFORCE(value.IsTensor(), "Only tensor OrtValue is supported.");

  ONNXTensorElementDataType element_type;
  Ort::Value::Shape shape{};
  value.GetTensorElementTypeAndShapeDataReference(element_type, shape);

  auto memory_info = value.GetTensorMemoryInfo();
  MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(element_type)->GetElementType();

  OrtMemoryInfo tensor_memory_info{memory_info.GetAllocatorName(),
                                   memory_info.GetAllocatorType(),
                                   OrtDevice{
                                       static_cast<OrtDevice::DeviceType>(memory_info.GetDeviceType()),
                                       static_cast<OrtDevice::MemoryType>(memory_info.GetMemoryType()),
                                       static_cast<OrtDevice::VendorId>(memory_info.GetVendorId()),
                                       static_cast<OrtDevice::DeviceId>(memory_info.GetDeviceId()),

                                   },
                                   memory_info.GetMemoryType()};

  return Tensor(data_type,
                TensorShape{shape.shape, shape.shape_len},
                value.GetTensorMutableRawData(),
                tensor_memory_info);
}

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
