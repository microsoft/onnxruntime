// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_EP_API_ADAPTER_HEADER_INCLUDED)
#error "This header should not be included directly. Include ep/adapters.h instead."
#endif

#include <memory>
#include <sstream>

#include "allocator.h"
#include "core/framework/tensor.h"
#include "core/session/onnxruntime_type_conversion.h"

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
  MLDataType data_type = DataTypeImpl::TensorTypeFromONNXEnum(
                             utils::ToTensorProtoElementType(element_type))
                             ->GetElementType();

  const auto tensor_memory_info = CreateMemoryInfoFromApi(memory_info);

  return Tensor(data_type,
                TensorShape{shape.shape, shape.shape_len},
                value.GetTensorMutableRawData(),
                tensor_memory_info);
}

}  // namespace adapter
}  // namespace ep
}  // namespace onnxruntime
