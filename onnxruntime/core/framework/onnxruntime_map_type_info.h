// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "core/graph/onnx_protobuf.h"
#include "onnxruntime_c_api.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtTypeInfo;

namespace onnxruntime {
namespace type_info_internal {

constexpr ONNXTensorElementDataType ToONNXTensorElementDataType(
    ONNX_NAMESPACE::TensorProto_DataType data_type) noexcept {
  const int value = static_cast<int>(data_type);
  if (value < static_cast<int>(ONNX_NAMESPACE::TensorProto_DataType_DataType_MIN) ||
      value > static_cast<int>(ONNX_NAMESPACE::TensorProto_DataType_DataType_MAX)) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }

  // The enums have the same values through FLOAT4E2M1. Their final three entries
  // differ because ONNX inserted FLOAT8E8M0 before UINT2 and INT2.
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2;
    case ONNX_NAMESPACE::TensorProto_DataType_INT2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2;
    default:
      return static_cast<ONNXTensorElementDataType>(value);
  }
}

consteval bool IsTensorProtoToOrtElementTypeMapBijective() {
  constexpr size_t ort_element_type_count =
      static_cast<size_t>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0) + 1;
  constexpr size_t onnx_element_type_count = ONNX_NAMESPACE::TensorProto_DataType_DataType_ARRAYSIZE;
  if constexpr (onnx_element_type_count != ort_element_type_count) {
    return false;
  }

  std::array<bool, ort_element_type_count> mapped_types{};
  for (size_t index = 0; index < onnx_element_type_count; ++index) {
    const auto mapped_type = ToONNXTensorElementDataType(
        static_cast<ONNX_NAMESPACE::TensorProto_DataType>(index));
    const int mapped_index = static_cast<int>(mapped_type);
    if (mapped_index < 0 || static_cast<size_t>(mapped_index) >= mapped_types.size() ||
        mapped_types[static_cast<size_t>(mapped_index)]) {
      return false;
    }

    mapped_types[static_cast<size_t>(mapped_index)] = true;
  }

  return true;
}

static_assert(IsTensorProtoToOrtElementTypeMapBijective(),
              "ONNX and ORT tensor element type enums must have a complete one-to-one mapping.");

}  // namespace type_info_internal
}  // namespace onnxruntime

struct OrtMapTypeInfo {
 public:
  ONNXTensorElementDataType map_key_type_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  std::unique_ptr<OrtTypeInfo> map_value_type_;

  static std::unique_ptr<OrtMapTypeInfo> FromTypeProto(const ONNX_NAMESPACE::TypeProto&);

  std::unique_ptr<OrtMapTypeInfo> Clone() const;

  OrtMapTypeInfo(ONNXTensorElementDataType map_key_type, std::unique_ptr<OrtTypeInfo> map_value_type) noexcept;
  ~OrtMapTypeInfo();

  OrtMapTypeInfo(const OrtMapTypeInfo& other) = delete;
  OrtMapTypeInfo& operator=(const OrtMapTypeInfo& other) = delete;
};
