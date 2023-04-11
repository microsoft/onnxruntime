// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtMapTypeInfo::OrtMapTypeInfo(ONNXTensorElementDataType map_key_type,
                               std::unique_ptr<OrtTypeInfo> map_value_type) noexcept
    : map_key_type_(map_key_type), map_value_type_(std::move(map_value_type)) {
}

OrtMapTypeInfo::~OrtMapTypeInfo() = default;

static ONNXTensorElementDataType
ToONNXTensorElementDataType(ONNX_NAMESPACE::TensorProto_DataType data_type) {
  using TensorType = ONNX_NAMESPACE::TensorProto_DataType;
  switch (data_type) {
    case TensorType::TensorProto_DataType_BOOL: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    case TensorType::TensorProto_DataType_STRING: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    }  // maps to c++ type std::string
    case TensorType::TensorProto_DataType_FLOAT16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }  // maps to c type float16
    case TensorType::TensorProto_DataType_FLOAT: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }  // maps to c type float
    case TensorType::TensorProto_DataType_DOUBLE: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    }  // maps to c type double
    case TensorType::TensorProto_DataType_INT8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }  // maps to c type int8_t
    case TensorType::TensorProto_DataType_INT16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    }  // maps to c type int16_t
    case TensorType::TensorProto_DataType_INT32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }  // maps to c type int32_t
    case TensorType::TensorProto_DataType_INT64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }  // maps to c type int64_t
    case TensorType::TensorProto_DataType_UINT8: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }  // maps to c type uint8_t
    case TensorType::TensorProto_DataType_UINT16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }  // maps to c type uint16_t
    case TensorType::TensorProto_DataType_UINT32: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    }  // maps to c type uint32_t
    case TensorType::TensorProto_DataType_UINT64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    }  // maps to c type uint64_t
    case TensorType::TensorProto_DataType_COMPLEX64: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;
    }  // complex with float32 real and imaginary components
    case TensorType::TensorProto_DataType_COMPLEX128: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;
    }  // complex with float64 real and imaginary components
    case TensorType::TensorProto_DataType_BFLOAT16: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    }  // Non-IEEE floating-point format based on IEEE754 single-precision
    default: {
      return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  }
}

std::unique_ptr<OrtMapTypeInfo> OrtMapTypeInfo::FromTypeProto(
    const ONNX_NAMESPACE::TypeProto& type_proto) {
  auto value_case = type_proto.value_case();

  ORT_ENFORCE(value_case == ONNX_NAMESPACE::TypeProto::kMapType, "type_proto is not of type map!");

  // Get the key type of the map
  const auto& type_proto_map = type_proto.map_type();
  const auto map_key_type = ToONNXTensorElementDataType(
      ONNX_NAMESPACE::TensorProto_DataType(type_proto_map.key_type()));

  // Get the value type of the map
  auto map_value_type_info = OrtTypeInfo::FromTypeProto(type_proto_map.value_type());

  return std::make_unique<OrtMapTypeInfo>(map_key_type, std::move(map_value_type_info));
}

std::unique_ptr<OrtMapTypeInfo> OrtMapTypeInfo::Clone() const {
  auto map_value_type_copy = map_value_type_->Clone();
  return std::make_unique<OrtMapTypeInfo>(map_key_type_, std::move(map_value_type_copy));
}

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(OrtApis::GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info,
                    _Out_ enum ONNXTensorElementDataType* out) {
  API_IMPL_BEGIN
  *out = map_type_info->map_key_type_;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetMapValueType,
                    _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto clone = map_type_info->map_value_type_->Clone();
  *out = clone.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseMapTypeInfo, _Frees_ptr_opt_ OrtMapTypeInfo* ptr) {
  std::unique_ptr<OrtMapTypeInfo> p(ptr);
}