// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtMapTypeInfo::OrtMapTypeInfo(ONNXTensorElementDataType map_key_type, OrtTypeInfo* map_value_type) noexcept : map_key_type_(map_key_type), map_value_type_(map_value_type, &OrtApis::ReleaseTypeInfo) {  
}

static ONNXTensorElementDataType
ToONNXTensorElementDataType(ONNX_NAMESPACE::TensorProto_DataType data_type) {
  using TensorType = ONNX_NAMESPACE::TensorProto_DataType;
  switch (data_type) {
    case TensorType::TensorProto_DataType_BOOL:            { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;         }
    case TensorType::TensorProto_DataType_STRING:          { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;       }   // maps to c++ type std::string
    case TensorType::TensorProto_DataType_FLOAT16:         { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;      }   // maps to c type float16
    case TensorType::TensorProto_DataType_FLOAT:           { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;        }   // maps to c type float
    case TensorType::TensorProto_DataType_DOUBLE:          { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;       }   // maps to c type double
    case TensorType::TensorProto_DataType_INT8:            { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;         }   // maps to c type int8_t
    case TensorType::TensorProto_DataType_INT16:           { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;        }   // maps to c type int16_t
    case TensorType::TensorProto_DataType_INT32:           { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;        }   // maps to c type int32_t
    case TensorType::TensorProto_DataType_INT64:           { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;        }   // maps to c type int64_t
    case TensorType::TensorProto_DataType_UINT8:           { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;        }   // maps to c type uint8_t
    case TensorType::TensorProto_DataType_UINT16:          { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;       }   // maps to c type uint16_t
    case TensorType::TensorProto_DataType_UINT32:          { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;       }   // maps to c type uint32_t
    case TensorType::TensorProto_DataType_UINT64:          { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;       }   // maps to c type uint64_t
    case TensorType::TensorProto_DataType_COMPLEX64:       { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64;    }   // complex with float32 real and imaginary components
    case TensorType::TensorProto_DataType_COMPLEX128:      { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128;   }   // complex with float64 real and imaginary components
    case TensorType::TensorProto_DataType_BFLOAT16:        { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;     }   // Non-IEEE floating-point format based on IEEE754 single-precision
    default:                                               { return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;    }
  }
}

OrtStatus* OrtMapTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto* type_proto, OrtMapTypeInfo** out) {
  auto value_case = type_proto->value_case();
  if (value_case != ONNX_NAMESPACE::TypeProto::kMapType)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "type_proto is not of type map!");;
  }

  // Get the key type of the map
  auto type_proto_map = type_proto->map_type();
  auto map_key_type = ToONNXTensorElementDataType(ONNX_NAMESPACE::TensorProto_DataType(type_proto_map.key_type()));

  // Get the value type of the map
  OrtTypeInfo* map_value_type_info = nullptr;
  if (auto status = OrtTypeInfo::FromTypeProto(&type_proto_map.value_type(), &map_value_type_info))
  {
    return status;
  }

  *out = new OrtMapTypeInfo(map_key_type, map_value_type_info);
  return nullptr;
}

OrtStatus* OrtMapTypeInfo::Clone(OrtMapTypeInfo** out) {
  OrtTypeInfo* map_value_type_copy = nullptr;
  if (auto status = map_value_type_->Clone(&map_value_type_copy))
  {
    return status;
  }
  *out = new OrtMapTypeInfo(map_key_type_, map_value_type_copy);
  return nullptr;
}

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(OrtApis::GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info,
                    _Out_ enum ONNXTensorElementDataType* out) {
  API_IMPL_BEGIN
  *out = map_type_info->map_key_type_;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** out) {
  API_IMPL_BEGIN
  return map_type_info->map_value_type_->Clone(out);
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseMapTypeInfo, _Frees_ptr_opt_ OrtMapTypeInfo* ptr) {
  delete ptr;
}