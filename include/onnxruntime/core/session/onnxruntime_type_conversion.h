// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"

namespace onnxruntime::utils {

// C API values are ABI-stable and differ from ONNX's serialized type numbers
// after FLOAT4E2M1. Use wire numbers here so shared providers and standalone
// plugins can use these conversions without depending on ONNX protobuf headers.
constexpr int ToTensorProtoElementType(ONNXTensorElementDataType type) noexcept {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0:
      return 24;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2:
      return 25;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2:
      return 26;
    default:
      return static_cast<int>(type);
  }
}

constexpr ONNXTensorElementDataType ToOrtTensorElementDataType(int tensor_proto_element_type) noexcept {
  switch (tensor_proto_element_type) {
    case 24:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0;
    case 25:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2;
    case 26:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2;
    default:
      if (tensor_proto_element_type < 0 ||
          tensor_proto_element_type > static_cast<int>(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0)) {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      }
      return static_cast<ONNXTensorElementDataType>(tensor_proto_element_type);
  }
}

}  // namespace onnxruntime::utils
