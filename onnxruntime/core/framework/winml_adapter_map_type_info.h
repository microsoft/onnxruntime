// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../../winml/adapter/winml_adapter_c_apis.h"

struct OrtMapTypeInfo {
 public:
  ONNXTensorElementDataType map_key_type_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  OrtTypeInfo* map_value_type_ = nullptr;
  
  static OrtStatus* FromTypeProto(const ONNX_NAMESPACE::TypeProto*, OrtMapTypeInfo** out); 

 private:
  OrtMapTypeInfo(ONNXTensorElementDataType map_key_type, OrtTypeInfo* map_value_type)noexcept;
  OrtMapTypeInfo(const OrtMapTypeInfo& other) = delete;
  OrtMapTypeInfo& operator=(const OrtMapTypeInfo& other) = delete;
};
