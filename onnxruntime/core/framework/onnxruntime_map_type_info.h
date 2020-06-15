// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

#include <memory>

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtMapTypeInfo {
 public:
  ONNXTensorElementDataType map_key_type_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  std::unique_ptr<OrtTypeInfo, decltype(OrtApi::ReleaseTypeInfo)> map_value_type_;

  static OrtStatus* FromTypeProto(const ONNX_NAMESPACE::TypeProto*, OrtMapTypeInfo** out); 

  OrtStatus* Clone(OrtMapTypeInfo** out);

 private:
  OrtMapTypeInfo(ONNXTensorElementDataType map_key_type, OrtTypeInfo* map_value_type)noexcept;
  OrtMapTypeInfo(const OrtMapTypeInfo& other) = delete;
  OrtMapTypeInfo& operator=(const OrtMapTypeInfo& other) = delete;

};
