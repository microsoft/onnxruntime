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

  using Ptr = std::unique_ptr<OrtMapTypeInfo>;

  ONNXTensorElementDataType map_key_type_ = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  std::unique_ptr<OrtTypeInfo> map_value_type_;

  static Ptr FromTypeProto(const ONNX_NAMESPACE::TypeProto&); 

  Ptr Clone() const;

  OrtMapTypeInfo(ONNXTensorElementDataType map_key_type, std::unique_ptr<OrtTypeInfo> map_value_type) noexcept;
  ~OrtMapTypeInfo();

 private:
  OrtMapTypeInfo(const OrtMapTypeInfo& other) = delete;
  OrtMapTypeInfo& operator=(const OrtMapTypeInfo& other) = delete;

};
