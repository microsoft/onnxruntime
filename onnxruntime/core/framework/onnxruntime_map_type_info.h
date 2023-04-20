// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

#include <memory>

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtTypeInfo;

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
