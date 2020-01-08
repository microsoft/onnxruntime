// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include "core/session/onnxruntime_c_api.h"
#include "../winml/adapter/winml_adapter_c_api.h"

namespace onnxruntime {
class DataTypeImpl;
class TensorShape;
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {
class TypeProto;
}

/**
 * the equivalent of ONNX_NAMESPACE::TypeProto
 * This class is mainly for the C API
 */
struct OrtTypeInfo {
 public:
  ONNXType type = ONNX_TYPE_UNKNOWN;

  ~OrtTypeInfo();

  //owned by this
  OrtTensorTypeAndShapeInfo* data = nullptr;
  OrtMapTypeInfo* map_type_info_ = nullptr;
  OrtSequenceTypeInfo* sequence_type_info_ = nullptr;
  OrtTypeInfo(const OrtTypeInfo& other) = delete;
  OrtTypeInfo& operator=(const OrtTypeInfo& other) = delete;

  static OrtStatus* FromOrtValue(const OrtValue& value, OrtTypeInfo** out);
  static OrtStatus* FromTypeProto(const ONNX_NAMESPACE::TypeProto*, OrtTypeInfo** out);

  static const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

 private:
  OrtTypeInfo(ONNXType type) noexcept;
  OrtTypeInfo(ONNXType type, OrtTensorTypeAndShapeInfo* data) noexcept;
  OrtTypeInfo(ONNXType type, OrtMapTypeInfo* map_type_info) noexcept;
  OrtTypeInfo(ONNXType type, OrtSequenceTypeInfo* sequence_type_info) noexcept;
};
