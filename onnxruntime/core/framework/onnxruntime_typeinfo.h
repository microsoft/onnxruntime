// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class DataTypeImpl;
class TensorShape;
}  // namespace onnxruntime

namespace onnx {
class TypeProto;
}

/**
 * the equivalent of onnx::TypeProto
 * This class is mainly for the C API
 */
struct OrtTypeInfo {
 public:
  ONNXType type = ONNX_TYPE_UNKNOWN;

  ~OrtTypeInfo();

  //owned by this
  OrtTensorTypeAndShapeInfo* data = nullptr;
  OrtTypeInfo(const OrtTypeInfo& other) = delete;
  OrtTypeInfo& operator=(const OrtTypeInfo& other) = delete;

  static OrtStatus* FromDataTypeImpl(const onnxruntime::DataTypeImpl* input, const onnxruntime::TensorShape* shape,
                                     const onnxruntime::DataTypeImpl* tensor_data_type, OrtTypeInfo** out);
  static OrtStatus* FromDataTypeImpl(const onnx::TypeProto*, OrtTypeInfo** out);

 private:
  OrtTypeInfo(ONNXType type, OrtTensorTypeAndShapeInfo* data) noexcept;
};
