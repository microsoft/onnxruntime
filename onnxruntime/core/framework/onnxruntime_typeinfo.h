// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/onnx_object_cxx.h"
#include <atomic>

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
struct OrtTypeInfo : public onnxruntime::ObjectBase<OrtTypeInfo> {
 public:
  friend class onnxruntime::ObjectBase<OrtTypeInfo>;

  ONNXType type = ONNX_TYPE_UNKNOWN;
  //owned by this
  void* data = nullptr;
  OrtTypeInfo(const OrtTypeInfo& other) = delete;
  OrtTypeInfo& operator=(const OrtTypeInfo& other) = delete;

  static OrtStatus* FromDataTypeImpl(const onnxruntime::DataTypeImpl* input, const onnxruntime::TensorShape* shape,
                                      const onnxruntime::DataTypeImpl* tensor_data_type, OrtTypeInfo** out);
  static OrtStatus* FromDataTypeImpl(const onnx::TypeProto*, OrtTypeInfo** out);

 private:
  OrtTypeInfo(ONNXType type, void* data) noexcept;
  ~OrtTypeInfo();
};
