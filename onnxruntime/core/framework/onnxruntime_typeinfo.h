// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/onnx_object.h"
#include "core/framework/onnx_object_cxx.h"
#include "core/framework/tensor_type_and_shape_c_api.h"
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
struct ONNXRuntimeTypeInfo : public onnxruntime::ObjectBase<ONNXRuntimeTypeInfo> {
 public:
  friend class onnxruntime::ObjectBase<ONNXRuntimeTypeInfo>;

  ONNXRuntimeType type = ONNXRUNTIME_TYPE_UNKNOWN;
  //owned by this
  void* data = nullptr;
  ONNXRuntimeTypeInfo(const ONNXRuntimeTypeInfo& other) = delete;
  ONNXRuntimeTypeInfo& operator=(const ONNXRuntimeTypeInfo& other) = delete;

  static ONNXStatusPtr FromDataTypeImpl(const onnxruntime::DataTypeImpl* input, const onnxruntime::TensorShape* shape,
                                        const onnxruntime::DataTypeImpl* tensor_data_type, ONNXRuntimeTypeInfo** out);
  static ONNXStatusPtr FromDataTypeImpl(const onnx::TypeProto*, ONNXRuntimeTypeInfo** out);

 private:
  ONNXRuntimeTypeInfo(ONNXRuntimeType type, void* data) noexcept;
  ~ONNXRuntimeTypeInfo();
};
