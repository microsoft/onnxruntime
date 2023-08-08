// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/framework/tensor_shape.h"
#include "core/session/onnxruntime_c_api.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

namespace onnxruntime {
class DataTypeImpl;
}

struct OrtTensorTypeAndShapeInfo {
 public:
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;
  // dim_param values. empty string if dim_value or no dim_param was specified.
  // one entry per dimension in shape. only guaranteed to be populated for graph inputs and outputs
  std::vector<std::string> dim_params;

  OrtTensorTypeAndShapeInfo();
  ~OrtTensorTypeAndShapeInfo();

  // Utils
  static std::unique_ptr<OrtTensorTypeAndShapeInfo> GetTensorShapeAndTypeHelper(
      ONNXTensorElementDataType type,
      onnxruntime::TensorShape shape,
      const std::vector<std::string>* dim_params);

  static std::unique_ptr<OrtTensorTypeAndShapeInfo> GetTensorShapeAndType(
      onnxruntime::TensorShape shape,
      const onnxruntime::DataTypeImpl& tensor_data_type);

  static std::unique_ptr<OrtTensorTypeAndShapeInfo> GetTensorShapeAndType(
      onnxruntime::TensorShape shape,
      const std::vector<std::string>* dim_params,
      const ONNX_NAMESPACE::TypeProto&);

  // We provide Clone() here to satisfy the existing coding pattern
  // as we need copies made on the heap even though we achieve that
  // via a copy __ctor which can not be made private due to make_unique
  // which is a requirement.
  std::unique_ptr<OrtTensorTypeAndShapeInfo> Clone() const {
    return std::make_unique<OrtTensorTypeAndShapeInfo>(*this);
  }

  // Copy ops are public because std::make_unique above requires them to be accessible
  OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo& other);
  OrtTensorTypeAndShapeInfo& operator=(const OrtTensorTypeAndShapeInfo& other);
};

constexpr ONNXTensorElementDataType TensorDataTypeToOnnxRuntimeTensorElementDataType(int32_t dtype);
