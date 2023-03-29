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
  using Ptr = std::unique_ptr<OrtTensorTypeAndShapeInfo>;

  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;
  // dim_param values. empty string if dim_value or no dim_param was specified.
  // one entry per dimension in shape. only guaranteed to be populated for graph inputs and outputs
  std::vector<std::string> dim_params;

  OrtTensorTypeAndShapeInfo() = default;
  ~OrtTensorTypeAndShapeInfo();

  Ptr Clone() const {
    return std::make_unique<OrtTensorTypeAndShapeInfo>(*this);
  }

  OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo& other) = default;
  OrtTensorTypeAndShapeInfo& operator=(const OrtTensorTypeAndShapeInfo& other) = default;

  // Utils
  static Ptr GetTensorShapeAndTypeHelper(ONNXTensorElementDataType type, onnxruntime::TensorShape shape,
                                         const std::vector<std::string>* dim_params);

  static Ptr GetTensorShapeAndType(onnxruntime::TensorShape shape,
                                   const onnxruntime::DataTypeImpl& tensor_data_type);

  static Ptr GetTensorShapeAndType(onnxruntime::TensorShape shape, const std::vector<std::string>* dim_params,
                                   const ONNX_NAMESPACE::TypeProto&);
};

constexpr ONNXTensorElementDataType TensorDataTypeToOnnxRuntimeTensorElementDataType(int32_t dtype);
