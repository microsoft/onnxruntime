// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

struct OrtTensorTypeAndShapeInfo {
 public:
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  onnxruntime::TensorShape shape;

  OrtTensorTypeAndShapeInfo() = default;
  OrtTensorTypeAndShapeInfo(const OrtTensorTypeAndShapeInfo& other) = delete;
  OrtTensorTypeAndShapeInfo& operator=(const OrtTensorTypeAndShapeInfo& other) = delete;
};
