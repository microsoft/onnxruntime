// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"
#include "test/providers/cpu/tensor/shape_inference_test_helper.h"

namespace onnxruntime {
namespace test {

TEST(ShapeInferenceTests, bernoulli_float_2D) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  TestShapeInference("Bernoulli", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, bernoulli_double_2d) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, shape);

  TestShapeInference("Bernoulli", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, bernoulli_float_4D) {
  std::vector<int64_t> shape = {2, 3, 4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  TestShapeInference("Bernoulli", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, bernoulli_double_4d) {
  std::vector<int64_t> shape = {2, 3, 4, 7};;
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_DOUBLE, shape);

  TestShapeInference("Bernoulli", kMSDomain, 1, 6, inputs, attributes, output);
}

}  // namespace test
}  // namespace onnxruntime
