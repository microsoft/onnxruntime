// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"
#include "shape_inference_test_helper.h"

namespace onnxruntime {
namespace test {

void TestShapeInference(
    const std::string& op_type,
    const std::vector<ONNX_NAMESPACE::ValueInfoProto>& inputs,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
    ONNX_NAMESPACE::ValueInfoProto& output) {

#ifndef DISABLE_CONTRIB_OPS
  // test trilu contrib op for maintaining backward compatibility
  TestShapeInference(op_type, kMSDomain, 1, 6, inputs, attributes, output);
#endif

  // test trilu onnx domain op
  TestShapeInference(op_type, kOnnxDomain, 14, 7, inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_float) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_zero_dim_int) {
  std::vector<int64_t> shape = {4, 7, 0};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_upper_4d_long) {
  std::vector<int64_t> shape = {2, 3, 7, 11};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(1);  // upper
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_lower_float) {
  std::vector<int64_t> shape = {4, 7};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(0);  // lower
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_lower_4d_int) {
  std::vector<int64_t> shape = {2, 3, 7, 11};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(0);  // lower
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

TEST(ShapeInferenceTests, tri_lower_zero_dim_long) {
  std::vector<int64_t> shape = {4, 7, 0};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {input};

  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_type(ONNX_NAMESPACE::AttributeProto::INT);
  upper.set_i(0);  // lower
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {upper};

  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);

  TestShapeInference("Trilu", inputs, attributes, output);
}

}  // namespace test
}  // namespace onnxruntime
