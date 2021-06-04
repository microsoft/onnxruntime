// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"
#include "test/providers/cpu/tensor/shape_inference_test_helper.h"

namespace onnxruntime {
namespace test {
using namespace ONNX_NAMESPACE;

TEST(ShapeInferenceTests, optional_empty_tensor) {
  ONNX_NAMESPACE::AttributeProto attrProto;
  attrProto.set_name("type");
  attrProto.set_type(ONNX_NAMESPACE::AttributeProto::TYPE_PROTO);
  attrProto.mutable_tp()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {attrProto};

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* tensor_type = output.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TestShapeInference("OptionalEmpty", kMSDomain, 1, 6, {}, attributes, output);
}

TEST(ShapeInferenceTests, optional_empty_sequence) {
  ONNX_NAMESPACE::AttributeProto attrProto;
  attrProto.set_name("type");
  attrProto.set_type(ONNX_NAMESPACE::AttributeProto::TYPE_PROTO);
  attrProto.mutable_tp()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {attrProto};

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* sequence_type = output.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_sequence_type();
  auto* tensor_type = sequence_type->mutable_elem_type()->mutable_tensor_type();  
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TestShapeInference("OptionalEmpty", kMSDomain, 1, 6, {}, attributes, output);
}

TEST(TriluContribOpTest, optional_construct_tensor) {
  ONNX_NAMESPACE::ValueInfoProto input;
  input.set_name("X");
  auto* tensor_type = input.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::vector<int64_t> shape = {2, 3};
  auto* value_info_shape = tensor_type->mutable_shape();
  for (int64_t dim_value : shape) {
    value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* output_type = output.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_value_info_shape = output_type->mutable_shape();
  for (int64_t dim_value : shape) {
    output_value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  TestShapeInference("OptionalConstruct", kMSDomain, 1, 6, {input}, {}, output);
}

TEST(TriluContribOpTest, optional_construct_sequence) {
  ONNX_NAMESPACE::ValueInfoProto input;
  input.set_name("X");
  auto* tensor_type = input.mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::vector<int64_t> shape = {2, 3};
  auto* value_info_shape = tensor_type->mutable_shape();
  for (int64_t dim_value : shape) {
    value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* output_type = output.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_value_info_shape = output_type->mutable_shape();
  for (int64_t dim_value : shape) {
    output_value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  TestShapeInference("OptionalConstruct", kMSDomain, 1, 6, {input}, {}, output);
}

TEST(TriluContribOpTest, optional_has_element) {
  ONNX_NAMESPACE::ValueInfoProto input;
  input.set_name("X");
  auto* input_type = input.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* output_type = output.mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  TestShapeInference("OptionalHasElement", kMSDomain, 1, 6, {input}, {}, output);
}

TEST(TriluContribOpTest, optional_get_tensor) {
  ONNX_NAMESPACE::ValueInfoProto input;
  input.set_name("X");
  auto* input_type = input.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* input_value_info_shape = input_type->mutable_shape();
  std::vector<int64_t> shape = {2, 3};
  for (int64_t dim_value : shape) {
    input_value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* tensor_type = output.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* value_info_shape = tensor_type->mutable_shape();
  for (int64_t dim_value : shape) {
    value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  TestShapeInference("OptionalGetElement", kMSDomain, 1, 6, {input}, {}, output);
}

TEST(TriluContribOpTest, optional_get_sequence) {
  ONNX_NAMESPACE::ValueInfoProto input;
  input.set_name("X");
  auto* tensor_type = input.mutable_type()->mutable_optional_type()->mutable_elem_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  std::vector<int64_t> shape = {2, 3};
  auto* value_info_shape = tensor_type->mutable_shape();
  for (int64_t dim_value : shape) {
    value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  ONNX_NAMESPACE::ValueInfoProto output;
  output.set_name("Y");
  auto* output_type = output.mutable_type()->mutable_sequence_type()->mutable_elem_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto* output_value_info_shape = output_type->mutable_shape();
  for (int64_t dim_value : shape) {
    output_value_info_shape->add_dim()->set_dim_value(dim_value);
  }

  TestShapeInference("OptionalGetElement", kMSDomain, 1, 6, {input}, {}, output);
}

}  // namespace test
}  // namespace onnxruntime
