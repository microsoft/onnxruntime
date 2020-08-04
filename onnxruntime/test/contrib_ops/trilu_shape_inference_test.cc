// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape_inference_helper.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(TriluShapeInferenceTests, trilu_triu) {
  ModelProto model;
  OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");

  std::vector<int64_t> dim_values = {7, 16};
  createValueInfo(*graph->add_input(), "X", dim_values);
  graph->set_name("Trilu1");
  auto schema_registry = OpSchemaRegistry::Instance();
  ModelProto& model_ = model;
  shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ValueInfoProto output;
  createValueInfo(output, "Y", dim_values);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

TEST(TriluShapeInferenceTests, trilu_triu3d) {
  ModelProto model;
  OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");

  std::vector<int64_t> dim_values = {3, 8, 16};
  createValueInfo(*graph->add_input(), "X", dim_values);
  graph->set_name("Trilu1");
  auto schema_registry = OpSchemaRegistry::Instance();
  ModelProto& model_ = model;
  shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ValueInfoProto output;
  createValueInfo(output, "Y", dim_values);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

TEST(TriluShapeInferenceTests, trilu_tril) {
  ModelProto model;
  OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  AttributeProto upper;
  upper.set_name("upper");
  upper.set_i(0);
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");
  *node.add_attribute() = upper;

  std::vector<int64_t> dim_values = {7, 16};
  createValueInfo(*graph->add_input(), "X", dim_values);
  graph->set_name("Trilu1");
  auto schema_registry = OpSchemaRegistry::Instance();
  ModelProto& model_ = model;
  shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ValueInfoProto output;
  createValueInfo(output, "Y", dim_values);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

TEST(TriluShapeInferenceTests, trilu_tril3d) {
  ModelProto model;
  OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  AttributeProto upper;
  upper.set_name("upper");
  upper.set_i(0);
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");
  *node.add_attribute() = upper;

  std::vector<int64_t> dim_values = {3, 8, 16};
  createValueInfo(*graph->add_input(), "X", dim_values);
  graph->set_name("Trilu1");
  auto schema_registry = OpSchemaRegistry::Instance();
  ModelProto& model_ = model;
  shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ValueInfoProto output;
  createValueInfo(output, "Y", dim_values);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

}  // namespace test
}  // namespace onnxruntime
