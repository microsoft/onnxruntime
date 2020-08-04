// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "onnx/shape_inference/implementation.h"
#include <onnx/onnx_pb.h>
#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace test {
inline void createValueInfo2D(
    ONNX_NAMESPACE::ValueInfoProto& value_info,
    const std::string& name,
    int64_t h,
    int64_t w) {
  value_info.set_name(name);
  ONNX_NAMESPACE::TypeProto* type = value_info.mutable_type();
  ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = type->mutable_tensor_type();
  tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ONNX_NAMESPACE::TensorShapeProto* shape = tensor_type->mutable_shape();
  shape->add_dim()->set_dim_value(h);
  shape->add_dim()->set_dim_value(w);
}

void checkShapeEquality(ONNX_NAMESPACE::TensorShapeProto* shape1, ONNX_NAMESPACE::TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  if ((shape1 != nullptr) && (shape2 != nullptr)) {
    EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
    auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
    for (int i = 0; i < min_dims; ++i) {
      auto dim1 = shape1->dim(i);
      auto dim2 = shape2->dim(i);
      EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
      if (dim1.has_dim_value()) {
        EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
      }
      EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
      if (dim1.has_dim_param()) {
        EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
      }
    }
  }
}

TEST(ShapeInferenceTests, trilu_triu) {
  ONNX_NAMESPACE::ModelProto model;
  ONNX_NAMESPACE::OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  ONNX_NAMESPACE::GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");
  createValueInfo2D(*graph->add_input(), "X", 7, 16);
  graph->set_name("Trilu1");
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  ONNX_NAMESPACE::ModelProto& model_ = model;
  ONNX_NAMESPACE::shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo2D(output, "Y", 7, 16);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

TEST(ShapeInferenceTests, trilu_tril) {
  ONNX_NAMESPACE::ModelProto model;
  ONNX_NAMESPACE::OperatorSetIdProto* op_set_id = model.add_opset_import();
  model.set_ir_version(4);
  op_set_id->set_domain("com.microsoft");
  op_set_id->set_version(1);

  ONNX_NAMESPACE::GraphProto* graph = model.mutable_graph();
  auto& node = *graph->add_node();
  ONNX_NAMESPACE::AttributeProto upper;
  upper.set_name("upper");
  upper.set_i(0);
  node.set_name("trilu_op");
  node.set_op_type("Trilu");
  node.set_domain("com.microsoft");
  node.add_input("X");
  node.add_output("Y");
  *node.add_attribute() = upper;

  createValueInfo2D(*graph->add_input(), "X", 7, 16);
  graph->set_name("Trilu1");
  auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();
  ONNX_NAMESPACE::ModelProto& model_ = model;
  ONNX_NAMESPACE::shape_inference::InferShapes(model_, false, schema_registry);
  auto inferedGraph = model_.graph();
  auto infered_output = inferedGraph.value_info();
  ONNX_NAMESPACE::ValueInfoProto output;
  createValueInfo2D(output, "Y", 7, 16);
  EXPECT_EQ(output.name(), infered_output[0].name());
  auto shape1 = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto shape2 = infered_output[0].mutable_type()->mutable_tensor_type()->mutable_shape();
  checkShapeEquality(shape1, shape2);
}

}  // namespace test
}  // namespace onnxruntime