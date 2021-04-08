// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"

namespace onnxruntime {
namespace test {

static auto schema_registry = ONNX_NAMESPACE::OpSchemaRegistry::Instance();

inline void CheckShapeEquality(ONNX_NAMESPACE::TensorShapeProto* shape1, ONNX_NAMESPACE::TensorShapeProto* shape2) {
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

inline void CreateValueInfo(
    ONNX_NAMESPACE::ValueInfoProto& value_info,
    const std::string& name,
    const ONNX_NAMESPACE::TensorProto_DataType& elem_type,
    const std::vector<int64_t> shape) {
  value_info.set_name(name);
  ONNX_NAMESPACE::TypeProto* type = value_info.mutable_type();
  ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = type->mutable_tensor_type();
  tensor_type->set_elem_type(elem_type);
  ONNX_NAMESPACE::TensorShapeProto* value_info_shape = tensor_type->mutable_shape();

  for (int64_t dim_value : shape) {
    value_info_shape->add_dim()->set_dim_value(dim_value);
  }
}

inline void TestShapeInference(const std::string& op_type,
                               const std::string& op_domain,
                               int op_version,
                               int ir_version,
                               const std::vector<ONNX_NAMESPACE::ValueInfoProto>& inputs,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
                               ONNX_NAMESPACE::ValueInfoProto& output) {
  ONNX_NAMESPACE::ModelProto model;
  // Set opset (domain + version)
  ONNX_NAMESPACE::OperatorSetIdProto* op_set_id = model.add_opset_import();
  op_set_id->set_domain(op_domain);
  op_set_id->set_version(op_version);
  model.set_ir_version(ir_version);
  model.set_producer_name("onnx");

  // Set model graph
  ONNX_NAMESPACE::GraphProto* graph = model.mutable_graph();
  graph->set_name("test-op");

  // Set add operator node to graph
  auto node = graph->add_node();
  node->set_op_type(op_type);
  node->set_domain(op_domain);
  node->set_name("test_node");

  // Add node inputs and graph inputs
	for (auto const& n_ : inputs) {
	  node->add_input(n_.name());
	  auto in = graph->add_input();
	  *in = n_;
	  auto v_ = graph->add_value_info();
	  *v_ = n_;
	}

  // Add node attributes
  for (auto const& attr : attributes) {
    node->add_attribute()->CopyFrom(attr);
  }

  node->add_output("Output");

  ONNX_NAMESPACE::shape_inference::InferShapes(model, true, schema_registry);
  ONNX_NAMESPACE::checker::check_model(model);

  auto inferredGraph = model.graph();
  int index = static_cast<int>(inputs.size());  // index for value_info of output
  auto inferred_output = inferredGraph.value_info(index);

  auto elem_type = output.mutable_type()->mutable_tensor_type()->elem_type();
  auto inferred_elem_type = inferred_output.mutable_type()->mutable_tensor_type()->elem_type();
  EXPECT_EQ(elem_type, inferred_elem_type);

  auto shape = output.mutable_type()->mutable_tensor_type()->mutable_shape();
  auto inferred_shape = inferred_output.mutable_type()->mutable_tensor_type()->mutable_shape();
  CheckShapeEquality(shape, inferred_shape);
}
}  // namespace test
}  // namespace onnxruntime