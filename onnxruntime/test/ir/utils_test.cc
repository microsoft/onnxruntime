// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"

using ONNX_NAMESPACE::Utils::DataTypeUtils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(OpUtilsTest, TestPTYPE) {
  DataType p1 = DataTypeUtils::ToType("tensor(int32)");
  DataType p2 = DataTypeUtils::ToType("tensor(int32)");
  DataType p3 = DataTypeUtils::ToType("tensor(int32)");
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(p2, p3);
  EXPECT_EQ(p1, p3);
  DataType p4 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p5 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p6 = DataTypeUtils::ToType("seq(tensor(int32))");
  EXPECT_EQ(p4, p5);
  EXPECT_EQ(p5, p6);
  EXPECT_EQ(p4, p6);

  TypeProto t1 = DataTypeUtils::ToTypeProto(p1);
  EXPECT_TRUE(t1.has_tensor_type());
  EXPECT_TRUE(t1.tensor_type().has_elem_type());
  EXPECT_EQ(t1.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t2 = DataTypeUtils::ToTypeProto(p2);
  EXPECT_TRUE(t2.has_tensor_type());
  EXPECT_TRUE(t2.tensor_type().has_elem_type());
  EXPECT_EQ(t2.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t3 = DataTypeUtils::ToTypeProto(p3);
  EXPECT_TRUE(t3.has_tensor_type());
  EXPECT_TRUE(t3.tensor_type().has_elem_type());
  EXPECT_EQ(t3.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t4 = DataTypeUtils::ToTypeProto(p4);
  EXPECT_TRUE(t4.has_sequence_type());
  EXPECT_TRUE(t4.sequence_type().has_elem_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t4.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t5 = DataTypeUtils::ToTypeProto(p5);
  EXPECT_TRUE(t5.has_sequence_type());
  EXPECT_TRUE(t5.sequence_type().has_elem_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t5.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t6 = DataTypeUtils::ToTypeProto(p6);
  EXPECT_TRUE(t6.has_sequence_type());
  EXPECT_TRUE(t6.sequence_type().has_elem_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t6.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
}

static GraphProto CreateNodeRemovalSubgraph(const std::string& input_name = {}) {
  const std::string add_input_name = input_name.empty() ? "local_add_in_0" : input_name;
  Model model("CreateNodeRemovalSubgraph:" + add_input_name);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  /*
  Create a graph with no inputs (all implicit)

  Add(add_input_name, outer_scope_0) -> add_out_0

  */

  TypeProto float_scalar_tensor;
  float_scalar_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_scalar_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // outer scope values
  auto& outer_scope_0 = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar_tensor);
  graph.AddOuterScopeNodeArg("outer_scope_0");

  {
    // Add
    auto& add_in_0 = graph.GetOrCreateNodeArg(add_input_name, &float_scalar_tensor);
    auto& add_out_0 = graph.GetOrCreateNodeArg("add_out_0", &float_scalar_tensor);

    inputs = {&add_in_0, &outer_scope_0};
    outputs = {&add_out_0};

    graph.AddNode("add", "Add", "Add two inputs.", inputs, outputs);
  }

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& proto = graph.ToGraphProto();
}

TEST(GraphUtils, UpdateSubgraphWhenRemovingNode) {
  /*
  Main Graph

  ?? Can we just have a single Identity node or will that have no edges?

  Worst case:

  Identity (input, graph_input_0, add_
  */

  Model model("SubgraphRemovalTest");
  auto& graph = model.MainGraph();
}

// we can't remove a node if it is used as an implicit input in a subgraph, and changing the implicit input name
// will result with in a clash with an existing node in the subgraph
TEST(GraphUtils, DontRemoveNodeIfItWillBreakSubgraph) {
}
}  // namespace test
}  // namespace onnxruntime
