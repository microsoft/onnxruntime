// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"

#include "test/test_environment.h"

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

/* Create a subgraph for testing node removal
@param new_output_name Provide the value that will become the new output name after node removal if you want this to
                       be made to clash with an existing name in the subgraph to test that node removal is not done.
@param add_second_level Add another level of subgraph so the recursion is tested
@param nested_new_output_name Same usage as new_output_name but will be applied to the second level of subgraph if
                              add_second_level is true.
*/
static GraphProto CreateNodeRemovalSubgraph(const std::string& new_output_name = {},
                                            bool add_second_level = false,
                                            const std::string& nested_new_output_name = {}) {
  std::string suffix = add_second_level ? ".top" : ".bottom";
  std::string constant_output_name = (new_output_name.empty() ? "constant_in_0" + suffix : new_output_name);

  Model model("CreateNodeRemovalSubgraph:" + constant_output_name, false,
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_scalar_tensor;
  float_scalar_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_scalar_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Constant that will get lifted to an initializer but leave a NodeArg that matches constant_output_name
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto& local_constant = graph.GetOrCreateNodeArg(constant_output_name, &float_scalar_tensor);
  graph.AddNode("local_constant", "Constant", "Local constant", {}, {&local_constant})
      .AddAttribute("value", value_tensor);

  // outer scope value which is the implicit input, and the output of the node to be removed
  auto& outer_scope_0 = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar_tensor);
  graph.AddOuterScopeNodeArg("outer_scope_0");

  auto& add_out_0 = graph.GetOrCreateNodeArg("add_out_0" + suffix, &float_scalar_tensor);
  graph.AddNode("add", "Add", "Add two inputs.", {&local_constant, &outer_scope_0}, {&add_out_0});

  if (add_second_level) {
    TypeProto bool_scalar_tensor;
    bool_scalar_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
    bool_scalar_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

    // create two nested subgraphs - one where there may be a clash with the implicit input name.
    // these use the same implicit input in the If node as this graph
    GraphProto then_branch = CreateNodeRemovalSubgraph(nested_new_output_name);
    GraphProto else_branch = CreateNodeRemovalSubgraph();

    auto& cond_in = graph.GetOrCreateNodeArg("cast_add_output_to_bool", &bool_scalar_tensor);
    auto& cast_node = graph.AddNode("cast", "Cast", "Cast Add output to bool to use as cond in subgraph",
                                    {&add_out_0}, {&cond_in});
    cast_node.AddAttribute("to", static_cast<int64_t>(TensorProto_DataType_BOOL));

    auto& nested_if_out_0 = graph.GetOrCreateNodeArg("nested_if_out_0", &float_scalar_tensor);

    auto& if_node = graph.AddNode("nested_if", "If", "Nested If node", {&cond_in}, {&nested_if_out_0});

    if_node.AddAttribute("then_branch", then_branch);
    if_node.AddAttribute("else_branch", else_branch);
  }

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto& proto = graph.ToGraphProto();
  return proto;
}

static void CreateNodeRemovalGraph(Model& model, bool removal_allowed, bool test_nested) {
  /*
  Main Graph.
  Graph input -> id_0 -> node_to_remove -> if
  The If node uses the output of node_to_remove as an implicit input. If removal is allowed, the subgraph should
  get updated to use id_0 output as an implicit input. If removal is not allowed because updating to a different
  implicit input would clash with a local name in the subgraph, the node should not be removed.

  Optionally a second level of subgraph can be created inside the 'If' node to test recursion into that.
  */

  auto& graph = model.MainGraph();

  TypeProto float_scalar_tensor;
  float_scalar_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_scalar_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_scalar_tensor;
  bool_scalar_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& graph_in_0 = graph.GetOrCreateNodeArg("graph_in_0", &float_scalar_tensor);
  auto& graph_in_1 = graph.GetOrCreateNodeArg("graph_in_1", &bool_scalar_tensor);

  const std::string id_out_0_name = "id_out_0";
  auto& id_out_0 = graph.GetOrCreateNodeArg(id_out_0_name, &float_scalar_tensor);
  auto& id_out_1 = graph.GetOrCreateNodeArg("outer_scope_0", &float_scalar_tensor);
  auto& if_out_0 = graph.GetOrCreateNodeArg("if_out_1", &float_scalar_tensor);

  graph.AddNode("id_0", "Identity", "Graph input id node", {&graph_in_0}, {&id_out_0});
  graph.AddNode("node_to_remove", "Identity", "Node to remove with implicit input to subgraph",
                {&id_out_0}, {&id_out_1});

  // create two subgraphs - one where there will be a clash with the implicit input name if removal_allowed is false.
  GraphProto then_branch;
  GraphProto else_branch;
  if (test_nested) {
    // if we're testing a nested subgraph set the name that clashes in the lower level
    then_branch = CreateNodeRemovalSubgraph("", true, removal_allowed ? "" : id_out_0_name);
    else_branch = CreateNodeRemovalSubgraph("", true);

  } else {
    then_branch = CreateNodeRemovalSubgraph(removal_allowed ? "" : id_out_0_name);
    else_branch = CreateNodeRemovalSubgraph();
  }

  // implicit inputs, one output
  auto& if_node = graph.AddNode("if", "If", "If node", {&graph_in_1}, {&if_out_0});
  if_node.AddAttribute("then_branch", then_branch);
  if_node.AddAttribute("else_branch", else_branch);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

static void CheckNodeRemovalSubgraphUpdate(const std::string& new_name, const Graph& subgraph) {
  const auto& nodes = subgraph.Nodes();
  // the second input to the Add node is the implicit input that should have been updated
  const auto& add_node = *std::find_if(nodes.cbegin(), nodes.cend(),
                                       [](const Node& node) { return node.OpType() == "Add"; });

  const auto& outer_scope_name = add_node.InputDefs()[1]->Name();
  EXPECT_TRUE(outer_scope_name == new_name);

  // if we have a nested subgraph, we have another If node
  auto if_node_iter = std::find_if(nodes.cbegin(), nodes.cend(),
                                   [](const Node& node) { return node.OpType() == "If"; });
  if (if_node_iter != nodes.cend()) {
    CheckNodeRemovalSubgraphUpdate(new_name, *if_node_iter->GetGraphAttribute("then_branch"));
    CheckNodeRemovalSubgraphUpdate(new_name, *if_node_iter->GetGraphAttribute("else_branch"));
  }
}

static void UpdateSubgraphWhenRemovingNode(bool include_nested = false) {
  Model model(std::string("UpdateSubgraphWhenRemovingNode") + (include_nested ? ":Nested" : ":SingleLevel"),
              false, DefaultLoggingManager().DefaultLogger());

  CreateNodeRemovalGraph(model, true, include_nested);

  auto& graph = model.MainGraph();
  auto& first_node = *graph.GetNode(0);
  auto& node_to_remove = *graph.GetNode(1);
  const auto& if_node = *graph.GetNode(2);

  bool removed = graph_utils::RemoveNode(graph, node_to_remove);
  ASSERT_TRUE(removed);

  // check subgraph implicit input was updated
  const auto& implicit_inputs = if_node.ImplicitInputDefs();
  const auto& implicit_input_name = implicit_inputs[0]->Name();
  const auto& first_node_output_name = first_node.OutputDefs()[0]->Name();
  EXPECT_TRUE(implicit_input_name == first_node_output_name);

  // check subgraphs were updated
  CheckNodeRemovalSubgraphUpdate(first_node_output_name, *if_node.GetGraphAttribute("then_branch"));
  CheckNodeRemovalSubgraphUpdate(first_node_output_name, *if_node.GetGraphAttribute("else_branch"));
}

TEST(GraphUtils, UpdateSubgraphWhenRemovingNode) {
  UpdateSubgraphWhenRemovingNode(false);
}

TEST(GraphUtils, UpdateNestedSubgraphWhenRemovingNode) {
  UpdateSubgraphWhenRemovingNode(true);
}

// we can't remove a node if it is used as an implicit input in a subgraph, and changing the implicit input name
// will result with in a clash with an existing node in the subgraph
static void DontRemoveNodeIfItWillBreakSubgraph(bool test_nested = false) {
  Model model(std::string("DontRemoveNodeIfItWillBreakSubgraph") + (test_nested ? ":Nested" : ":SingleLevel"),
              false, DefaultLoggingManager().DefaultLogger());
  CreateNodeRemovalGraph(model, false, test_nested);

  auto& graph = model.MainGraph();
  auto& node_to_remove = *graph.GetNode(1);

  ASSERT_FALSE(graph_utils::CanRemoveNode(graph, node_to_remove,
                                          DefaultLoggingManager().DefaultLogger()));
}

TEST(GraphUtils, DontRemoveNodeIfItWillBreakSubgraph) {
  DontRemoveNodeIfItWillBreakSubgraph(false);
}

TEST(GraphUtils, DontRemoveNodeIfItWillBreakNestedSubgraph) {
  DontRemoveNodeIfItWillBreakSubgraph(true);
}

TEST(GraphUtils, TestMultiEdgeRemovalNodes) {
  // Create a graph with 5 Id nodes. The graph structure is as follows: Id0 ( Id1 Id2 ( Id3 Id4 ) ).
  // First we remove Id2, which leads to: Id0 ( Id1 Id4 Id5 ).
  // Then we remove Id1, which leads to: Id2 Id4 Id5, being fed the initializer.
  Model model("MultiEdgeRemovalGraph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& id_0_in = graph.GetOrCreateNodeArg("id_0_in", &float_tensor);
  auto& id_0_out = graph.GetOrCreateNodeArg("id_0_out", &float_tensor);
  auto& id_1_out = graph.GetOrCreateNodeArg("id_1_out", &float_tensor);
  auto& id_2_out = graph.GetOrCreateNodeArg("id_2_out", &float_tensor);
  auto& id_3_out = graph.GetOrCreateNodeArg("id_3_out", &float_tensor);
  auto& id_4_out = graph.GetOrCreateNodeArg("id_4_out", &float_tensor);

  std::vector<Node*> nodes;
  nodes.push_back(&graph.AddNode("id_0", "Identity", "Identity node 0", {&id_0_in}, {&id_0_out}));
  nodes.push_back(&graph.AddNode("id_1", "Identity", "Identity node 1", {&id_0_out}, {&id_1_out}));
  nodes.push_back(&graph.AddNode("id_2", "Identity", "Identity node 2", {&id_0_out}, {&id_2_out}));
  nodes.push_back(&graph.AddNode("id_3", "Identity", "Identity node 3", {&id_2_out}, {&id_3_out}));
  nodes.push_back(&graph.AddNode("id_4", "Identity", "Identity node 4", {&id_2_out}, {&id_4_out}));

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  ASSERT_EQ(graph.NumberOfNodes(), 5);

  // Check inputs/outputs of id_0 and id_2
  ASSERT_EQ(nodes[0]->GetInputEdgesCount(), 0u);
  ASSERT_EQ(nodes[0]->GetOutputEdgesCount(), 2u);
  ASSERT_EQ(nodes[2]->GetInputEdgesCount(), 1u);
  ASSERT_EQ(nodes[2]->GetOutputEdgesCount(), 2u);

  // Remove id_2. This leaves id_0 with 3 output edges. id_0 is now incoming node to id_3 and id_4.
  ASSERT_TRUE(graph_utils::RemoveNode(graph, *nodes[2]));
  ASSERT_EQ(graph.NumberOfNodes(), 4);
  ASSERT_EQ(nodes[0]->GetOutputEdgesCount(), 3u);
  ASSERT_EQ(nodes[3]->InputDefs().size(), 1u);
  ASSERT_TRUE(nodes[3]->InputDefs()[0]->Name() == "id_0_out");
  ASSERT_EQ(nodes[4]->InputDefs().size(), 1u);
  ASSERT_TRUE(nodes[4]->InputDefs()[0]->Name() == "id_0_out");

  // Remove id_0
  ASSERT_TRUE(graph_utils::RemoveNode(graph, *nodes[0]));
  ASSERT_EQ(graph.NumberOfNodes(), 3);
  ASSERT_TRUE(nodes[1]->InputDefs()[0]->Name() == "id_0_in");
  ASSERT_TRUE(nodes[3]->InputDefs()[0]->Name() == "id_0_in");
  ASSERT_TRUE(nodes[4]->InputDefs()[0]->Name() == "id_0_in");
}

TEST(GraphUtils, TestMultiOutputRemoveNode) {
  Model model("MultiOutputRemovalGraph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& do_0_in = graph.GetOrCreateNodeArg("do_0_in", &float_tensor);
  auto& do_0_out = graph.GetOrCreateNodeArg("do_0_out", &float_tensor);
  auto& do_0_out1 = graph.GetOrCreateNodeArg("do_0_out1", &bool_tensor);
  auto& id_1_out = graph.GetOrCreateNodeArg("id_1_out", &float_tensor);
  auto& id_2_out = graph.GetOrCreateNodeArg("id_2_out", &bool_tensor);

  std::vector<Node*> nodes;
  nodes.push_back(&graph.AddNode("do_0", "Dropout", "Dropout node 0", {&do_0_in}, {&do_0_out, &do_0_out1}));
  nodes.push_back(&graph.AddNode("id_1", "Identity", "Identity node 1", {&do_0_out}, {&id_1_out}));
  nodes.push_back(&graph.AddNode("id_2", "Identity", "Identity node 2", {&do_0_out1}, {&id_2_out}));

  std::vector<const NodeArg*> graph_outputs;
  graph_outputs.push_back(&id_2_out);
  graph.SetOutputs(graph_outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  ASSERT_EQ(graph.NumberOfNodes(), 3);

  // Check inputs/outputs of do_0, id_1, id_2
  ASSERT_EQ(nodes[0]->GetOutputEdgesCount(), 2u);
  ASSERT_EQ(nodes[1]->GetInputEdgesCount(), 1u);
  ASSERT_EQ(nodes[2]->GetInputEdgesCount(), 1u);

  // Try to remove do_0, which should return false
  // because both outputs are consumed by downstream Operators.
  ASSERT_FALSE(graph_utils::CanRemoveNode(graph, *nodes[0],
                                          DefaultLoggingManager().DefaultLogger()));

  // Try removing do_0 after removing id_2, which should return true
  // because it now has exactly one output consumed by downstream Operators.
  ASSERT_TRUE(graph_utils::CanRemoveNode(graph, *nodes[1],
                                         DefaultLoggingManager().DefaultLogger()));
  ASSERT_TRUE(graph_utils::RemoveNode(graph, *nodes[1]));
  ASSERT_FALSE(graph_utils::IsOutputUsed(*nodes[0], 0));
  ASSERT_TRUE(graph_utils::CanRemoveNode(graph, *nodes[0],
                                         DefaultLoggingManager().DefaultLogger()));
  ASSERT_TRUE(graph_utils::RemoveNode(graph, *nodes[0]));
}

}  // namespace test
}  // namespace onnxruntime
