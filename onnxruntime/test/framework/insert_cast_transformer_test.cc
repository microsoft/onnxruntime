// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocatormgr.h"
#include "core/framework/allocator.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/asserts.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

typedef std::vector<onnxruntime::NodeArg*> ArgMap;
TEST(TransformerTest, InsertCastGPUTest) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  onnxruntime::NodeArg i1_def("I1", &tensor_float_16),
      i2_def("I2", &tensor_float_16),
      i3_def("I3", &tensor_float_16),
      o1_def("O1", &tensor_float_16),
      o2_def("O2", &tensor_float_16),
      o3_def("O3", &tensor_float_16);

  auto& node1 = graph.AddNode("node1", "MatMul", "cpu operator1", ArgMap{&i1_def, &i2_def}, ArgMap{&o1_def});
  auto& node2 = graph.AddNode("node2", "MatMul", "gpu operator1", ArgMap{&o1_def, &i3_def}, ArgMap{&o2_def});
  node2.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  auto& node3 = graph.AddNode("node3", "Clip", "cpu operator2", ArgMap{&o2_def}, ArgMap{&o3_def});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  InsertCastTransformer transformer("Test");

  bool modified = true;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK());
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(graph.NumberOfNodes(), 8);
  for (auto it = node1.InputNodesBegin(); it != node1.InputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node1.OutputNodesBegin(); it != node1.OutputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node2.InputNodesBegin(); it != node2.InputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node2.OutputNodesBegin(); it != node2.OutputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node3.InputNodesBegin(); it != node3.InputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node3.OutputNodesBegin(); it != node3.OutputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
}

TEST(TransformerTest, InsertCastAllCPUTest) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  onnxruntime::NodeArg i1_def("I1", &tensor_float_16),
      i2_def("I2", &tensor_float_16),
      i3_def("I3", &tensor_float_16),
      o1_def("O1", &tensor_float_16),
      o2_def("O2", &tensor_float_16),
      o3_def("O3", &tensor_float_16);

  auto& node1 = graph.AddNode("node1", "MatMul", "cpu operator1", ArgMap{&i1_def, &i2_def}, ArgMap{&o1_def});
  auto& node2 = graph.AddNode("node2", "MatMul", "gpu operator1", ArgMap{&o1_def, &i3_def}, ArgMap{&o2_def});
  auto& node3 = graph.AddNode("node3", "Clip", "cpu operator2", ArgMap{&o2_def}, ArgMap{&o3_def});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test");

  bool modified = true;
  EXPECT_TRUE(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()).IsOK());
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(graph.NumberOfNodes(), 7);
  for (auto it = node1.InputNodesBegin(); it != node1.InputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
  for (auto it = node1.OutputNodesBegin(); it != node1.OutputNodesEnd(); ++it) {
    EXPECT_NE((*it).OpType(), "Cast");
  }
  for (auto it = node2.OutputNodesBegin(); it != node2.OutputNodesEnd(); ++it) {
    EXPECT_NE((*it).OpType(), "Cast");
  }
  for (auto it = node3.InputNodesBegin(); it != node3.InputNodesEnd(); ++it) {
    EXPECT_NE((*it).OpType(), "Cast");
  }
  for (auto it = node3.OutputNodesBegin(); it != node3.OutputNodesEnd(); ++it) {
    EXPECT_EQ((*it).OpType(), "Cast");
  }
}

// test that when there are 3 Cast ops in a row we remove the correct ones
TEST(TransformerTest, ThreeInARowRemoval) {
  auto model_uri = MODEL_FOLDER ORT_TSTR("triple-cast.onnx");
  std::shared_ptr<Model> model;
  auto status = Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // there are 3 in a row prior to a Transpose, and one post-Transpose.
  // we want to remove 2 of the first 3
  ASSERT_TRUE(op_to_count["Cast"] == 4);

  InsertCastTransformer transformer("Test");

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status;
  EXPECT_TRUE(modified) << "Transformer should have removed some Cast nodes";
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status;

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 2);
}

// test a case where the ONNX inferred output type (float16) is different from the type bound
// to the output NodeArg of the "RandomNormalLike" node (input is float16) because of the InsertCaseTransformer
// Here the ONNX inferred output type (float16) must be made float because that is what the kernel produces
TEST(TransformerTest, RandomNormalLikeWithFloat16Inputs) {
  auto model_uri = MODEL_FOLDER ORT_TSTR("random_normal_like_float16.onnx");
  std::shared_ptr<Model> model;
  auto status = Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  Graph& graph = model->MainGraph();
  InsertCastTransformer transformer("Test");

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status;
  EXPECT_TRUE(modified) << "Transformer should have added some Cast nodes";
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status;
}

// A case where the ONNX inferred output type is int32 to a node that consumes float16 input
// Here the InsertCastTransformer must not change the ONNX inferred output type and keep it
// as is (int32)
TEST(TransformerTest, MultinomialWithFloat16Input) {
  auto model_uri = MODEL_FOLDER ORT_TSTR("multinomial_float16.onnx");
  std::shared_ptr<Model> model;
  auto status = Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  Graph& graph = model->MainGraph();
  InsertCastTransformer transformer("Test");

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status;
  EXPECT_TRUE(modified) << "Transformer should have added some Cast nodes";
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status;
}

// This test is to test insert_cast_transform the same graph twice
// insert_cast_transform needs to detect existing Cast Node
// Prevent inserting the same Cast node twice
TEST(TransformerTest, InsertCastNodeTwice) {
  auto model_uri = MODEL_FOLDER ORT_TSTR("insert_cast_twice.onnx");
  std::shared_ptr<Model> model;
  auto status = Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  Graph& graph = model->MainGraph();
  InsertCastTransformer transformer("Test");

  // First insert
  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_TRUE(modified) << "Transformer should have added some Cast nodes";
  EXPECT_TRUE(op_to_count["Cast"] == 4) << "Insert 7 and remove 5 Cast nodes.";

  // Second insert
  modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;
  op_to_count = CountOpsInGraph(graph);
  // Same graph without modification; The number of Cast node remains
  EXPECT_TRUE(!modified) << "Transformer should not modify the modified graph again";
  EXPECT_TRUE(op_to_count["Cast"] == 4) << "Remain the same number of Cast node";
}

// Test that a node processing fp16 input with a subgraph does not get forced to fp32,
// and that the subgraph is processed to insert casts
TEST(TransformerTest, Fp16NodeWithSubgraph) {
  auto model_uri = MODEL_FOLDER ORT_TSTR("fp16model_loop.onnx");

  SessionOptions so;
  so.session_logid = "Fp16NodeWithSubgraph";
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model_uri));

  const Graph& graph = session.GetGraph();
  const auto& nodes = graph.Nodes();

  auto node_with_subgraph_iter = std::find_if(nodes.cbegin(), nodes.cend(),
                                              [](const Node& node) {
                                                return node.ContainsSubgraph();
                                              });

  ASSERT_NE(node_with_subgraph_iter, nodes.cend());

  const Graph& subgraph = *node_with_subgraph_iter->GetSubgraphs().front();

  const bool recurse_into_subgraphs = false;
  std::map<std::string, int> orig_graph_ops = CountOpsInGraph(graph, recurse_into_subgraphs);
  std::map<std::string, int> orig_subgraph_ops = CountOpsInGraph(subgraph, recurse_into_subgraphs);

  EXPECT_EQ(orig_graph_ops.find("Cast"), orig_graph_ops.cend());
  EXPECT_EQ(orig_subgraph_ops.find("Cast"), orig_subgraph_ops.cend());

  ASSERT_STATUS_OK(session.Initialize());

  std::map<std::string, int> new_graph_ops = CountOpsInGraph(graph, recurse_into_subgraphs);
  std::map<std::string, int> new_subgraph_ops = CountOpsInGraph(subgraph, recurse_into_subgraphs);

  EXPECT_EQ(new_graph_ops.find("Cast"), new_graph_ops.cend()) << "Main graph should not have been altered.";
  EXPECT_EQ(new_subgraph_ops.find("Cast")->second, 3) << "'Add' node in subgraph should have had Casts added";
}

TEST(TransformerTest, IsIsolatedFp16NodeOnCpuTest) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg i1_def("I1", &tensor_float_16),
      o1_def("O1", &tensor_float_16),
      o2_def("O2", &tensor_float_16),
      o3_def("O3", &tensor_float_16),
      o4_def("O4", &tensor_float_16),
      o5_def("O5", &tensor_float_16);

  // for the sake of this example, pretend Clip has no fp16 kernel but Abs does
  // -> Clip -> Abs -> Clip -> Abs -> Clip ->
  //                            |       |
  //                            - O4     - O5
  auto& node1 = graph.AddNode("node1", "Clip", "no fp16", {&i1_def}, {&o1_def});
  auto& node2 = graph.AddNode("node2", "Abs", "fp16", {&o1_def}, {&o2_def});
  auto& node3 = graph.AddNode("node3", "Clip", "no fp16", {&o2_def}, {&o3_def});
  auto& node4 = graph.AddNode("node4", "Abs", "fp16 producing graph output", {&o3_def}, {&o4_def});
  auto& node5 = graph.AddNode("node5", "Clip", "no fp16", {&o4_def}, {&o5_def});

  // manually set outputs as we want O4 and well as O5 to be graph outputs.
  // AddNode creates a NodeArg instance in Graph so need to get address from the node
  graph.SetOutputs({node4.OutputDefs()[0], node5.OutputDefs()[0]});

  // node2 and node4 have a kernel
  node2.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  node4.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test");

  bool modified = true;
  EXPECT_TRUE(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()).IsOK());

  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  // we expect:
  //   node2 Abs to get forced to fp32 as it's isolated between node1 and node3 which need Casts
  //   node4 Abs should not get forced to fp32 as it produces a graph output
  //
  // -> CastFp32 -> Clip -> Abs -> Clip -> CastFp16 -> Abs -> CastFp32 -> Clip -> CastFp16
  //                                                    |                            |
  //                                                     - O4                         - O5
  EXPECT_TRUE(is_type(*node1.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node2.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node3.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node4.InputDefs()[0], DataTypeImpl::GetTensorType<MLFloat16>()));
  EXPECT_TRUE(is_type(*node5.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));

  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops["Cast"], 4);
}

}  // namespace test
}  // namespace onnxruntime
