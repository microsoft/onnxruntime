// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "gtest/gtest.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"
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
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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

TEST(TransformerTest, CastRemovalDoesNotLowerPrecisionTest) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float_32;
  tensor_float_32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_float_64;
  tensor_float_64.mutable_tensor_type()->set_elem_type(TensorProto_DataType_DOUBLE);
  onnxruntime::NodeArg n1_def("N1", &tensor_float_64),
      n2_def("N2", &tensor_float_32),
      n3_def("N3", &tensor_float_64);

  NodeAttributes n1_attrs = {{"to", utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT))}};
  NodeAttributes n2_attrs = {{"to", utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE))}};

  graph.AddNode("node1", "Cast", "F64 to F32 cast", ArgMap{&n1_def}, ArgMap{&n2_def}, &n1_attrs);
  graph.AddNode("node2", "Cast", "F32 to F64 cast", ArgMap{&n2_def}, ArgMap{&n3_def}, &n2_attrs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer cast_inserter("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = true;
  status = cast_inserter.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // When casting f64 -> f32 -> f64 we should not be optimizing away the cast since there is a loss of precision.
  EXPECT_EQ(graph.NumberOfNodes(), 2);
}

TEST(TransformerTest, CastRemovalDoesNotRemoveSignednessTest) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_uint32;
  tensor_uint32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_UINT32);
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  onnxruntime::NodeArg n1_def("N1", &tensor_int32),
      n2_def("N2", &tensor_uint32),
      n3_def("N3", &tensor_int32);

  NodeAttributes n1_attrs = {{"to", utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_UINT32))}};
  NodeAttributes n2_attrs = {{"to", utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32))}};

  graph.AddNode("node1", "Cast", "I32 to UI32 cast", ArgMap{&n1_def}, ArgMap{&n2_def}, &n1_attrs);
  graph.AddNode("node2", "Cast", "UI32 to I32 cast", ArgMap{&n2_def}, ArgMap{&n3_def}, &n2_attrs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer cast_inserter("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = true;
  status = cast_inserter.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // When casting i32 -> ui32 -> i32 we should not be optimizing away the cast since applying the casts produces a very different result.
  EXPECT_EQ(graph.NumberOfNodes(), 2);
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

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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

  constexpr bool recurse_into_subgraphs = false;
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

  // for the sake of this example, pretend Clip has no fp16 kernel; Round genuinely has one on CPU
  // -> Clip -> Round -> Clip -> Round -> Clip ->
  //                              |         |
  //                              - O4       - O5
  auto& node1 = graph.AddNode("node1", "Clip", "no fp16", {&i1_def}, {&o1_def});
  auto& node2 = graph.AddNode("node2", "Round", "fp16", {&o1_def}, {&o2_def});
  auto& node3 = graph.AddNode("node3", "Clip", "no fp16", {&o2_def}, {&o3_def});
  auto& node4 = graph.AddNode("node4", "Round", "fp16 producing graph output", {&o3_def}, {&o4_def});
  auto& node5 = graph.AddNode("node5", "Clip", "no fp16", {&o4_def}, {&o5_def});

  // manually set outputs as we want O4 and well as O5 to be graph outputs.
  // AddNode creates a NodeArg instance in Graph so need to get address from the node
  graph.SetOutputs({node4.OutputDefs()[0], node5.OutputDefs()[0]});

  // node2 and node4 have a kernel
  node2.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  node4.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = true;
  EXPECT_TRUE(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()).IsOK());

  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  // we expect:
  //   node2 Round to get forced to fp32 as it's isolated between node1 and node3 which need Casts
  //   node4 Round should not get forced to fp32 as it produces a graph output
  //
  // -> CastFp32 -> Clip -> Round -> Clip -> CastFp16 -> Round -> CastFp32 -> Clip -> CastFp16
  //                                                       |                              |
  //                                                        - O4                           - O5
  EXPECT_TRUE(is_type(*node1.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node2.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node3.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*node4.InputDefs()[0], DataTypeImpl::GetTensorType<MLFloat16>()));
  EXPECT_TRUE(is_type(*node5.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));

  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops["Cast"], 4);
}

// Regression test for a Level2+ fusion transformer assigning a fused node to the CPU EP without
// checking that a kernel actually exists for it. com.microsoft.BiasGelu only has a CPU kernel for
// float, not float16, but fusing an fp16 Add and an fp16 Gelu (both of which do have CPU kernels)
// can still produce a CPU-assigned fp16 BiasGelu node. The node here has no input edges (it only
// consumes graph inputs) and produces a graph output, so the pre-fix "isolated fp16 node" check
// would have skipped it entirely, leaving it fp16 and causing kernel lookup to fail when a session
// is initialized with this graph. IsFp16NodeOnCpuWithoutKernel must force it to fp32 regardless.
TEST(TransformerTest, Fp16FusedNodeWithNoCpuKernelForcedToFp32) {
#if defined(DISABLE_CONTRIB_OPS)
  GTEST_SKIP() << "BiasGelu is unavailable when contrib ops are disabled.";
#endif
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg a_def("A", &tensor_float_16),
      b_def("B", &tensor_float_16),
      c_def("C", &tensor_float_16);

  auto& bias_gelu = graph.AddNode("bias_gelu", "BiasGelu", "fp16 fusion output with no CPU fp16 kernel",
                                  ArgMap{&a_def, &b_def}, ArgMap{&c_def}, nullptr, kMSDomain);
  // Simulate the fusion transformer assigning the fused node the CPU EP without checking for a kernel.
  bias_gelu.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.SetOutputs({bias_gelu.OutputDefs()[0]});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(modified) << "The kernel-less fp16 node should have been wrapped with fp32 casts";
  status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  EXPECT_TRUE(is_type(*bias_gelu.InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*bias_gelu.InputDefs()[1], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(is_type(*bias_gelu.OutputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_EQ(bias_gelu.GetExecutionProviderType(), onnxruntime::kCpuExecutionProvider);

  // Cast A and B to fp32 on the way in, and cast the output back to fp16 for the graph output.
  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops["Cast"], 3);
}

// Contrast case for Fp16FusedNodeWithNoCpuKernelForcedToFp32: Round has a genuine CPU fp16 kernel,
// so even with the same "no input edges, produces a graph output" shape it must be left running in
// fp16. Forcing fp32 is only correct when there is no fp16 kernel to begin with.
TEST(TransformerTest, Fp16NodeWithCpuKernelAtGraphBoundaryNotForced) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg i1_def("I1", &tensor_float_16),
      o1_def("O1", &tensor_float_16);

  auto& round = graph.AddNode("round", "Round", "fp16 with real CPU kernel", {&i1_def}, {&o1_def});
  round.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.SetOutputs({round.OutputDefs()[0]});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_FALSE(modified) << "A node with a real fp16 kernel should not be forced to fp32";
  status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  ASSERT_TRUE(round.InputDefs()[0]->Type() != nullptr);
  EXPECT_EQ(DataTypeImpl::TypeFromProto(*round.InputDefs()[0]->TypeAsProto()),
            DataTypeImpl::GetTensorType<MLFloat16>());
  EXPECT_EQ(round.GetExecutionProviderType(), onnxruntime::kCpuExecutionProvider);

  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops.find("Cast"), ops.end());
}

// Regression test for a CPU-assigned fp16 node whose only fp16 value is an output, with no fp16
// (or any) input at all. RandomNormal(dtype=float16) has no CPU kernel for float16 (only float and
// double), so IsFp16NodeOnCpuWithoutKernel forces it to fp32, but NeedInsertCast only ever fires on
// fp16 *inputs* and this node has none. ApplyImpl must still assign it the CPU EP, rewrite its
// `dtype` attribute to FLOAT, and cast its output back to float16 for the graph output.
TEST(TransformerTest, Fp16OutputOnlyNodeWithNoCpuKernelForcedToFp32) {
  // Pin the ai.onnx opset to one where RandomNormal's schema (and therefore Node::SinceVersion())
  // resolves to version 1, matching the only CPU kernel registered for it. RandomNormal's schema was
  // later revised for a newer opset, and the CPU kernel for that revision isn't the point of this test.
  auto model = std::make_shared<onnxruntime::Model>(
      "test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
      std::unordered_map<std::string, int>{{onnxruntime::kOnnxDomain, 12}},
      std::vector<ONNX_NAMESPACE::FunctionProto>(), DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg o1_def("O1", &tensor_float_16);

  NodeAttributes attrs = {
      {"dtype", utils::MakeAttribute("dtype", static_cast<int64_t>(TensorProto_DataType_FLOAT16))},
      {"shape", utils::MakeAttribute("shape", std::vector<int64_t>{1})}};

  auto& random_normal = graph.AddNode("random_normal", "RandomNormal", "fp16 output with no CPU fp16 kernel",
                                      ArgMap{}, ArgMap{&o1_def}, &attrs);
  // Simulate the node somehow ending up assigned to the CPU EP without a kernel check.
  random_normal.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.SetOutputs({random_normal.OutputDefs()[0]});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(modified) << "The kernel-less output-only fp16 node should have been forced to fp32";
  status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  EXPECT_TRUE(is_type(*random_normal.OutputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_EQ(random_normal.GetExecutionProviderType(), onnxruntime::kCpuExecutionProvider);

  const auto& post_attrs = random_normal.GetAttributes();
  auto dtype_attr = post_attrs.find("dtype");
  ASSERT_NE(dtype_attr, post_attrs.end());
  EXPECT_EQ(dtype_attr->second.i(), static_cast<int64_t>(TensorProto_DataType_FLOAT));

  // Cast the output back to fp16 for the graph output.
  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops["Cast"], 1);
}

// A stand-in for a kernel registry a user adds to a session (SessionOptions::AddCustomOpDomain and
// friends end up in KernelRegistryManager::custom_kernel_registries_). Only the kernel def matters
// here: InsertCastTransformer looks kernels up but never creates one, so the create function is
// never called.
static std::shared_ptr<KernelRegistry> MakeCustomCpuRegistry(const char* op_type, const char* domain,
                                                             MLDataType constrained_type) {
  auto kernel_def = KernelDefBuilder()
                        .SetName(op_type)
                        .SetDomain(domain)
                        .SinceVersion(1)
                        .Provider(onnxruntime::kCpuExecutionProvider)
                        .TypeConstraint("T", constrained_type)
                        .Build();

  auto registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(registry->Register(KernelCreateInfo(
      std::move(kernel_def),
      [](FuncManager&, const OpKernelInfo&, std::unique_ptr<OpKernel>&) -> Status {
        return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "kernel is only ever looked up, never created");
      })));
  return registry;
}

// Custom kernel registries must be searched alongside the CPU EP's own registry when deciding whether
// a node has a kernel. This is the graph from Fp16FusedNodeWithNoCpuKernelForcedToFp32 -- an fp16
// com.microsoft.BiasGelu, which the CPU EP implements for float only -- except that a custom registry
// supplies the missing fp16 kernel. The node is therefore not kernel-less: rewriting it to fp32 would
// wrap it in casts it does not need and, worse, mean the kernel the user registered never runs.
TEST(TransformerTest, Fp16NodeWithCustomRegistryFp16KernelNotForced) {
#if defined(DISABLE_CONTRIB_OPS)
  GTEST_SKIP() << "BiasGelu is unavailable when contrib ops are disabled.";
#endif
  // Pin both opsets the graph uses. com.microsoft must be listed explicitly: Model uses a non-empty
  // domain_to_version map verbatim, so omitting it would leave the contrib node without a schema and
  // fail Resolve. Its version is 1 and stays there, but ai.onnx is pinned for the same reason as in
  // Fp16OutputOnlyNodeWithNoCpuKernelForcedToFp32 -- so nothing here moves when main's default opset
  // does.
  auto model = std::make_shared<onnxruntime::Model>(
      "test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
      std::unordered_map<std::string, int>{{onnxruntime::kOnnxDomain, 12}, {onnxruntime::kMSDomain, 1}},
      std::vector<ONNX_NAMESPACE::FunctionProto>(), DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg a_def("A", &tensor_float_16),
      b_def("B", &tensor_float_16),
      c_def("C", &tensor_float_16);

  auto& bias_gelu = graph.AddNode("bias_gelu", "BiasGelu", "fp16, kernel comes from a custom registry",
                                  ArgMap{&a_def, &b_def}, ArgMap{&c_def}, nullptr, kMSDomain);
  bias_gelu.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.SetOutputs({bias_gelu.OutputDefs()[0]});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto custom_registry = MakeCustomCpuRegistry("BiasGelu", kMSDomain, DataTypeImpl::GetTensorType<MLFloat16>());
  auto cpu_registry = DefaultCpuExecutionProvider()->GetKernelRegistry();
  // Custom registries first, matching the order GetKernelRegistriesByProviderType returns and the
  // priority SearchKernelRegistry applies when the kernel is looked up for real.
  InsertCastTransformer transformer("Test",
                                    InsertCastTransformer::KernelRegistryList{custom_registry.get(),
                                                                              cpu_registry.get()});

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_FALSE(modified) << "A node whose fp16 kernel comes from a custom registry must be left in fp16";
  status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  EXPECT_TRUE(is_type(*bias_gelu.InputDefs()[0], DataTypeImpl::GetTensorType<MLFloat16>()));
  EXPECT_TRUE(is_type(*bias_gelu.InputDefs()[1], DataTypeImpl::GetTensorType<MLFloat16>()));
  EXPECT_TRUE(is_type(*bias_gelu.OutputDefs()[0], DataTypeImpl::GetTensorType<MLFloat16>()));
  EXPECT_EQ(bias_gelu.GetExecutionProviderType(), onnxruntime::kCpuExecutionProvider);

  auto ops = CountOpsInGraph(graph);
  EXPECT_EQ(ops.find("Cast"), ops.end());
}

// Builds a graph holding a single CPU-assigned fp16 com.microsoft.BiasAdd producing a graph output.
// BiasAdd has no CPU kernel in any precision, so the node is kernel-less and whether it can be
// rewritten to fp32 depends entirely on where an fp32 kernel is found.
static onnxruntime::Node& BuildFp16BiasAddGraph(onnxruntime::Graph& graph, TypeProto& tensor_float_16,
                                                std::vector<std::unique_ptr<onnxruntime::NodeArg>>& arg_storage) {
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  ArgMap inputs;
  for (const char* name : {"X", "bias", "skip"}) {
    arg_storage.push_back(std::make_unique<onnxruntime::NodeArg>(name, &tensor_float_16));
    inputs.push_back(arg_storage.back().get());
  }
  arg_storage.push_back(std::make_unique<onnxruntime::NodeArg>("Y", &tensor_float_16));

  auto& bias_add = graph.AddNode("bias_add", "BiasAdd", "fp16 with no CPU kernel at all",
                                 inputs, ArgMap{arg_storage.back().get()}, nullptr, kMSDomain);
  bias_add.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  graph.SetOutputs({bias_add.OutputDefs()[0]});
  return bias_add;
}

// The other half of the same requirement: the fp32 kernel the fallback relies on may also live in a
// custom registry. Without one the transformer cannot confirm an fp32 kernel exists and has to leave
// the node alone (it warns and session initialization fails later); with one it converts the node.
TEST(TransformerTest, Fp16NodeWithCustomRegistryFp32KernelForcedToFp32) {
#if defined(DISABLE_CONTRIB_OPS)
  GTEST_SKIP() << "BiasGelu is unavailable when contrib ops are disabled.";
#endif
  auto& logger = DefaultLoggingManager().DefaultLogger();
  auto cpu_registry = DefaultCpuExecutionProvider()->GetKernelRegistry();
  auto is_type = [](const NodeArg& node_arg, const MLDataType type) {
    return node_arg.Type() != nullptr &&
           DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
  };

  // Both blocks below pin the two opsets the graph uses. com.microsoft must be listed explicitly:
  // Model uses a non-empty domain_to_version map verbatim, so omitting it would leave the contrib
  // node without a schema and fail Resolve. Its version is 1 and stays there, but ai.onnx is pinned
  // for the same reason as in Fp16OutputOnlyNodeWithNoCpuKernelForcedToFp32 -- so nothing here moves
  // when main's default opset does.

  // Baseline: the CPU EP registry alone has neither an fp16 nor an fp32 BiasAdd kernel, so there is
  // nothing to fall back to and the node stays as it is.
  {
    auto model = std::make_shared<onnxruntime::Model>(
        "test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
        std::unordered_map<std::string, int>{{onnxruntime::kOnnxDomain, 12}, {onnxruntime::kMSDomain, 1}},
        std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
    TypeProto tensor_float_16;
    std::vector<std::unique_ptr<onnxruntime::NodeArg>> arg_storage;
    auto& bias_add = BuildFp16BiasAddGraph(model->MainGraph(), tensor_float_16, arg_storage);

    auto status = model->MainGraph().Resolve();
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    InsertCastTransformer transformer("Test", cpu_registry.get());
    bool modified = false;
    status = transformer.Apply(model->MainGraph(), modified, logger);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_FALSE(modified) << "With no fp32 kernel to fall back to the node must be left untouched";
    EXPECT_TRUE(is_type(*bias_add.OutputDefs()[0], DataTypeImpl::GetTensorType<MLFloat16>()));
  }

  // With the fp32 kernel supplied by a custom registry the fallback is available and is applied.
  {
    auto model = std::make_shared<onnxruntime::Model>(
        "test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
        std::unordered_map<std::string, int>{{onnxruntime::kOnnxDomain, 12}, {onnxruntime::kMSDomain, 1}},
        std::vector<ONNX_NAMESPACE::FunctionProto>(), logger);
    TypeProto tensor_float_16;
    std::vector<std::unique_ptr<onnxruntime::NodeArg>> arg_storage;
    auto& bias_add = BuildFp16BiasAddGraph(model->MainGraph(), tensor_float_16, arg_storage);

    auto status = model->MainGraph().Resolve();
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    auto custom_registry = MakeCustomCpuRegistry("BiasAdd", kMSDomain, DataTypeImpl::GetTensorType<float>());
    InsertCastTransformer transformer("Test",
                                      InsertCastTransformer::KernelRegistryList{custom_registry.get(),
                                                                                cpu_registry.get()});
    bool modified = false;
    status = transformer.Apply(model->MainGraph(), modified, logger);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_TRUE(modified) << "The fp32 kernel in the custom registry makes the fp32 fallback possible";
    status = model->MainGraph().Resolve();
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    for (const auto* input_def : bias_add.InputDefs()) {
      EXPECT_TRUE(is_type(*input_def, DataTypeImpl::GetTensorType<float>()));
    }
    EXPECT_TRUE(is_type(*bias_add.OutputDefs()[0], DataTypeImpl::GetTensorType<float>()));
    EXPECT_EQ(bias_add.GetExecutionProviderType(), onnxruntime::kCpuExecutionProvider);

    // Three inputs cast to fp32 on the way in, and the output cast back to fp16 for the graph output.
    auto ops = CountOpsInGraph(model->MainGraph());
    EXPECT_EQ(ops["Cast"], 4);
  }
}

// Verify that RemoveDuplicateCastTransformer does not fuse Cast(float->int32)->Cast(int32->bool)
// because the intermediate int32 truncation changes semantics (e.g. -0.1 -> 0 -> false vs -0.1 -> true).
// Regression test for https://github.com/microsoft/onnxruntime/issues/28089
TEST(TransformerTest, CastFloatToIntToBoolNotFused) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float32;
  tensor_float32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  TypeProto tensor_bool;
  tensor_bool.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);

  onnxruntime::NodeArg x_def("X", &tensor_float32);
  onnxruntime::NodeArg mid_def("mid", &tensor_int32);
  onnxruntime::NodeArg y_def("Y", &tensor_bool);

  NodeAttributes cast1_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_INT32))}};
  NodeAttributes cast2_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_BOOL))}};

  graph.AddNode("Cast_1", "Cast", "float to int32",
                ArgMap{&x_def}, ArgMap{&mid_def}, &cast1_attrs);
  graph.AddNode("Cast_2", "Cast", "int32 to bool",
                ArgMap{&mid_def}, ArgMap{&y_def}, &cast2_attrs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Both Cast nodes must survive — float->int32 truncation is semantically significant.
  std::map<std::string, int> op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts["Cast"], 2)
      << "Cast(float->int32)->Cast(int32->bool) must not be fused to Cast(float->bool)";
}

// Verify that Cast(float->float16)->Cast(float16->int32) can still be optimized to Cast(float->int32).
// The first cast is lossy (float->float16) but the destination is not bool, so removal is allowed.
TEST(TransformerTest, LossyCastChainWithNonBoolDestIsOptimized) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float32;
  tensor_float32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_float16;
  tensor_float16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);

  onnxruntime::NodeArg x_def("X", &tensor_float32);
  onnxruntime::NodeArg mid_def("mid", &tensor_float16);
  onnxruntime::NodeArg y_def("Y", &tensor_int32);

  NodeAttributes cast1_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_FLOAT16))}};
  NodeAttributes cast2_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_INT32))}};

  graph.AddNode("Cast_1", "Cast", "float to float16",
                ArgMap{&x_def}, ArgMap{&mid_def}, &cast1_attrs);
  graph.AddNode("Cast_2", "Cast", "float16 to int32",
                ArgMap{&mid_def}, ArgMap{&y_def}, &cast2_attrs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // The first Cast should be removed, leaving only Cast(float->int32).
  std::map<std::string, int> op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts["Cast"], 1)
      << "Cast(float->float16)->Cast(float16->int32) should be optimized to Cast(float->int32)";
}

// Verify that Cast(float->int64)->Cast(int64->int32) can still be optimized to Cast(float->int32).
// The first cast is lossy (float->int64) but the destination is not bool, so removal is allowed.
TEST(TransformerTest, LossyCastFloatToInt64ToInt32IsOptimized) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float32;
  tensor_float32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_int64;
  tensor_int64.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  TypeProto tensor_int32;
  tensor_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);

  onnxruntime::NodeArg x_def("X", &tensor_float32);
  onnxruntime::NodeArg mid_def("mid", &tensor_int64);
  onnxruntime::NodeArg y_def("Y", &tensor_int32);

  NodeAttributes cast1_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_INT64))}};
  NodeAttributes cast2_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_INT32))}};

  graph.AddNode("Cast_1", "Cast", "float to int64",
                ArgMap{&x_def}, ArgMap{&mid_def}, &cast1_attrs);
  graph.AddNode("Cast_2", "Cast", "int64 to int32",
                ArgMap{&mid_def}, ArgMap{&y_def}, &cast2_attrs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // The first Cast should be removed, leaving only Cast(float->int32).
  std::map<std::string, int> op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts["Cast"], 1)
      << "Cast(float->int64)->Cast(int64->int32) should be optimized to Cast(float->int32)";
}

// Verify that RemoveDuplicateCastTransformer does not fuse consecutive Cast nodes
// that are assigned to different execution providers.
// Regression test for https://github.com/microsoft/onnxruntime/issues/27291
TEST(TransformerTest, CrossEpCastNodesNotFused) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  // Build: X(int64) -> Cast(int64->float32) -> Cast(float32->float16) -> Y(float16)
  TypeProto tensor_int64;
  tensor_int64.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  TypeProto tensor_float32;
  tensor_float32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_float16;
  tensor_float16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg x_def("X", &tensor_int64);
  onnxruntime::NodeArg mid_def("mid", &tensor_float32);
  onnxruntime::NodeArg y_def("Y", &tensor_float16);

  NodeAttributes cast1_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_FLOAT))}};
  NodeAttributes cast2_attrs = {
      {"to", utils::MakeAttribute("to",
                                  static_cast<int64_t>(TensorProto_DataType_FLOAT16))}};

  // Cast_1 on CPU EP, Cast_2 on WebGPU EP.
  auto& cast1 = graph.AddNode("Cast_1", "Cast", "int64 to float32",
                              ArgMap{&x_def}, ArgMap{&mid_def}, &cast1_attrs);
  cast1.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);

  auto& cast2 = graph.AddNode("Cast_2", "Cast", "float32 to float16",
                              ArgMap{&mid_def}, ArgMap{&y_def}, &cast2_attrs);
  cast2.SetExecutionProviderType(onnxruntime::kWebGpuExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Run InsertCastTransformer (which internally runs RemoveDuplicateCastTransformer)
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Both Cast nodes must survive — they should NOT be fused across EP boundaries.
  std::map<std::string, int> op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts["Cast"], 2) << "Cast nodes on different EPs must not be fused";

  // Verify Cast_2's input is still float32 (not changed to int64)
  const auto* cast2_input_type = cast2.InputDefs()[0]->TypeAsProto();
  ASSERT_NE(cast2_input_type, nullptr);
  EXPECT_EQ(cast2_input_type->tensor_type().elem_type(), TensorProto_DataType_FLOAT)
      << "Cast_2 input should remain float32, not be changed to int64";
}

// Verify that on_partition_assignment_fn_ is NOT called for a node whose CPU EP assignment was
// already recorded by the partitioner, even when ForceSingleNodeCPUFloat16ToFloat32 later clears
// that assignment to force the node through the fp32 cast-wrapping path.
//
// Graph: I1(fp16) -> Relu(no fp16 kernel, EP empty) -> O1(fp16)
//                 -> Concat(fp16 kernel, EP=CPU)     -> O2(fp16)
//                 -> Relu(no fp16 kernel, EP empty)  -> O3(fp16)
//
// The two Relu nodes are unassigned.  InsertCastTransformer's fallback policy wraps them with
// fp32 casts and assigns them to CPU EP; the callback should fire once for each.
//
// Concat was already assigned to CPU EP by the partitioner (it has an fp16 CPU kernel).
// ForceSingleNodeCPUFloat16ToFloat32 clears its EP so it also gets cast-wrapped (avoiding an
// isolated fp16 island).  When the transformer then assigns Concat to CPU EP as a fallback, it
// must NOT fire the callback — that would duplicate the partitioner's already-recorded assignment.
TEST(TransformerTest, IsolatedFp16NodeDoesNotDuplicatePartitionCallback) {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);

  onnxruntime::NodeArg i1_def("I1", &tensor_float_16),
      o1_def("O1", &tensor_float_16),
      o2_def("O2", &tensor_float_16),
      o3_def("O3", &tensor_float_16);

  // Leave the Relu nodes' EP unset so InsertCastTransformer treats them as newly assigned and wraps them.
  auto& relu1 = graph.AddNode("relu1", "Relu", "EP unset", {&i1_def}, {&o1_def});
  // Simulate a node that was already assigned by the partitioner.
  NodeAttributes concat_attrs = {{"axis", utils::MakeAttribute("axis", static_cast<int64_t>(0))}};
  auto& concat = graph.AddNode("concat", "Concat", "pre-assigned", {&o1_def}, {&o2_def}, &concat_attrs);
  concat.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  auto& relu2 = graph.AddNode("relu2", "Relu", "EP unset", {&o2_def}, {&o3_def});

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // Collect the node indices reported to the partition callback.
  std::vector<NodeIndex> callback_indices;
  auto on_assignment = [&callback_indices](const Graph&, const ComputeCapability& capability,
                                           const std::string&) {
    if (capability.sub_graph) {
      for (NodeIndex idx : capability.sub_graph->nodes) {
        callback_indices.push_back(idx);
      }
    }
  };

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    on_assignment);

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(modified);

  // Only the two Relu nodes (new CPU assignments) should have fired the callback.
  // Concat was already assigned by the partitioner — re-assigning it must not produce a duplicate.
  ASSERT_EQ(callback_indices.size(), 2u)
      << "on_partition_assignment_fn_ must fire exactly once per newly-assigned node; "
         "Concat was already assigned and must not produce a duplicate record";

  const NodeIndex relu1_idx = relu1.Index();
  const NodeIndex relu2_idx = relu2.Index();
  EXPECT_NE(std::find(callback_indices.begin(), callback_indices.end(), relu1_idx), callback_indices.end())
      << "Relu1 should have been reported as a new CPU assignment";
  EXPECT_NE(std::find(callback_indices.begin(), callback_indices.end(), relu2_idx), callback_indices.end())
      << "Relu2 should have been reported as a new CPU assignment";
  EXPECT_EQ(std::find(callback_indices.begin(), callback_indices.end(), concat.Index()), callback_indices.end())
      << "Concat was already assigned to CPU EP — its re-assignment must not fire the callback again";
}

}  // namespace test
}  // namespace onnxruntime
