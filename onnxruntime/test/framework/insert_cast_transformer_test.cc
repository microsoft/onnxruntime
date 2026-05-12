// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include <filesystem>
#include "gtest/gtest.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/asserts.h"
#include <initializer_list>

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

static TypeProto MakeFp16TensorType(std::initializer_list<int64_t> shape = {}) {
  TypeProto tensor_float_16;
  tensor_float_16.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  if (shape.size() > 0) {
    auto* tensor_shape = tensor_float_16.mutable_tensor_type()->mutable_shape();
    for (const auto dim : shape) {
      tensor_shape->add_dim()->set_dim_value(dim);
    }
  }

  return tensor_float_16;
}

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

static std::shared_ptr<Model> MakeCpuFp16Model(const std::string& model_name, const std::string& op_type,
                                               bool assign_cpu_ep) {
  auto model = std::make_shared<onnxruntime::Model>(model_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();

  TypeProto tensor_float_16 = MakeFp16TensorType();

  auto& a = graph.GetOrCreateNodeArg("A", &tensor_float_16);
  auto& b = graph.GetOrCreateNodeArg("B", &tensor_float_16);
  auto& y = graph.GetOrCreateNodeArg("Y", &tensor_float_16);

  Node& node = [&]() -> Node& {
    if (op_type == "MatMul" || op_type == "Add") {
      return graph.AddNode(op_type, op_type, "fp16 test op", ArgMap{&a, &b}, ArgMap{&y});
    }

    if (op_type == "Abs") {
      return graph.AddNode(op_type, op_type, "fp16 test op", ArgMap{&a}, ArgMap{&y});
    }

    if (op_type == "Gemm") {
      auto& c = graph.GetOrCreateNodeArg("C", &tensor_float_16);
      graph.SetInputs({&a, &b, &c});
      return graph.AddNode(op_type, op_type, "fp16 test op", ArgMap{&a, &b, &c}, ArgMap{&y});
    }

    ORT_THROW("Unsupported op type for test: ", op_type);
  }();

  if (op_type == "Gemm") {
    // Inputs already set when the node is created.
  } else if (op_type == "Abs") {
    graph.SetInputs({&a});
  } else {
    graph.SetInputs({&a, &b});
  }
  graph.SetOutputs({&y});

  if (assign_cpu_ep) {
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  ORT_THROW_IF_ERROR(graph.Resolve());
  return model;
}

static std::shared_ptr<Model> MakeCpuFp16MatMulModelWithShapes(const std::string& model_name,
                                                               std::initializer_list<int64_t> a_shape,
                                                               std::initializer_list<int64_t> b_shape,
                                                               std::initializer_list<int64_t> y_shape,
                                                               bool assign_cpu_ep,
                                                               bool constant_b = false) {
  auto model = std::make_shared<onnxruntime::Model>(model_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model->MainGraph();

  TypeProto a_type = MakeFp16TensorType(a_shape);
  TypeProto b_type = MakeFp16TensorType(b_shape);
  TypeProto y_type = MakeFp16TensorType(y_shape);

  auto& a = graph.GetOrCreateNodeArg("A", &a_type);
  auto& b = graph.GetOrCreateNodeArg("B", &b_type);
  auto& y = graph.GetOrCreateNodeArg("Y", &y_type);

  if (constant_b) {
    TensorProto b_initializer;
    b_initializer.set_name("B");
    b_initializer.set_data_type(TensorProto_DataType_FLOAT16);
    for (const auto dim : b_shape) {
      b_initializer.add_dims(dim);
    }

    size_t element_count = 1;
    for (const auto dim : b_shape) {
      element_count *= static_cast<size_t>(dim);
    }
    std::vector<MLFloat16> data(element_count, MLFloat16::Zero);
    utils::SetRawDataInTensorProto(b_initializer, data.data(), data.size() * sizeof(MLFloat16));
    graph.AddInitializedTensor(b_initializer);
  }

  auto& node = graph.AddNode("MatMul", "MatMul", "fp16 matmul shape heuristic test", ArgMap{&a, &b}, ArgMap{&y});
  if (assign_cpu_ep) {
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  if (constant_b) {
    graph.SetInputs({&a});
  } else {
    graph.SetInputs({&a, &b});
  }
  graph.SetOutputs({&y});
  ORT_THROW_IF_ERROR(graph.Resolve());

  return model;
}

static void ExpectMlFloat16Output(const std::vector<MLFloat16>& output,
                                  const std::vector<MLFloat16>& expected) {
  ASSERT_EQ(output.size(), expected.size());
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_EQ(output[i].val, expected[i].val);
  }
}

static bool IsNodeArgType(const NodeArg& node_arg, const MLDataType type) {
  return node_arg.Type() != nullptr &&
         DataTypeImpl::TypeFromProto(*node_arg.TypeAsProto()) == type;
}

static bool CanRunNativeCpuFp16GemmRuntime() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return MlasFp16AccelerationSupported();
#else
  return false;
#endif
}

static const Node* FindNodeByOpType(const Graph& graph, const std::string& op_type) {
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) {
      return &node;
    }
  }

  return nullptr;
}

static std::vector<MLFloat16> RunCpuFp16Model(const std::string& model_name,
                                              const std::string& op_type,
                                              bool enable_cpu_fp16) {
  const auto input_path = ToPathString(model_name + (enable_cpu_fp16 ? "_enabled_runtime.onnx" : "_disabled_runtime.onnx"));
  ORT_THROW_IF_ERROR(Model::Save(*MakeCpuFp16Model(model_name, op_type, false), input_path));

  SessionOptions so;
  so.session_logid = model_name;
  so.graph_optimization_level = TransformerLevel::Level4;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(
      kOrtSessionOptionsEnableCpuFp16, enable_cpu_fp16 ? "1" : "0"));
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(
      kOrtSessionOptionsCpuFp16UseFp32FallbackHeuristic, enable_cpu_fp16 ? "0" : "1"));

  InferenceSessionWrapper session{so, GetEnvironment()};
  ORT_THROW_IF_ERROR(session.Load(input_path));
  ORT_THROW_IF_ERROR(session.Initialize());

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  NameMLValMap feeds;
  OrtValue a;
  CreateMLValue<MLFloat16>(allocator, {2, 2},
                           std::vector<MLFloat16>{MLFloat16(1.0f), MLFloat16(2.0f),
                                                  MLFloat16(3.0f), MLFloat16(4.0f)},
                           &a);
  feeds.insert({"A", a});

  OrtValue b;
  CreateMLValue<MLFloat16>(allocator, {2, 2},
                           std::vector<MLFloat16>{MLFloat16(5.0f), MLFloat16(6.0f),
                                                  MLFloat16(7.0f), MLFloat16(8.0f)},
                           &b);
  feeds.insert({"B", b});

  if (op_type == "Gemm") {
    OrtValue c;
    CreateMLValue<MLFloat16>(allocator, {2, 2},
                             std::vector<MLFloat16>{MLFloat16(1.0f), MLFloat16(1.0f),
                                                    MLFloat16(1.0f), MLFloat16(1.0f)},
                             &c);
    feeds.insert({"C", c});
  }

  std::vector<OrtValue> fetches;
  RunOptions run_options;
  ORT_THROW_IF_ERROR(session.Run(run_options, feeds, AsSpan({std::string("Y")}), &fetches));
  ORT_ENFORCE(fetches.size() == 1u, "Expected exactly one output for ", model_name);

  const auto& output = fetches[0].Get<Tensor>();
  std::vector<MLFloat16> result(output.Data<MLFloat16>(),
                                output.Data<MLFloat16>() + output.Shape().Size());

  std::error_code ec;
  std::filesystem::remove(std::filesystem::path{input_path}, ec);

  return result;
}

TEST(TransformerTest, CpuFp16MatMulPreservesWhenEnabled) {
  {
    auto model = MakeCpuFp16Model("cpu_fp16_matmul_optin", "MatMul", true);
    auto& graph = model->MainGraph();

    InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                      true, false);
    bool modified = false;
    EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
    EXPECT_STATUS_OK(graph.Resolve());

    const auto op_counts = CountOpsInGraph(graph);
    EXPECT_TRUE(op_counts.find("Cast") == op_counts.cend());
  }
}

TEST(TransformerTest, CpuFp16CpuAssignedMatMulHasNoCastsWhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_matmul_unassigned", "MatMul", false);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);
  node->SetExecutionProviderType(kCpuExecutionProvider);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_TRUE(op_counts.find("Cast") == op_counts.cend());

  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16CpuAssignedGemmHasNoCastsWhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_gemm_unassigned", "Gemm", false);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);
  node->SetExecutionProviderType(kCpuExecutionProvider);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_TRUE(op_counts.find("Cast") == op_counts.cend());

  EXPECT_EQ(node->OpType(), "Gemm");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16NonCpuAssignedMatMulIsUntouchedWhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_matmul_other_ep", "MatMul", false);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);
  node->SetExecutionProviderType("SomeOtherEP");

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_TRUE(op_counts.find("Cast") == op_counts.cend());

  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), "SomeOtherEP");
}

TEST(TransformerTest, CpuFp16NonCpuAssignedGemmIsUntouchedWhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_gemm_other_ep", "Gemm", false);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);
  node->SetExecutionProviderType("SomeOtherEP");

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_TRUE(op_counts.find("Cast") == op_counts.cend());

  EXPECT_EQ(node->OpType(), "Gemm");
  EXPECT_EQ(node->GetExecutionProviderType(), "SomeOtherEP");
}

TEST(TransformerTest, CpuFp16UnassignedMatMulKeepsFp16WhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_matmul_unassigned", "MatMul", false);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);

  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicForcesBertLikeShapeToFp32) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_bert_like",
                                                {128, 512}, {512, 512}, {128, 512}, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.at("Cast"), 3);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicKeepsLargeGemvNativeFp16) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_large_gemv",
                                                {1, 4096}, {4096, 4096}, {1, 4096}, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicKeepsBatchedLargeGemvNativeFp16) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_batched_large_gemv",
                                                {8, 1, 4096}, {8, 4096, 4096}, {8, 1, 4096}, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicForcesSingletonBroadcastRhsToFp32) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_singleton_broadcast_rhs",
                                                {8, 1, 4096}, {1, 4096, 4096}, {8, 1, 4096}, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.at("Cast"), 3);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicForcesFlattenedLargeGemvToFp32) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_flattened_large_gemv",
                                                {8, 1, 4096}, {4096, 4096}, {8, 1, 4096}, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.at("Cast"), 3);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicKeepsConstantRhsBertLikeShapeNativeFp16) {
  if (MlasHalfGemmNativePackBSize(CblasNoTrans, CblasNoTrans, 1024, 512) == 0) {
    GTEST_SKIP() << "No native packed-B fp16 MatMul backend is available.";
  }

  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_constant_rhs_bert_like",
                                                {128, 512}, {512, 1024}, {128, 1024}, true, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicForcesBatchedConstantRhsToFp32) {
  if (MlasHalfGemmNativePackBSize(CblasNoTrans, CblasNoTrans, 1024, 512) == 0) {
    GTEST_SKIP() << "No native packed-B fp16 MatMul backend is available.";
  }

  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_batched_constant_rhs",
                                                {2, 1, 512}, {2, 512, 1024}, {2, 1, 1024}, true, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.at("Cast"), 3);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulHeuristicForcesConstantRhsWhenNativePackedBUnavailable) {
  auto model = MakeCpuFp16MatMulModelWithShapes("cpu_fp16_matmul_heuristic_constant_rhs_no_native_pack",
                                                {128, 512}, {512, 1024}, {128, 1024}, true, true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config;
  mlas_backend_kernel_selector_config.use_kleidiai = false;
  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, true, &mlas_backend_kernel_selector_config);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.at("Cast"), 3);
  EXPECT_EQ(node->OpType(), "MatMul");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulSessionConfigUsesFallbackHeuristicKey) {
  const auto input_path = ToPathString("cpu_fp16_matmul_heuristic_mlas_config_runtime.onnx");
  ORT_THROW_IF_ERROR(Model::Save(*MakeCpuFp16MatMulModelWithShapes(
                                     "cpu_fp16_matmul_heuristic_mlas_config_runtime",
                                     {128, 512}, {512, 1024}, {128, 1024}, false, true),
                                 input_path));

  SessionOptions so;
  so.session_logid = "cpu_fp16_matmul_heuristic_mlas_config_runtime";
  so.graph_optimization_level = TransformerLevel::Level4;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableCpuFp16, "1"));
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsCpuFp16UseFp32FallbackHeuristic, "1"));
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsMlasDisableKleidiAi, "1"));

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(input_path));
  ASSERT_STATUS_OK(session.Initialize());

  const auto op_counts = CountOpsInGraph(session.GetGraph());
  EXPECT_EQ(op_counts.at("Cast"), 2);

  const Node* matmul_node = FindNodeByOpType(session.GetGraph(), "MatMul");
  ASSERT_NE(matmul_node, nullptr);

  ASSERT_EQ(matmul_node->InputDefs().size(), 2U);
  ASSERT_EQ(matmul_node->OutputDefs().size(), 1U);
  EXPECT_TRUE(IsNodeArgType(*matmul_node->InputDefs()[0], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(IsNodeArgType(*matmul_node->InputDefs()[1], DataTypeImpl::GetTensorType<float>()));
  EXPECT_TRUE(IsNodeArgType(*matmul_node->OutputDefs()[0], DataTypeImpl::GetTensorType<float>()));

  std::error_code ec;
  std::filesystem::remove(std::filesystem::path{input_path}, ec);
}

TEST(TransformerTest, CpuFp16UnsupportedOpStillGetsCastsWhenEnabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_abs_unassigned", "Abs", false);
  auto& graph = model->MainGraph();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_TRUE(op_counts.find("Cast") != op_counts.cend());
  EXPECT_EQ(op_counts.at("Cast"), 2);

  const Node* abs_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Abs") {
      abs_node = &node;
      break;
    }
  }
  ASSERT_NE(abs_node, nullptr);
  EXPECT_EQ(abs_node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16SupportedCpuOpKeepsFp16WhenEnabled) {
  if (!MlasFp16AccelerationSupported()) {
    GTEST_SKIP() << "CPU fp16 kernels are not registered on this platform.";
  }

  auto model = MakeCpuFp16Model("cpu_fp16_add_unassigned", "Add", false);
  auto& graph = model->MainGraph();

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    true, false);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);

  const Node* add_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Add") {
      add_node = &node;
      break;
    }
  }
  ASSERT_NE(add_node, nullptr);
  EXPECT_EQ(add_node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16CpuAssignedNewOptInOpsUseFp32FallbackWhenDisabled) {
  const std::vector<std::pair<std::string, int>> test_cases{
      {"MatMul", 3},
      {"Gemm", 4},
  };

  for (const auto& [op_type, expected_cast_count] : test_cases) {
    auto model = MakeCpuFp16Model("cpu_fp16_" + op_type + "_disabled_cpu_assigned", op_type, true);
    auto& graph = model->MainGraph();
    auto* node = graph.Nodes().begin().operator->();
    ASSERT_NE(node, nullptr);

    InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                      false, true);
    bool modified = false;
    EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
    EXPECT_STATUS_OK(graph.Resolve());

    const auto op_counts = CountOpsInGraph(graph);
    EXPECT_EQ(op_counts.at("Cast"), expected_cast_count);
    EXPECT_EQ(node->OpType(), op_type);
    EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
  }
}

TEST(TransformerTest, CpuFp16CpuAssignedExistingFp16OpHasNoExtraCastsWhenDisabled) {
  auto model = MakeCpuFp16Model("cpu_fp16_add_disabled_cpu_assigned", "Add", true);
  auto& graph = model->MainGraph();
  auto* node = graph.Nodes().begin().operator->();
  ASSERT_NE(node, nullptr);

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get(),
                                    false, true);
  bool modified = false;
  EXPECT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_STATUS_OK(graph.Resolve());

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts.count("Cast"), 0U);
  EXPECT_EQ(node->OpType(), "Add");
  EXPECT_EQ(node->GetExecutionProviderType(), kCpuExecutionProvider);
}

TEST(TransformerTest, CpuFp16MatMulRuntimeRunsWhenEnabled) {
  if (!CanRunNativeCpuFp16GemmRuntime()) {
    GTEST_SKIP() << "Native CPU fp16 MatMul runtime is not available on this platform.";
  }

  const auto output = RunCpuFp16Model("cpu_fp16_matmul_runtime", "MatMul", true);
  const std::vector<MLFloat16> expected{
      MLFloat16(19.0f), MLFloat16(22.0f),
      MLFloat16(43.0f), MLFloat16(50.0f)};
  ExpectMlFloat16Output(output, expected);
}

TEST(TransformerTest, CpuFp16GemmRuntimeRunsWhenEnabled) {
  if (!CanRunNativeCpuFp16GemmRuntime()) {
    GTEST_SKIP() << "Native CPU fp16 Gemm runtime is not available on this platform.";
  }

  const auto output = RunCpuFp16Model("cpu_fp16_gemm_runtime", "Gemm", true);
  const std::vector<MLFloat16> expected{
      MLFloat16(20.0f), MLFloat16(23.0f),
      MLFloat16(44.0f), MLFloat16(51.0f)};
  ExpectMlFloat16Output(output, expected);
}

TEST(TransformerTest, CpuFp16MatMulRuntimeRunsWhenDisabled) {
  const auto output = RunCpuFp16Model("cpu_fp16_matmul_runtime", "MatMul", false);
  const std::vector<MLFloat16> expected{
      MLFloat16(19.0f), MLFloat16(22.0f),
      MLFloat16(43.0f), MLFloat16(50.0f)};
  ExpectMlFloat16Output(output, expected);
}

TEST(TransformerTest, CpuFp16GemmRuntimeRunsWhenDisabled) {
  const auto output = RunCpuFp16Model("cpu_fp16_gemm_runtime", "Gemm", false);
  const std::vector<MLFloat16> expected{
      MLFloat16(20.0f), MLFloat16(23.0f),
      MLFloat16(44.0f), MLFloat16(51.0f)};
  ExpectMlFloat16Output(output, expected);
}

TEST(TransformerTest, CpuFp16MixedEpMatMulStaysOnNonCpuEpWhenEnabled) {
  const auto input_path = ToPathString("cpu_fp16_matmul_mixed_ep_runtime.onnx");
  ORT_THROW_IF_ERROR(Model::Save(*MakeCpuFp16Model("cpu_fp16_matmul_mixed_ep_runtime", "MatMul", false), input_path));

  SessionOptions so;
  so.session_logid = "cpu_fp16_matmul_mixed_ep_runtime";
  so.graph_optimization_level = TransformerLevel::Level4;
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsEnableCpuFp16, "1"));
  ORT_THROW_IF_ERROR(so.config_options.AddConfigEntry(kOrtSessionOptionsCpuFp16UseFp32FallbackHeuristic, "0"));

  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(
      std::make_unique<internal_testing_ep::InternalTestingExecutionProvider>(std::unordered_set<std::string>{"MatMul"})));
  ASSERT_STATUS_OK(session.Load(input_path));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& graph = session.GetGraph();
  size_t internal_testing_nodes = 0;
  size_t cpu_nodes = 0;
  size_t cast_nodes = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() == internal_testing_ep::kInternalTestingExecutionProvider) {
      ++internal_testing_nodes;
    }
    if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
      ++cpu_nodes;
    }
    if (node.OpType() == "Cast") {
      ++cast_nodes;
    }
  }

  EXPECT_EQ(internal_testing_nodes, 1u);
  EXPECT_EQ(cpu_nodes, 0u);
  EXPECT_EQ(cast_nodes, 0u);

  std::error_code ec;
  std::filesystem::remove(std::filesystem::path{input_path}, ec);
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

  InsertCastTransformer transformer("Test", DefaultCpuExecutionProvider()->GetKernelRegistry().get());

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

}  // namespace test
}  // namespace onnxruntime
