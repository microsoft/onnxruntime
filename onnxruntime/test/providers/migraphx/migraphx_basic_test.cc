// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "core/providers/migraphx/migraphx_execution_provider_utils.h"
#include <string>
#include <thread>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {

namespace test {

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with two inputs and one initializer.
 * input: "X", "Y" and "Z"
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Ini"  Add
 *     | \  /
 *     |  Add
 *     |      \
 *      \     Shape
 *       \    /
 *       Reshape
 *          |
 *          M
 */
void CreateBaseModel(onnxruntime::Model& model, std::vector<int> dims) {
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }

  // INT tensor
  ONNX_NAMESPACE::TypeProto int64_tensor;
  int64_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  int64_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  // constant
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  value_tensor.set_name("Ini");
  graph.AddInitializedTensor(value_tensor);

  // Create node1 (Add)
  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  // Create node2 (Add)
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Ini", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  // Create node3 (Shape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&output_arg_2);
  auto& output_arg_3 = graph.GetOrCreateNodeArg("S", &int64_tensor);
  outputs.push_back(&output_arg_3);
  graph.AddNode("node_3", "Shape", "node 3.", inputs, outputs);

  // Create node4 (Reshape)
  inputs.clear();
  outputs.clear();
  inputs.push_back(&input_arg_3);
  inputs.push_back(&output_arg_3);
  auto& output_arg_4 = graph.GetOrCreateNodeArg("R", &float_tensor);
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_4", "Reshape", "node 4.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
}

TEST(MIGraphXExecutionProviderTest, GraphInputName) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInput(gv, "X"), true);
}

TEST(MIGraphXExecutionProviderTest, GraphInitializer) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  ASSERT_EQ(IsGraphInitializer(gv, "Ini"), true);
}

TEST(MIGraphXExecutionProviderTest, NodeInputNum) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node0 = gv.GetNode(0);
  const auto& node1 = gv.GetNode(1);

  ASSERT_EQ(getNodeInputNum(*node0), 0);
  ASSERT_EQ(getNodeInputNum(*node1), 1);
}

TEST(MIGraphXExecutionProviderTest, IsNodeInput) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(1);
  ASSERT_EQ(isInputNode(node2, "M"), true);
}

TEST(MIGraphXExecutionProviderTest, canEvalArgument) {
  std::string graph_name = "migraphx_util_test";
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model, dims);

  auto& graph = model.MainGraph();
  GraphViewer gv(graph);

  // get the first add node
  const auto& node2 = gv.GetNode(3);
  std::vector<NodeIndex> input_nodes;
  ASSERT_EQ(canEvalNodeArgument(gv, node2, {1}, input_nodes), true);
}

#if defined(WIN32)
static bool SessionHasEp(Ort::Session& session, const char* ep_name) {
  // Access the underlying InferenceSession.
  const OrtSession* ort_session = session;
  const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);
  bool has_ep = false;

  for (const auto& provider : s->GetRegisteredProviderTypes()) {
    if (provider == ep_name) {
      has_ep = true;
      break;
    }
  }
  return has_ep;
}

// Tests autoEP feature to automatically select an EP that supports the GPU.
// Currently only works on Windows.
TEST(MIGraphXExecutionProviderTest, AutoEp_PreferGpu) {
  PathString model_name = ORT_TSTR("migraphx_basic_test.onnx");

  onnxruntime::Model model("test", false, DefaultLoggingManager().DefaultLogger());
  std::vector<int> dims = {1, 3, 2};
  CreateBaseModel(model, dims);

  auto status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());

  auto env = Ort::Env();
  env.UpdateEnvWithCustomLogLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

  {
    env.RegisterExecutionProviderLibrary(kMIGraphXExecutionProvider, ORT_TSTR("onnxruntime_providers_migraphx.dll"));

    Ort::SessionOptions so;
    so.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
    Ort::Session session_object(env, model_name.c_str(), so);
    EXPECT_TRUE(SessionHasEp(session_object, kMIGraphXExecutionProvider));
  }

  env.UnregisterExecutionProviderLibrary(kMIGraphXExecutionProvider);
}
#endif

// Create a minimal Loop body subgraph (identity pass-through).
static GraphProto CreateSimpleLoopBody() {
  Model model("loop_body", true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Body inputs
  auto& iter_num = graph.GetOrCreateNodeArg("i", &int64_scalar);
  auto& cond_in = graph.GetOrCreateNodeArg("body_cond", &bool_scalar);
  auto& v_in = graph.GetOrCreateNodeArg("body_v", &float_tensor_1d);

  // Body outputs
  auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &float_tensor_1d);

  graph.AddNode("cond_id", "Identity", "", {&cond_in}, {&cond_out});
  graph.AddNode("v_id", "Identity", "", {&v_in}, {&v_out});

  graph.SetInputs({&iter_num, &cond_in, &v_in});
  graph.SetOutputs({&cond_out, &v_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

// Create a Loop model with a trip-count initializer.
static Model BuildLoopModel(const char* model_name, const char* m_name, int64_t m_value,
                            std::function<void(Graph&, const char*, int64_t)> add_initializer) {
  Model model(model_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Trip count
  auto& m_arg = graph.GetOrCreateNodeArg(m_name, &int64_scalar);
  add_initializer(graph, m_name, m_value);

  // cond input
  auto& cond = graph.GetOrCreateNodeArg("cond", &bool_scalar);

  // v_init (loop carried dependency)
  auto& v_init = graph.GetOrCreateNodeArg("v_init", &float_tensor_1d);

  // v_final (loop output)
  auto& v_final = graph.GetOrCreateNodeArg("v_final", &float_tensor_1d);

  auto& loop_node = graph.AddNode("loop", "Loop", "",
                                   {&m_arg, &cond, &v_init}, {&v_final});
  loop_node.AddAttribute("body", CreateSimpleLoopBody());

  graph.SetInputs({&cond, &v_init});
  graph.SetOutputs({&v_final});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return model;
}

// Run a model with MIGraphX EP and verify the graph via the callback.
static void RunLoopTest(Model&& model, std::function<void(const Graph&)> verify,
                        ExpectedEPNodeAssignment expected = ExpectedEPNodeAssignment::Some,
                        NameMLValMap extra_feeds = {},
                        std::vector<int64_t> cond_shape = {}) {
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  // Feeds: cond (bool) and v_init (float[1])
  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  OrtValue cond_value;
  std::vector<bool> cond_data{true};
  CreateMLValue<bool>(cpu_alloc, cond_shape, cond_data, &cond_value);

  OrtValue v_init_value;
  std::vector<float> v_init_data{0.f};
  CreateMLValue<float>(cpu_alloc, std::vector<int64_t>{1}, v_init_data, &v_init_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("cond", cond_value));
  feeds.insert(std::make_pair("v_init", v_init_value));
  feeds.insert(extra_feeds.begin(), extra_feeds.end());

  EPVerificationParams params;
  params.ep_node_assignment = expected;
  params.graph_verifier = &verify;

  RunAndVerifyOutputsWithEP(model_data_span, "MIGraphX.LoopTest",
                            DefaultMIGraphXExecutionProvider(), feeds, params);
}

TEST(MIGraphXExecutionProviderTest, LoopSmallRawData) {
  auto add_raw = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.set_raw_data(&value, sizeof(int64_t));
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_small_raw", "m", 100, add_raw);
  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Loop with m=100 (raw_data) should be assigned to MIGraphX";
      }
    }
  };
  RunLoopTest(std::move(model), verify);
}

TEST(MIGraphXExecutionProviderTest, LoopBoundaryRawData) {
  auto add_raw = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.set_raw_data(&value, sizeof(int64_t));
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_boundary_raw", "m", 65535, add_raw);
  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Loop with m=65535 (raw_data) should be assigned to MIGraphX";
      }
    }
  };
  RunLoopTest(std::move(model), verify);
}

TEST(MIGraphXExecutionProviderTest, LoopOverLimitRawData) {
  auto add_raw = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.set_raw_data(&value, sizeof(int64_t));
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_overlimit_raw", "m", 65536, add_raw);
  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider)
            << "Loop with m=65536 (raw_data) should fall back to CPU";
      }
    }
  };
  RunLoopTest(std::move(model), verify, ExpectedEPNodeAssignment::None);
}

TEST(MIGraphXExecutionProviderTest, LoopSmallInt64Data) {
  auto add_typed = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.add_int64_data(value);
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_small_typed", "m", 100, add_typed);
  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Loop with m=100 (int64_data) should be assigned to MIGraphX";
      }
    }
  };
  RunLoopTest(std::move(model), verify);
}

TEST(MIGraphXExecutionProviderTest, LoopOverLimitInt64Data) {
  auto add_typed = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.add_int64_data(value);
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_overlimit_typed", "m", 100000, add_typed);
  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider)
            << "Loop with m=100000 (int64_data) should fall back to CPU";
      }
    }
  };
  RunLoopTest(std::move(model), verify, ExpectedEPNodeAssignment::None);
}

TEST(MIGraphXExecutionProviderTest, LoopDynamicTripCount) {
  // Trip count is a graph input, not an initializer.
  Model model("loop_dynamic", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& m_arg = graph.GetOrCreateNodeArg("m", &int64_scalar);
  auto& cond = graph.GetOrCreateNodeArg("cond", &bool_scalar);
  auto& v_init = graph.GetOrCreateNodeArg("v_init", &float_tensor_1d);
  auto& v_final = graph.GetOrCreateNodeArg("v_final", &float_tensor_1d);

  auto& loop_node = graph.AddNode("loop", "Loop", "",
                                   {&m_arg, &cond, &v_init}, {&v_final});
  loop_node.AddAttribute("body", CreateSimpleLoopBody());

  graph.SetInputs({&m_arg, &cond, &v_init});
  graph.SetOutputs({&v_final});
  ASSERT_TRUE(graph.Resolve().IsOK());

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider)
            << "Loop with dynamic trip count should fall back to CPU";
      }
    }
  };
  NameMLValMap extra_feeds;
  {
    auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    OrtValue m_value;
    std::vector<int64_t> m_data{10};
    CreateMLValue<int64_t>(cpu_alloc, std::vector<int64_t>{}, m_data, &m_value);
    extra_feeds.insert(std::make_pair("m", m_value));
  }
  RunLoopTest(std::move(model), verify, ExpectedEPNodeAssignment::None, std::move(extra_feeds));
}

// Loop body that contains a nested Loop.
static GraphProto CreateLoopBodyWithNestedLoop() {
  Model model("loop_body_nested", true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Body inputs
  auto& cond_in = graph.GetOrCreateNodeArg("outer_cond", &bool_scalar);
  auto& v_in = graph.GetOrCreateNodeArg("outer_v", &float_tensor_1d);

  // Nested loop trip count (constant)
  TensorProto m_proto;
  m_proto.set_name("inner_m");
  m_proto.set_data_type(TensorProto_DataType_INT64);
  m_proto.add_int64_data(1);
  graph.AddInitializedTensor(m_proto);
  auto& inner_m = graph.GetOrCreateNodeArg("inner_m", &int64_scalar);

  // Nested loop inputs/outputs
  auto& inner_v_final = graph.GetOrCreateNodeArg("inner_v_final", &float_tensor_1d);

  // Wire cond_in and v_in directly to the inner loop
  auto& inner_loop = graph.AddNode("inner_loop", "Loop", "",
                                    {&inner_m, &cond_in, &v_in}, {&inner_v_final});
  inner_loop.AddAttribute("body", CreateSimpleLoopBody());

  // Body outputs
  auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &float_tensor_1d);

  graph.AddNode("cond_id", "Identity", "", {&cond_in}, {&cond_out});
  graph.AddNode("v_id", "Identity", "", {&inner_v_final}, {&v_out});

  graph.SetInputs({&cond_in, &v_in});
  graph.SetOutputs({&cond_out, &v_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

// Loop body that contains an If.
static GraphProto CreateLoopBodyWithNestedIf() {
  Model model("loop_body_nested_if", true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Body inputs
  auto& cond_in = graph.GetOrCreateNodeArg("loop_cond", &bool_scalar);

  // Minimal then/else branches (no inputs; produce output via initializer).
  auto make_branch = [&]() -> GraphProto {
    Model branch("if_branch", true, DefaultLoggingManager().DefaultLogger());
    auto& bgraph = branch.MainGraph();
    TensorProto const_proto;
    const_proto.set_name("branch_out");
    const_proto.set_data_type(TensorProto_DataType_FLOAT);
    const_proto.add_dims(1);
    const_proto.add_float_data(1.f);
    bgraph.AddInitializedTensor(const_proto);
    auto& b_out = *bgraph.GetNodeArg("branch_out");
    bgraph.SetOutputs({&b_out});
    ORT_ENFORCE(bgraph.Resolve().IsOK(), "Graph resolve failed");
    return bgraph.ToGraphProto();
  };

  // If node takes only the condition as input.
  auto& if_v_out = graph.GetOrCreateNodeArg("if_v_out", &float_tensor_1d);

  auto& if_node = graph.AddNode("inner_if", "If", "", {&cond_in}, {&if_v_out});
  if_node.AddAttribute("then_branch", make_branch());
  if_node.AddAttribute("else_branch", make_branch());

  // Body outputs
  auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &float_tensor_1d);

  graph.AddNode("cond_id", "Identity", "", {&cond_in}, {&cond_out});
  graph.AddNode("v_id", "Identity", "", {&if_v_out}, {&v_out});

  graph.SetInputs({&cond_in});
  graph.SetOutputs({&cond_out, &v_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

TEST(MIGraphXExecutionProviderTest, LoopWithNestedLoop) {
  auto add_typed = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.add_int64_data(value);
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_nested_loop", "m", 10, add_typed);
  // Replace the simple body with one that contains a nested Loop
  auto& graph = model.MainGraph();
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Loop") {
      node.AddAttribute("body", CreateLoopBodyWithNestedLoop());
    }
  }
  ASSERT_TRUE(graph.Resolve().IsOK());

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Loop with nested Loop body should be assigned to MIGraphX";
      }
    }
  };
  RunLoopTest(std::move(model), verify);
}

TEST(MIGraphXExecutionProviderTest, LoopWithNestedIf) {
  auto add_typed = [](Graph& g, const char* name, int64_t value) {
    TensorProto tp;
    tp.set_name(name);
    tp.set_data_type(TensorProto_DataType_INT64);
    tp.add_int64_data(value);
    g.AddInitializedTensor(tp);
  };
  auto model = BuildLoopModel("loop_nested_if", "m", 10, add_typed);
  // Replace the simple body with one that contains an If
  auto& graph = model.MainGraph();
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Loop") {
      node.AddAttribute("body", CreateLoopBodyWithNestedIf());
    }
  }
  ASSERT_TRUE(graph.Resolve().IsOK());

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Loop with nested If body should be assigned to MIGraphX";
      }
    }
  };
  RunLoopTest(std::move(model), verify);
}
// Loop body with an unsupported op.
static GraphProto CreateLoopBodyWithUnsupportedOp() {
  Model model("loop_body_unsupported", true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_tensor_1d;
  bool_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Body inputs (required by Loop body schema, even if not consumed)
  auto& iter_num = graph.GetOrCreateNodeArg("i", &int64_scalar);
  auto& cond_in = graph.GetOrCreateNodeArg("body_cond", &bool_tensor_1d);
  auto& v_in = graph.GetOrCreateNodeArg("body_v", &float_tensor_1d);

  // Body outputs
  auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_tensor_1d);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &float_tensor_1d);

  // IsInf is not in mgx_supported_ops.
  graph.AddNode("isinf", "IsInf", "", {&v_in}, {&cond_out});
  graph.AddNode("v_id", "Identity", "", {&v_in}, {&v_out});

  graph.SetInputs({&iter_num, &cond_in, &v_in});
  graph.SetOutputs({&cond_out, &v_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

// Build a Loop model for the unsupported body test.
static Model BuildLoopModelWithUnsupportedBody(int64_t m_value) {
  Model model("loop_unsupported_body", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto bool_tensor_1d;
  bool_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Trip count
  TensorProto m_proto;
  m_proto.set_name("m");
  m_proto.set_data_type(TensorProto_DataType_INT64);
  m_proto.add_int64_data(m_value);
  graph.AddInitializedTensor(m_proto);
  auto& m_arg = graph.GetOrCreateNodeArg("m", &int64_scalar);

  // Condition
  auto& cond = graph.GetOrCreateNodeArg("cond", &bool_tensor_1d);

  // Loop carried dependency
  auto& v_init = graph.GetOrCreateNodeArg("v_init", &float_tensor_1d);
  auto& v_final = graph.GetOrCreateNodeArg("v_final", &float_tensor_1d);

  auto& loop_node = graph.AddNode("loop", "Loop", "",
                                   {&m_arg, &cond, &v_init}, {&v_final});
  loop_node.AddAttribute("body", CreateLoopBodyWithUnsupportedOp());

  graph.SetInputs({&cond, &v_init});
  graph.SetOutputs({&v_final});
  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return model;
}

TEST(MIGraphXExecutionProviderTest, LoopWithUnsupportedBody) {
  auto model = BuildLoopModelWithUnsupportedBody(10);

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "Loop") {
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider)
            << "Loop with unsupported body op (IsInf) should fall back to CPU";
      }
    }
  };
  RunLoopTest(std::move(model), verify, ExpectedEPNodeAssignment::None, {}, {1});
}

// If operator tests.
static GraphProto CreateSimpleIfBranch(const char* branch_name) {
  Model model(branch_name, true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Outer scope value.
  auto& outer_in = graph.GetOrCreateNodeArg("if_v", &float_tensor_1d);
  graph.AddOuterScopeNodeArg("if_v");

  auto& b_out = graph.GetOrCreateNodeArg("branch_out", &float_tensor_1d);

  graph.AddNode("id", "Identity", "", {&outer_in}, {&b_out});
  graph.SetOutputs({&b_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

// If branch with an unsupported op.
static GraphProto CreateIfBranchWithUnsupportedOp(const char* branch_name) {
  Model model(branch_name, true, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_tensor_1d;
  bool_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Outer scope value.
  auto& outer_in = graph.GetOrCreateNodeArg("if_v", &float_tensor_1d);
  graph.AddOuterScopeNodeArg("if_v");

  auto& b_out = graph.GetOrCreateNodeArg("branch_out", &bool_tensor_1d);

  // IsInf is not in mgx_supported_ops.
  graph.AddNode("isinf", "IsInf", "", {&outer_in}, {&b_out});
  graph.SetOutputs({&b_out});

  ORT_ENFORCE(graph.Resolve().IsOK(), "Graph resolve failed");
  return graph.ToGraphProto();
}

// Run an If model with MIGraphX EP and verify the graph via the callback.
static void RunIfTest(Model&& model, std::function<void(const Graph&)> verify,
                      ExpectedEPNodeAssignment expected = ExpectedEPNodeAssignment::Some) {
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  OrtValue cond_value;
  std::vector<bool> cond_data{true};
  CreateMLValue<bool>(cpu_alloc, std::vector<int64_t>{}, cond_data, &cond_value);

  // Outer scope value.
  OrtValue v_value;
  std::vector<float> v_data{1.0f};
  CreateMLValue<float>(cpu_alloc, std::vector<int64_t>{1}, v_data, &v_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("if_cond", cond_value));
  feeds.insert(std::make_pair("v_in", v_value));

  EPVerificationParams params;
  params.ep_node_assignment = expected;
  params.graph_verifier = &verify;

  RunAndVerifyOutputsWithEP(model_data_span, "MIGraphX.IfTest",
                            DefaultMIGraphXExecutionProvider(), feeds, params);
}

TEST(MIGraphXExecutionProviderTest, IfSimple) {
  Model model("if_simple", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Create outer scope variable.
  auto& v_in = graph.GetOrCreateNodeArg("v_in", &float_tensor_1d);
  auto& if_v = graph.GetOrCreateNodeArg("if_v", &float_tensor_1d);
  graph.AddNode("v_id", "Identity", "", {&v_in}, {&if_v});

  // If node.
  auto& if_cond = graph.GetOrCreateNodeArg("if_cond", &bool_scalar);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &float_tensor_1d);

  auto& if_node = graph.AddNode("if_op", "If", "", {&if_cond}, {&v_out});
  if_node.AddAttribute("then_branch", CreateSimpleIfBranch("then"));
  if_node.AddAttribute("else_branch", CreateSimpleIfBranch("else"));

  graph.SetInputs({&if_cond, &v_in});
  graph.SetOutputs({&v_out});
  ASSERT_TRUE(graph.Resolve().IsOK());

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "If") {
        EXPECT_EQ(node.GetExecutionProviderType(), kMIGraphXExecutionProvider)
            << "Simple If should be assigned to MIGraphX";
      }
    }
  };
  RunIfTest(std::move(model), verify);
}

TEST(MIGraphXExecutionProviderTest, IfWithUnsupportedBody) {
  Model model("if_unsupported", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape();

  TypeProto float_tensor_1d;
  float_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // Create outer scope variable.
  auto& v_in = graph.GetOrCreateNodeArg("v_in", &float_tensor_1d);
  auto& if_v = graph.GetOrCreateNodeArg("if_v", &float_tensor_1d);
  graph.AddNode("v_id", "Identity", "", {&v_in}, {&if_v});

  // If node.
  TypeProto bool_tensor_1d;
  bool_tensor_1d.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_tensor_1d.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& if_cond = graph.GetOrCreateNodeArg("if_cond", &bool_scalar);
  auto& v_out = graph.GetOrCreateNodeArg("v_out", &bool_tensor_1d);

  auto& if_node = graph.AddNode("if_op", "If", "", {&if_cond}, {&v_out});
  if_node.AddAttribute("then_branch", CreateIfBranchWithUnsupportedOp("then"));
  if_node.AddAttribute("else_branch", CreateIfBranchWithUnsupportedOp("else"));

  graph.SetInputs({&if_cond, &v_in});
  graph.SetOutputs({&v_out});
  ASSERT_TRUE(graph.Resolve().IsOK());

  std::function<void(const Graph&)> verify = [](const Graph& g) {
    for (const auto& node : g.Nodes()) {
      if (node.OpType() == "If") {
        EXPECT_EQ(node.GetExecutionProviderType(), kCpuExecutionProvider)
            << "If with unsupported body op (IsInf) should fall back to CPU";
      }
    }
  };
  RunIfTest(std::move(model), verify, ExpectedEPNodeAssignment::None);
}

}  // namespace test
}  // namespace onnxruntime
