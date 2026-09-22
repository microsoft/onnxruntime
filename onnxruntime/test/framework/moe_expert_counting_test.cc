// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <fstream>

#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime::test {
#if !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_MINIMAL_BUILD)
namespace {
using namespace ONNX_NAMESPACE;
constexpr int64_t kWidth = 128;
constexpr int64_t kExperts = 4;

void SetValue(ValueInfoProto& value, const std::string& name, int type,
              std::initializer_list<int64_t> dimensions) {
  value.set_name(name);
  auto* tensor = value.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(type);
  auto* shape = tensor->mutable_shape();
  for (auto dim : dimensions) {
    shape->add_dim()->set_dim_value(dim);
  }
}

void AddZeroInitializer(GraphProto& graph, const char* name, int type,
                        std::initializer_list<int64_t> dimensions, size_t element_size) {
  auto* tensor = graph.add_initializer();
  tensor->set_name(name);
  tensor->set_data_type(type);
  size_t size = element_size;
  for (auto dim : dimensions) {
    tensor->add_dims(dim);
    size *= static_cast<size_t>(dim);
  }
  tensor->set_raw_data(std::string(size, '\0'));
}

void PopulateMoeGraph(GraphProto& graph, bool quantized, bool cuda, bool subgraph) {
  graph.set_name("expert_counting");
  SetValue(subgraph ? *graph.add_value_info() : *graph.add_input(), "input",
           TensorProto_DataType_FLOAT16, {3, kWidth});
  SetValue(subgraph ? *graph.add_value_info() : *graph.add_input(), "router",
           TensorProto_DataType_FLOAT16, {3, kExperts});
  SetValue(*graph.add_output(), "output", TensorProto_DataType_FLOAT16, {3, kWidth});
  const int weight_type = quantized ? TensorProto_DataType_UINT8 : TensorProto_DataType_FLOAT16;
  AddZeroInitializer(graph, "w1", weight_type, {kExperts, kWidth, quantized ? kWidth / 2 : kWidth},
                     quantized ? 1 : 2);
  AddZeroInitializer(graph, "w2", weight_type, {kExperts, kWidth, quantized ? kWidth / 2 : kWidth},
                     quantized ? 1 : 2);
  if (quantized) {
    const int scale_type = cuda ? TensorProto_DataType_FLOAT16 : TensorProto_DataType_FLOAT;
    AddZeroInitializer(graph, "s1", scale_type, {kExperts, kWidth}, cuda ? 2 : 4);
    AddZeroInitializer(graph, "s2", scale_type, {kExperts, kWidth}, cuda ? 2 : 4);
  }
  for (int i = 0; i < 2; ++i) {
    auto* node = graph.add_node();
    node->set_name(MakeString("moe", i));
    node->set_op_type(quantized ? "QMoE" : "MoE");
    node->set_domain(kMSDomain);
    node->add_input(i == 0 ? "input" : "intermediate");
    node->add_input("router");
    node->add_input("w1");
    node->add_input(quantized ? "s1" : "");
    if (quantized) node->add_input("");
    node->add_input("w2");
    if (quantized) node->add_input("s2");
    node->add_output(i == 0 ? "intermediate" : "output");
    auto* k = node->add_attribute();
    k->set_name("k");
    k->set_type(AttributeProto_AttributeType_INT);
    k->set_i(1);
    auto* activation = node->add_attribute();
    activation->set_name("activation_type");
    activation->set_type(AttributeProto_AttributeType_STRING);
    activation->set_s("relu");
  }
}

std::string MakeCountingModel(bool quantized = false, bool cuda = false, bool subgraphs = false) {
  ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);
  opset = model.add_opset_import();
  opset->set_domain(kMSDomain);
  opset->set_version(1);
  auto& graph = *model.mutable_graph();
  if (!subgraphs) {
    PopulateMoeGraph(graph, quantized, cuda, false);
  } else {
    graph.set_name("conditional_counting");
    SetValue(*graph.add_input(), "input", TensorProto_DataType_FLOAT16, {3, kWidth});
    SetValue(*graph.add_input(), "router", TensorProto_DataType_FLOAT16, {3, kExperts});
    SetValue(*graph.add_input(), "condition", TensorProto_DataType_BOOL, {});
    SetValue(*graph.add_output(), "output", TensorProto_DataType_FLOAT16, {3, kWidth});
    auto* node = graph.add_node();
    node->set_op_type("If");
    node->add_input("condition");
    node->add_output("output");
    for (const char* branch : {"then_branch", "else_branch"}) {
      auto* attr = node->add_attribute();
      attr->set_name(branch);
      attr->set_type(AttributeProto_AttributeType_GRAPH);
      PopulateMoeGraph(*attr->mutable_g(), quantized, cuda, true);
    }
  }
  return model.SerializeAsString();
}

void RunCountingModel(InferenceSession& session, bool subgraphs = false, bool condition = true) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue input, router, cond;
  const std::vector<MLFloat16> values(3 * kWidth, MLFloat16(1.0f));
  CreateMLValue<MLFloat16>(allocator, {3, kWidth}, values, &input);
  const std::vector<MLFloat16> routing{
      MLFloat16(9.f), MLFloat16(1.f), MLFloat16(0.f), MLFloat16(0.f),
      MLFloat16(8.f), MLFloat16(1.f), MLFloat16(0.f), MLFloat16(0.f),
      MLFloat16(0.f), MLFloat16(1.f), MLFloat16(9.f), MLFloat16(0.f)};
  CreateMLValue<MLFloat16>(allocator, {3, kExperts}, routing, &router);
  NameMLValMap feeds{{"input", input}, {"router", router}};
  if (subgraphs) {
    CreateMLValue<bool>(allocator, {}, {condition}, &cond);
    feeds.emplace("condition", cond);
  }
  std::vector<OrtValue> outputs;
  const std::array<std::string, 1> output_names{"output"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
  ASSERT_EQ(outputs.size(), 1U);
  for (auto value : outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>()) {
    EXPECT_EQ(value.ToFloat(), 0.f);
  }
}

SessionOptions CountingOptions() {
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertCounting, "1"));
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  return options;
}

void TestCounting(bool quantized, bool cuda) {
  auto options = CountingOptions();
  if (cuda) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  InferenceSessionWrapper session(options, GetEnvironment());
  if (cuda) {
    auto provider = DefaultCudaExecutionProvider();
    if (!provider) {
      GTEST_SKIP() << "CUDA execution provider is unavailable.";
    }
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  }
  const auto model = MakeCountingModel(quantized, cuda);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->GetSnapshot().size(), 2U);
  for (int run = 1; run <= 2; ++run) {
    RunCountingModel(session);
    for (const auto& [key, node] : state->GetSnapshot()) {
      EXPECT_EQ(node.counters, (InlinedVector<double>{double(run), 0, double(run), 0}));
    }
  }
}
}  // namespace

TEST(MoeExpertCountingTest, CpuMoE) { TestCounting(false, false); }
TEST(MoeExpertCountingTest, CpuQMoE) { TestCounting(true, false); }
#if defined(USE_CUDA)
TEST(MoeExpertCountingTest, CudaMoE) { TestCounting(false, true); }
TEST(MoeExpertCountingTest, CudaQMoE) { TestCounting(true, true); }
#endif

TEST(MoeExpertCountingTest, DisabledAndIndependentSessions) {
  const auto model = MakeCountingModel();
  InferenceSessionWrapper disabled(SessionOptions{}, GetEnvironment());
  ASSERT_STATUS_OK(disabled.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(disabled.Initialize());
  EXPECT_EQ(disabled.GetSessionState().GetMoeExpertState(), nullptr);
  RunCountingModel(disabled);
  InferenceSessionWrapper first(CountingOptions(), GetEnvironment()), second(CountingOptions(), GetEnvironment());
  for (auto* session : {&first, &second}) {
    ASSERT_STATUS_OK(session->Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session->Initialize());
  }
  RunCountingModel(first);
  for (const auto& [key, node] : second.GetSessionState().GetMoeExpertState()->GetSnapshot()) {
    EXPECT_EQ(node.counters, (InlinedVector<double>{0, 0, 0, 0}));
  }
}

TEST(MoeExpertCountingTest, SharesStateWithSubgraphs) {
  const auto model = MakeCountingModel(false, false, true);
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->GetSnapshot().size(), 4U);
  for (const auto& [node, subgraphs] : session.GetSessionState().GetSubgraphSessionStateMap()) {
    for (const auto& [attribute, subgraph] : subgraphs) {
      EXPECT_EQ(subgraph->GetMoeExpertState(), state);
    }
  }
  RunCountingModel(session, true, true);
  RunCountingModel(session, true, false);
  for (const auto& [key, node] : state->GetSnapshot()) {
    EXPECT_EQ(node.counters, (InlinedVector<double>{1, 0, 1, 0}));
  }
}

TEST(MoeExpertCountingTest, LoadsInitialStateFile) {
  const char* path = "moe_expert_counting_initial_state.txt";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << "moe_expert_state 1\n\"main\" 0 MoE 0 3.5\n";
    ASSERT_TRUE(file.good());
  }
  auto options = CountingOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
  InferenceSessionWrapper session(options, GetEnvironment());
  const auto model = MakeCountingModel();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  RunCountingModel(session);
  EXPECT_EQ(session.GetSessionState().GetMoeExpertState()->GetSnapshot().at({"main", 0}).counters,
            (InlinedVector<double>{4.5, 0, 1, 0}));
}

TEST(MoeExpertCountingTest, InvalidConfigurationFailsInitialization) {
  for (const auto& entry : {
           std::pair{kOrtSessionOptionsConfigEnableMoeExpertCounting, "true"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterStateFile, "missing.txt"}}) {
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first, entry.second));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }
}
#endif
}  // namespace onnxruntime::test
