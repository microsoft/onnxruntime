// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <fstream>
#include <tuple>
#include <utility>

#include "core/framework/customregistry.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

namespace onnxruntime::test {
#if !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_MINIMAL_BUILD)
namespace {
using namespace ONNX_NAMESPACE;
constexpr int64_t kWidth = 128;
constexpr int64_t kPackedInt2GemvWidth = 512;
constexpr int64_t kPackedInt2GemvBlockSize = 64;
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

void PopulateMoeGraph(GraphProto& graph, bool quantized, bool cuda, bool subgraph, int64_t rows,
                      bool use_packed_int2_gemv = false, bool add_initializers = true) {
  const int64_t width = use_packed_int2_gemv ? kPackedInt2GemvWidth : kWidth;
  const int64_t fc1_rows = use_packed_int2_gemv ? 2 * width : width;
  const int64_t packed_width = quantized ? width / (use_packed_int2_gemv ? 4 : 2) : width;
  graph.set_name("expert_counting");
  SetValue(subgraph ? *graph.add_value_info() : *graph.add_input(), "input",
           TensorProto_DataType_FLOAT16, {rows, width});
  SetValue(subgraph ? *graph.add_value_info() : *graph.add_input(), "router",
           TensorProto_DataType_FLOAT16, {rows, kExperts});
  SetValue(*graph.add_output(), "output", TensorProto_DataType_FLOAT16, {rows, width});
  const int weight_type = quantized ? TensorProto_DataType_UINT8 : TensorProto_DataType_FLOAT16;
  if (add_initializers) {
    AddZeroInitializer(graph, "w1", weight_type, {kExperts, fc1_rows, packed_width}, quantized ? 1 : 2);
    AddZeroInitializer(graph, "w2", weight_type, {kExperts, width, packed_width}, quantized ? 1 : 2);
  }
  if (quantized && add_initializers) {
    const int scale_type = cuda ? TensorProto_DataType_FLOAT16 : TensorProto_DataType_FLOAT;
    if (use_packed_int2_gemv) {
      AddZeroInitializer(graph, "s1", scale_type, {kExperts, fc1_rows, width / kPackedInt2GemvBlockSize},
                         cuda ? 2 : 4);
      AddZeroInitializer(graph, "s2", scale_type, {kExperts, width, width / kPackedInt2GemvBlockSize},
                         cuda ? 2 : 4);
    } else {
      AddZeroInitializer(graph, "s1", scale_type, {kExperts, width}, cuda ? 2 : 4);
      AddZeroInitializer(graph, "s2", scale_type, {kExperts, width}, cuda ? 2 : 4);
    }
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
    activation->set_s(use_packed_int2_gemv ? "swiglu" : "relu");
    if (use_packed_int2_gemv) {
      for (const auto& [name, value] : {
               std::pair{"swiglu_fusion", int64_t{1}},
               std::pair{"expert_weight_bits", int64_t{2}},
               std::pair{"weights_prepacked", int64_t{0}},
               std::pair{"block_size", kPackedInt2GemvBlockSize}}) {
        auto* attribute = node->add_attribute();
        attribute->set_name(name);
        attribute->set_type(AttributeProto_AttributeType_INT);
        attribute->set_i(value);
      }
    }
  }
}

std::string MakeCountingModel(bool quantized = false, bool cuda = false, bool subgraphs = false, int64_t rows = 3,
                              bool use_packed_int2_gemv = false, bool outer_scope_weights = false) {
  ORT_ENFORCE(!use_packed_int2_gemv || (quantized && cuda && !subgraphs));
  ORT_ENFORCE(!outer_scope_weights || (cuda && subgraphs && !quantized));
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
    PopulateMoeGraph(graph, quantized, cuda, false, rows, use_packed_int2_gemv);
  } else {
    graph.set_name("conditional_counting");
    SetValue(*graph.add_input(), "input", TensorProto_DataType_FLOAT16, {rows, kWidth});
    SetValue(*graph.add_input(), "router", TensorProto_DataType_FLOAT16, {rows, kExperts});
    SetValue(*graph.add_input(), "condition", TensorProto_DataType_BOOL, {});
    SetValue(*graph.add_output(), "output", TensorProto_DataType_FLOAT16, {rows, kWidth});
    if (outer_scope_weights) {
      AddZeroInitializer(graph, "w1", TensorProto_DataType_FLOAT16, {kExperts, kWidth, kWidth}, 2);
      AddZeroInitializer(graph, "w2", TensorProto_DataType_FLOAT16, {kExperts, kWidth, kWidth}, 2);
    }
    auto* node = graph.add_node();
    node->set_op_type("If");
    node->add_input("condition");
    node->add_output("output");
    for (const char* branch : {"then_branch", "else_branch"}) {
      auto* attr = node->add_attribute();
      attr->set_name(branch);
      attr->set_type(AttributeProto_AttributeType_GRAPH);
      PopulateMoeGraph(*attr->mutable_g(), quantized, cuda, true, rows,
                       false, !outer_scope_weights);
    }
  }
  return model.SerializeAsString();
}

template <typename T>
NameMLValMap CountingFeedsTyped(bool subgraphs = false, bool condition = true, int64_t rows = 3,
                                int64_t width = kWidth) {
  ORT_ENFORCE(rows == 1 || rows == 3);
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue input, router, cond;
  const std::vector<T> values(rows * width, T(1.0f));
  CreateMLValue<T>(allocator, {rows, width}, values, &input);
  const std::vector<T> routing{
      T(9.f), T(1.f), T(0.f), T(0.f),
      T(8.f), T(1.f), T(0.f), T(0.f),
      T(0.f), T(1.f), T(9.f), T(0.f)};
  CreateMLValue<T>(allocator, {rows, kExperts},
                   gsl::make_span(routing).first(static_cast<size_t>(rows * kExperts)), &router);
  NameMLValMap feeds{{"input", input}, {"router", router}};
  if (subgraphs) {
    CreateMLValue<bool>(allocator, {}, {condition}, &cond);
    feeds.emplace("condition", cond);
  }
  return feeds;
}

NameMLValMap CountingFeeds(bool subgraphs = false, bool condition = true, int64_t rows = 3,
                           int64_t width = kWidth) {
  return CountingFeedsTyped<MLFloat16>(subgraphs, condition, rows, width);
}

template <typename T>
Status ExecuteCountingModelTyped(InferenceSession& session, std::vector<OrtValue>& outputs,
                                 bool subgraphs = false, bool condition = true, int64_t rows = 3,
                                 int64_t width = kWidth) {
  const std::array<std::string, 1> output_names{"output"};
  return session.Run(RunOptions{}, CountingFeedsTyped<T>(subgraphs, condition, rows, width), output_names, &outputs);
}

Status ExecuteCountingModel(InferenceSession& session, std::vector<OrtValue>& outputs,
                            bool subgraphs = false, bool condition = true, int64_t rows = 3,
                            int64_t width = kWidth) {
  return ExecuteCountingModelTyped<MLFloat16>(session, outputs, subgraphs, condition, rows, width);
}

void RunCountingModel(InferenceSession& session, bool subgraphs = false, bool condition = true, int64_t rows = 3,
                      int64_t width = kWidth) {
  std::vector<OrtValue> outputs;
  ASSERT_STATUS_OK(ExecuteCountingModel(session, outputs, subgraphs, condition, rows, width));
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

#if defined(USE_CUDA)
InlinedVector<int> CudaExperts(KernelPilotMoeExpertState& state, const OpKernel* kernel) {
  gsl::span<const int> experts;
  ORT_THROW_IF_ERROR(state.GetKernelPilot(kernel)->GetMoeCudaExperts(experts));
  return InlinedVector<int>(experts.begin(), experts.end());
}
#endif

// Groups the flat ExpertStat list by kernel, ordered by expert_id, for tests that don't
// care about graph identity and just want each registered node's counters.
std::map<const OpKernel*, InlinedVector<double>> CountersByKernel(const KernelPilotMoeExpertState& state) {
  std::map<const OpKernel*, std::map<size_t, double>> by_kernel;
  for (const auto& stat : state.GetExpertStats()) {
    by_kernel[stat.kernel][stat.expert_id] = stat.popularity;
  }
  std::map<const OpKernel*, InlinedVector<double>> counters;
  for (const auto& [kernel, experts] : by_kernel) {
    InlinedVector<double> values;
    values.reserve(experts.size());
    for (const auto& [expert_id, popularity] : experts) {
      values.push_back(popularity);
    }
    counters.emplace(kernel, std::move(values));
  }
  return counters;
}

class NoKernelPilotContext final : public OpKernelContextInternal {
 public:
  using OpKernelContextInternal::OpKernelContextInternal;

  KernelPilot* GetKernelPilot() const override {
    ADD_FAILURE() << "Disabled expert counting must not access KernelPilot.";
    return nullptr;
  }
};

class CollectingTestKernel final : public OpKernel {
 public:
  CollectingTestKernel(const OpKernelInfo& info, const bool& fail) : OpKernel(info), fail_(fail) {}

  Status Compute(OpKernelContext* context) const override {
    auto* pilot = context->GetKernelPilot();
    ORT_RETURN_IF_NOT(pilot, "Missing test kernel collector.");
    auto& usage = pilot->Moe();
    ORT_RETURN_IF_ERROR(usage.BeginInvocation(kExperts));
    const int selected[] = {fail_ ? 1 : 2};
    ORT_RETURN_IF_ERROR(usage.Collect(selected));
    ORT_RETURN_IF(fail_, "Intentional failure after collecting usage.");
    auto* output = context->Output(0, context->Input<Tensor>(0)->Shape());
    for (auto& value : output->MutableDataAsSpan<MLFloat16>()) {
      value = MLFloat16(0.f);
    }
    return Status::OK();
  }

 private:
  const bool& fail_;
};

void TestDisabledRecording(bool quantized) {
  for (bool explicit_disable : {false, true}) {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    if (explicit_disable) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertCounting, "0"));
    }
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel(quantized);
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    const auto& state = session.GetSessionState();
    ASSERT_EQ(state.GetMoeExpertState(), nullptr);
    InlinedVector<int> feed_indices;
    InlinedVector<OrtValue> feeds;
    for (const auto& [name, value] : CountingFeeds()) {
      int index = 0;
      ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx(name, index));
      feed_indices.push_back(index);
      feeds.push_back(value);
    }
    int output_index = 0;
    ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("output", output_index));
    const std::array<int, 1> fetch_indices{output_index};
    ExecutionFrame frame(feed_indices, feeds, fetch_indices, {}, {},
#ifdef ORT_ENABLE_STREAM
                         nullptr,
#endif
                         state);
    const bool terminate = false;
    size_t moe_nodes = 0;
    for (NodeIndex index : state.GetGraphViewer().GetNodesInTopologicalOrder()) {
      const auto* kernel = state.GetKernel(index);
      ASSERT_NE(kernel, nullptr);
      NoKernelPilotContext context(state, frame, *kernel, state.Logger(), terminate, nullptr);
      EXPECT_EQ(context.OpKernelContextInternal::GetKernelPilot(), nullptr);
      EXPECT_EQ(context.OpKernelContext::GetKernelPilot(), nullptr);
      ASSERT_STATUS_OK(context.RecordKernelUsage());
      ASSERT_STATUS_OK(kernel->Compute(&context));
      if (kernel->Node().OpType() == (quantized ? "QMoE" : "MoE")) {
        ++moe_nodes;
      }
    }
    EXPECT_EQ(moe_nodes, 2U);
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(frame.GetOutputs(outputs));
    ASSERT_EQ(outputs.size(), 1U);
    for (auto value : outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>()) {
      EXPECT_EQ(value.ToFloat(), 0.f);
    }
  }
}

void TestCounting(bool quantized, bool cuda, bool tiled = false, int64_t rows = 3,
                  bool use_packed_int2_gemv = false) {
  auto options = CountingOptions();
  if (tiled) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.qmoe_row_tile_size", "1"));
  }
  if (cuda) {
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  }
  if (use_packed_int2_gemv) {
    // Make the dense INT2 fallback fail so this test cannot pass without selecting packed GEMV.
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1"));
  }
  InferenceSessionWrapper session(options, GetEnvironment());
  if (cuda) {
    auto provider = DefaultCudaExecutionProvider();
    if (!provider) {
      GTEST_SKIP() << "CUDA execution provider is unavailable.";
    }
    if (GetCudaArchitecture() < 800) {
      GTEST_SKIP() << "CUDA MoE grouped GEMM requires compute capability 8.0 or newer.";
    }
    if (provider->GetOrtEp() != nullptr) {
      GTEST_SKIP() << "MoE expert counting is not supported by the CUDA plugin execution provider.";
    }
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  }
  const auto model = MakeCountingModel(quantized, cuda, false, rows, use_packed_int2_gemv);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->TotalExpertCount(), 2U * kExperts);
  const auto& session_state = session.GetSessionState();
  const std::array<size_t, 2> node_indices{0, 1};
  InlinedHashSet<size_t> expert_ids;
  for (size_t node_index : node_indices) {
    for (int expert = 0; expert < kExperts; ++expert) {
      size_t global_id = 0;
      ASSERT_STATUS_OK(state->GetExpertId(session_state.GetKernel(node_index), expert, global_id));
      EXPECT_TRUE(expert_ids.insert(global_id).second);
    }
  }
  EXPECT_EQ(expert_ids.size(), 2U * kExperts);
  const int64_t width = use_packed_int2_gemv ? kPackedInt2GemvWidth : kWidth;
  for (int run = 1; run <= 2; ++run) {
    RunCountingModel(session, false, true, rows, width);
    for (size_t node_index : node_indices) {
      const double expected = run == 1 ? 0.1 : 0.19;
      InlinedVector<double> counters;
      ASSERT_STATUS_OK(state->GetCounters(session_state.GetKernel(node_index), counters));
      ASSERT_EQ(counters.size(), 4U);
      EXPECT_DOUBLE_EQ(counters[0], expected);
      EXPECT_DOUBLE_EQ(counters[1], 0);
      EXPECT_DOUBLE_EQ(counters[2], rows == 3 ? expected : 0);
      EXPECT_DOUBLE_EQ(counters[3], 0);
    }
  }
}
}  // namespace

TEST(MoeExpertCountingTest, CpuMoE) { TestCounting(false, false); }
TEST(MoeExpertCountingTest, CpuQMoE) { TestCounting(true, false); }
TEST(MoeExpertCountingTest, CpuMoEDisabledDoesNotRecordUsage) { TestDisabledRecording(false); }
TEST(MoeExpertCountingTest, CpuQMoEDisabledDoesNotRecordUsage) { TestDisabledRecording(true); }

TEST(MoeExpertCountingTest, ContextExposesSessionOwnedCollector) {
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  const auto model = MakeCountingModel();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto& session_state = session.GetSessionState();
  auto* state = session_state.GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ExecutionFrame frame({}, {}, {}, {}, {},
#ifdef ORT_ENABLE_STREAM
                       nullptr,
#endif
                       session_state);
  const bool terminate = false;
  const auto* kernel = session_state.GetKernel(0);
  ASSERT_NE(kernel, nullptr);
  OpKernelContextInternal context(session_state, frame, *kernel, session_state.Logger(), terminate, nullptr);
  auto* pilot = context.GetKernelPilot();
  ASSERT_NE(pilot, nullptr);
  EXPECT_EQ(pilot, state->GetKernelPilot(kernel));
  EXPECT_EQ(context.GetKernelPilot(), pilot);
  EXPECT_NE(pilot, state->GetKernelPilot(session_state.GetKernel(1)));
  auto& usage = pilot->Moe();
  ASSERT_STATUS_OK(usage.BeginInvocation(kExperts));
  const int selected[] = {0, 2, 0};
  ASSERT_STATUS_OK(usage.Collect(selected));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state->GetCounters(kernel, counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0, 0, 0, 0}));
  ASSERT_STATUS_OK(context.RecordKernelUsage());
  ASSERT_STATUS_OK(state->GetCounters(kernel, counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0.1, 0, 0.1, 0}));

  OpKernelContextInternal unused_context(session_state, frame, *kernel, session_state.Logger(), terminate, nullptr);
  ASSERT_NE(unused_context.GetKernelPilot(), nullptr);
  ASSERT_STATUS_OK(unused_context.RecordKernelUsage());
  ASSERT_STATUS_OK(state->GetCounters(kernel, counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0.1, 0, 0.1, 0}));
}

TEST(MoeExpertCountingTest, FailedKernelDoesNotCommitCollectedUsage) {
  bool fail = true;
  auto registry = std::make_shared<CustomRegistry>();
  KernelDefBuilder definition;
  definition.SetName("MoE")
      .SetDomain(kMSDomain)
      .SinceVersion(1)
      .Provider(kCpuExecutionProvider)
      .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>());
  ASSERT_STATUS_OK(registry->RegisterCustomKernel(
      definition, [&fail](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) {
        kernel = std::make_unique<CollectingTestKernel>(info, fail);
        return Status::OK();
      }));
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  const auto model = MakeCountingModel();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  std::vector<OrtValue> outputs;
  const auto status = ExecuteCountingModel(session, outputs);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("Intentional failure after collecting usage."), std::string::npos);
  for (const auto& [kernel, counters] : CountersByKernel(*session.GetSessionState().GetMoeExpertState())) {
    EXPECT_EQ(counters, (InlinedVector<double>{0, 0, 0, 0}));
  }
  fail = false;
  RunCountingModel(session);
  for (const auto& [kernel, counters] : CountersByKernel(*session.GetSessionState().GetMoeExpertState())) {
    EXPECT_EQ(counters, (InlinedVector<double>{0, 0, 0.1, 0}));
  }
}

#if defined(USE_CUDA)
TEST(MoeExpertCountingTest, CudaMoE) { TestCounting(false, true); }
TEST(MoeExpertCountingTest, CudaQMoE) { TestCounting(true, true); }
TEST(MoeExpertCountingTest, CudaQMoETiled) { TestCounting(true, true, true); }
TEST(MoeExpertCountingTest, CudaQMoESingleToken) { TestCounting(true, true, false, 1); }
TEST(MoeExpertCountingTest, CudaQMoEPackedIntGemv) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE expert counting is not supported by the CUDA plugin execution provider.";
  }
  ScopedEnvironmentVariables env_vars{{{"ORT_DISABLE_MOE_GEMV", "0"}}};
  auto options = CountingOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  // Dense dequantization cannot fit: successful inference proves packed-INT GEMV dispatch.
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.qmoe_int_dequant_max_scratch_bytes", "1"));

  constexpr int64_t rows = 3;
  constexpr int64_t width = 512;
  constexpr int64_t block_size = 64;
  constexpr int64_t pack_size = 4;
  ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain(kMSDomain);
  opset->set_version(1);
  auto& graph = *model.mutable_graph();
  graph.set_name("packed_int_expert_counting");
  SetValue(*graph.add_input(), "input", TensorProto_DataType_FLOAT16, {rows, width});
  SetValue(*graph.add_input(), "router", TensorProto_DataType_FLOAT16, {rows, kExperts});
  SetValue(*graph.add_output(), "output", TensorProto_DataType_FLOAT16, {rows, width});

  auto add_identity_weights = [&](const char* name, int64_t output_width) {
    AddZeroInitializer(graph, name, TensorProto_DataType_UINT8,
                       {kExperts, output_width, width / pack_size}, 1);
    auto& data = *graph.mutable_initializer(graph.initializer_size() - 1)->mutable_raw_data();
    // Symmetric INT2 zero is code 2; each identity entry uses code 3 (+1).
    data.assign(data.size(), '\xAA');
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      for (int64_t row = 0; row < output_width; ++row) {
        const int64_t column = row % width;
        data[(expert * output_width + row) * (width / pack_size) + column / pack_size] |=
            static_cast<char>(1u << (2 * (column % pack_size)));
      }
    }
  };
  add_identity_weights("w1", 2 * width);
  add_identity_weights("w2", width);
  auto add_scales = [&](const char* name, int64_t output_width, bool per_expert) {
    auto* tensor = graph.add_initializer();
    tensor->set_name(name);
    tensor->set_data_type(TensorProto_DataType_FLOAT16);
    tensor->add_dims(kExperts);
    tensor->add_dims(output_width);
    tensor->add_dims(width / block_size);
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      const MLFloat16 scale(per_expert ? 0.25f * static_cast<float>(expert + 1) : 1.f);
      for (int64_t i = 0; i < output_width * (width / block_size); ++i) {
        tensor->add_int32_data(scale.val);
      }
    }
  };
  add_scales("s1", 2 * width, false);
  add_scales("s2", width, true);

  auto* node = graph.add_node();
  node->set_name("packed_moe");
  node->set_op_type("QMoE");
  node->set_domain(kMSDomain);
  for (const char* input : {"input", "router", "w1", "s1", "", "w2", "s2"}) {
    node->add_input(input);
  }
  node->add_output("output");
  for (const auto& [name, value] :
       {std::pair{"k", int64_t{1}}, {"expert_weight_bits", int64_t{2}}, {"block_size", block_size}, {"swiglu_fusion", int64_t{1}}, {"normalize_routing_weights", int64_t{1}}, {"weights_prepacked", int64_t{0}}}) {
    auto* attr = node->add_attribute();
    attr->set_name(name);
    attr->set_type(AttributeProto_AttributeType_INT);
    attr->set_i(value);
  }
  auto* activation = node->add_attribute();
  activation->set_name("activation_type");
  activation->set_type(AttributeProto_AttributeType_STRING);
  activation->set_s("swiglu");

  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto serialized = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->TotalExpertCount(), static_cast<size_t>(kExperts));
  const auto* kernel = session.GetSessionState().GetKernel(0);
  ASSERT_NE(kernel, nullptr);

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  auto feeds = CountingFeeds();
  const InlinedVector<MLFloat16> input(rows * width, MLFloat16(1.f));
  CreateMLValue<MLFloat16>(allocator, {rows, width}, input, &feeds.at("input"));
  const std::array<std::string, 1> output_names{"output"};
  const std::array<std::array<double, kExperts>, 3> expected_counters{{{0.1, 0, 0.1, 0}, {0.19, 0, 0.19, 0}, {0.171, 0.1, 0.171, 0}}};
  for (size_t run = 0; run < expected_counters.size(); ++run) {
    if (run == 2) {
      const InlinedVector<MLFloat16> router{
          MLFloat16(0.f), MLFloat16(9.f), MLFloat16(0.f), MLFloat16(0.f),
          MLFloat16(0.f), MLFloat16(9.f), MLFloat16(0.f), MLFloat16(0.f),
          MLFloat16(0.f), MLFloat16(9.f), MLFloat16(0.f), MLFloat16(0.f)};
      CreateMLValue<MLFloat16>(allocator, {rows, kExperts}, router, &feeds.at("router"));
    }
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    ASSERT_EQ(outputs.size(), 1U);
    const auto& output = outputs[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), TensorShape({rows, width}));
    for (int64_t row = 0; row < rows; ++row) {
      const int expert = run == 2 ? 1 : (row == 2 ? 2 : 0);
      const float expected = 0.25f * static_cast<float>(expert + 1) / (1.f + std::exp(-1.f));
      for (int64_t col = 0; col < width; ++col) {
        EXPECT_NEAR(output.Data<MLFloat16>()[row * width + col].ToFloat(), expected, 0.001f);
      }
    }
    InlinedVector<double> counters;
    ASSERT_STATUS_OK(state->GetCounters(kernel, counters));
    ASSERT_EQ(counters.size(), static_cast<size_t>(kExperts));
    for (size_t expert = 0; expert < counters.size(); ++expert) {
      EXPECT_NEAR(counters[expert], expected_counters[run][expert], 1e-12)
          << "run=" << run << ", expert=" << expert;
    }
  }
}
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
  for (const auto& [kernel, counters] : CountersByKernel(*second.GetSessionState().GetMoeExpertState())) {
    EXPECT_EQ(counters, (InlinedVector<double>{0, 0, 0, 0}));
  }
}

TEST(MoeExpertCountingTest, SharesStateWithSubgraphs) {
  const auto model = MakeCountingModel(false, false, true);
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->TotalExpertCount(), 4U * kExperts);
  for (const auto& [node, subgraphs] : session.GetSessionState().GetSubgraphSessionStateMap()) {
    for (const auto& [attribute, subgraph] : subgraphs) {
      EXPECT_EQ(subgraph->GetMoeExpertState(), state);
    }
  }
  RunCountingModel(session, true, true);
  RunCountingModel(session, true, false);
  const auto counters_by_kernel = CountersByKernel(*state);
  ASSERT_EQ(counters_by_kernel.size(), 4U);
  for (const auto& [kernel, counters] : counters_by_kernel) {
    EXPECT_EQ(counters, (InlinedVector<double>{0.1, 0, 0.1, 0}));
  }
}

TEST(MoeExpertCountingTest, AppliesConfiguredExponentialCounters) {
  const auto model = MakeCountingModel();
  for (const auto& [alpha, beta, expected] : {
           std::tuple{"0.5", "0.25", 0.375}, std::tuple{"0.5", "0.5", 0.75},
           std::tuple{"0", "0", 0.0}, std::tuple{"1", "0", 0.0}, std::tuple{"0", "1", 1.0}}) {
    SCOPED_TRACE(MakeString("alpha=", alpha, ", beta=", beta));
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterAlpha, alpha));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterBeta, beta));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    RunCountingModel(session);
    RunCountingModel(session);
    for (const auto& [kernel, counters] : CountersByKernel(*session.GetSessionState().GetMoeExpertState())) {
      EXPECT_EQ(counters, (InlinedVector<double>{expected, 0, expected, 0}));
    }
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
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(session.GetSessionState().GetMoeExpertState()->GetCounters(
      session.GetSessionState().GetKernel(0), counters));
  EXPECT_EQ(counters, (InlinedVector<double>{3.25, 0, 0.1, 0}));
}

#if defined(USE_CUDA)
TEST(MoeExpertCountingTest, StaticCpuOffloadDistributesZeroCountersAcrossNodes) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = MakeCountingModel(false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(CudaExperts(*state, session.GetSessionState().GetKernel(0)),
            (InlinedVector<int>{0, 1, 2}));
  EXPECT_EQ(CudaExperts(*state, session.GetSessionState().GetKernel(1)),
            (InlinedVector<int>{0, 1}));

  const auto& session_state = session.GetSessionState();
  const auto* execution_plan = session_state.GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);
  for (const char* initializer_name : {"w1", "w2"}) {
    int initializer_index = -1;
    ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx(initializer_name, initializer_index));
    EXPECT_EQ(execution_plan->GetLocation(static_cast<size_t>(initializer_index)).Type(), OrtDevice::CPU);
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadPlacesOuterScopeWeightsOnCpu) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = MakeCountingModel(false, true, true, 3, false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& session_state = session.GetSessionState();
  const auto* execution_plan = session_state.GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr);
  for (const char* initializer_name : {"w1", "w2"}) {
    int initializer_index = -1;
    ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx(initializer_name, initializer_index));
    EXPECT_EQ(execution_plan->GetLocation(static_cast<size_t>(initializer_index)).Type(), OrtDevice::CPU);
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadKeepsBranchLocalWeightsSeparateFromAncestor) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  ModelProto model;
  ASSERT_TRUE(model.ParseFromString(MakeCountingModel(false, true, true)));
  auto& graph = *model.mutable_graph();
  AddZeroInitializer(graph, "w1", TensorProto_DataType_FLOAT16, {kExperts, kWidth, kWidth}, sizeof(MLFloat16));
  const std::vector<MLFloat16> root_weights(static_cast<size_t>(kExperts * kWidth * kWidth), MLFloat16(3.0f));
  graph.mutable_initializer(0)->set_raw_data(root_weights.data(), root_weights.size() * sizeof(MLFloat16));
  SetValue(*graph.add_output(), "root_weights", TensorProto_DataType_FLOAT16, {kExperts, kWidth, kWidth});
  auto* identity = graph.add_node();
  identity->set_op_type("Identity");
  identity->add_input("w1");
  identity->add_output("root_weights");
  for (auto& attribute : *graph.mutable_node(0)->mutable_attribute()) {
    auto& branch = *attribute.mutable_g();
    for (auto& tensor : *branch.mutable_initializer()) {
      std::vector<MLFloat16> weights(root_weights.size(), MLFloat16(0.0f));
      for (int64_t expert = 0; expert < kExperts; ++expert) {
        for (int64_t column = 0; column < kWidth; ++column) {
          weights[static_cast<size_t>(expert * kWidth * kWidth + column * kWidth + column)] =
              MLFloat16(tensor.name() == "w1" ? 2.0f : 1.0f);
        }
      }
      tensor.set_raw_data(weights.data(), weights.size() * sizeof(MLFloat16));
    }
    for (auto& node : *branch.mutable_node()) {
      auto* normalize = node.add_attribute();
      normalize->set_name("normalize_routing_weights");
      normalize->set_type(AttributeProto_AttributeType_INT);
      normalize->set_i(1);
    }
  }
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "8"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto serialized = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& state = session.GetSessionState();
  int weight_index = -1;
  ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("w1", weight_index));
  ASSERT_NE(state.GetExecutionPlan(), nullptr);
  EXPECT_EQ(state.GetExecutionPlan()->GetLocation(static_cast<size_t>(weight_index)).Type(), OrtDevice::GPU);
  const std::array<std::string, 2> output_names{"output", "root_weights"};
  for (bool condition : {true, false}) {
    SCOPED_TRACE(condition);
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, CountingFeeds(true, condition), output_names, &outputs));
    ASSERT_EQ(outputs.size(), 2U);
    for (auto value : outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>()) {
      EXPECT_NEAR(value.ToFloat(), 4.0f, 0.01f);
    }
    const auto result = outputs[1].Get<Tensor>().DataAsSpan<MLFloat16>();
    ASSERT_EQ(result.size(), root_weights.size());
    for (auto value : result) {
      ASSERT_EQ(value.ToFloat(), 3.0f);
    }
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadIgnoresShadowedLoopValues) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  for (const std::string shadow : {"input", "initializer"}) {
    SCOPED_TRACE(shadow);
    ModelProto model;
    ASSERT_TRUE(model.ParseFromString(MakeCountingModel(false, true)));
    auto& graph = *model.mutable_graph();
    auto* trips = graph.add_initializer();
    trips->set_name("trips");
    trips->set_data_type(TensorProto_DataType_INT64);
    trips->add_int64_data(1);
    auto* condition = graph.add_initializer();
    condition->set_name("loop_condition");
    condition->set_data_type(TensorProto_DataType_BOOL);
    condition->add_int32_data(1);
    SetValue(*graph.add_output(), "loop_output", TensorProto_DataType_FLOAT16, {3, kWidth});
    auto* loop = graph.add_node();
    loop->set_op_type("Loop");
    loop->add_input("trips");
    loop->add_input("loop_condition");
    loop->add_input("input");
    loop->add_output("loop_output");
    auto* attribute = loop->add_attribute();
    attribute->set_name("body");
    attribute->set_type(AttributeProto_AttributeType_GRAPH);
    auto& body = *attribute->mutable_g();
    body.set_name("shadowed_weight");
    SetValue(*body.add_input(), "iteration", TensorProto_DataType_INT64, {});
    SetValue(*body.add_input(), "condition", TensorProto_DataType_BOOL, {});
    SetValue(*body.add_input(), shadow == "input" ? "w1" : "loop_input",
             TensorProto_DataType_FLOAT16, {3, kWidth});
    SetValue(*body.add_output(), "condition_out", TensorProto_DataType_BOOL, {});
    SetValue(*body.add_output(), "body_output", TensorProto_DataType_FLOAT16, {3, kWidth});
    auto* condition_identity = body.add_node();
    condition_identity->set_op_type("Identity");
    condition_identity->add_input("condition");
    condition_identity->add_output("condition_out");
    if (shadow == "initializer") {
      AddZeroInitializer(body, "w1", TensorProto_DataType_FLOAT16, {3, kWidth}, sizeof(MLFloat16));
    }
    auto* identity = body.add_node();
    identity->set_op_type("Identity");
    identity->add_input("w1");
    identity->add_output("body_output");

    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "1"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    const auto serialized = model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());

    const auto& state = session.GetSessionState();
    int weight_index = -1;
    ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("w1", weight_index));
    ASSERT_NE(state.GetExecutionPlan(), nullptr);
    EXPECT_EQ(state.GetExecutionPlan()->GetLocation(static_cast<size_t>(weight_index)).Type(), OrtDevice::CPU);
    std::vector<OrtValue> outputs;
    const std::array<std::string, 2> output_names{"output", "loop_output"};
    ASSERT_STATUS_OK(session.Run(RunOptions{}, CountingFeeds(), output_names, &outputs));
    ASSERT_EQ(outputs.size(), 2U);
    for (auto value : outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>()) {
      EXPECT_EQ(value.ToFloat(), 0.0f);
    }
    for (auto value : outputs[1].Get<Tensor>().DataAsSpan<MLFloat16>()) {
      EXPECT_EQ(value.ToFloat(), shadow == "initializer" ? 0.0f : 1.0f);
    }
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadRunsNodeWithAllCudaExperts) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  const char* path = "moe_static_cpu_offload_all_cuda_node_state.txt";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << "moe_expert_state 1\n"
            "\"main\" 0 MoE 0 10\n"
            "\"main\" 0 MoE 1 9\n"
            "\"main\" 0 MoE 2 8\n"
            "\"main\" 0 MoE 3 7\n";
    ASSERT_TRUE(file.good());
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "4"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = MakeCountingModel(false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(CudaExperts(*state, session.GetSessionState().GetKernel(0)),
            (InlinedVector<int>{0, 1, 2, 3}));
  EXPECT_TRUE(CudaExperts(*state, session.GetSessionState().GetKernel(1)).empty());
  RunCountingModel(session);
}

TEST(MoeExpertCountingTest, StaticCpuOffloadRanksLoadedCountersAcrossNodes) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  const char* path = "moe_static_cpu_offload_initial_state.txt";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << "moe_expert_state 1\n"
            "\"main\" 0 MoE 3 10\n"
            "\"main\" 1 MoE 2 9\n"
            "\"main\" 1 MoE 0 8\n"
            "\"main\" 0 MoE 1 7\n"
            "\"main\" 1 MoE 3 6\n";
    ASSERT_TRUE(file.good());
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  ModelProto numerical_model;
  ASSERT_TRUE(numerical_model.ParseFromString(MakeCountingModel(false, true)));
  for (auto& tensor : *numerical_model.mutable_graph()->mutable_initializer()) {
    std::vector<MLFloat16> weights(static_cast<size_t>(kExperts * kWidth * kWidth), MLFloat16(0.0f));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      for (int64_t column = 0; column < kWidth; ++column) {
        weights[static_cast<size_t>(expert * kWidth * kWidth + column * kWidth + column)] =
            MLFloat16(tensor.name() == "w1" ? 1.0f : static_cast<float>(expert + 1));
      }
    }
    tensor.set_raw_data(weights.data(), weights.size() * sizeof(MLFloat16));
  }
  for (auto& node : *numerical_model.mutable_graph()->mutable_node()) {
    auto* attribute = node.add_attribute();
    attribute->set_name("normalize_routing_weights");
    attribute->set_type(AttributeProto_AttributeType_INT);
    attribute->set_i(1);
  }
  const auto model = numerical_model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(CudaExperts(*state, session.GetSessionState().GetKernel(0)),
            (InlinedVector<int>{1, 3}));
  EXPECT_EQ(CudaExperts(*state, session.GetSessionState().GetKernel(1)),
            (InlinedVector<int>{0, 2, 3}));

  auto feeds = CountingFeeds();
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue router;
  const std::vector<MLFloat16> routing{
      MLFloat16(0.f), MLFloat16(9.f), MLFloat16(0.f), MLFloat16(0.f),
      MLFloat16(0.f), MLFloat16(0.f), MLFloat16(0.f), MLFloat16(9.f),
      MLFloat16(9.f), MLFloat16(0.f), MLFloat16(0.f), MLFloat16(0.f)};
  CreateMLValue<MLFloat16>(allocator, {3, kExperts}, routing, &router);
  feeds["router"] = router;
  std::vector<OrtValue> outputs;
  const std::array<std::string, 1> output_names{"output"};
  ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
  ASSERT_EQ(outputs.size(), 1U);
  const auto values = outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>();
  ASSERT_EQ(values.size(), static_cast<size_t>(3 * kWidth));
  const std::array<float, 3> expected{4.0f, 16.0f, 1.0f};
  for (size_t row = 0; row < expected.size(); ++row) {
    for (size_t column = 0; column < static_cast<size_t>(kWidth); ++column) {
      EXPECT_NEAR(values[row * static_cast<size_t>(kWidth) + column].ToFloat(), expected[row], 0.01f);
    }
  }
}

template <typename T>
void RunAdaptiveCpuOffloadPublishesCompletedSwapsAtNextInvocation() {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "4"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertSwapEpsilon, "0"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  ModelProto numerical_model;
  ASSERT_TRUE(numerical_model.ParseFromString(MakeCountingModel(false, true)));
  constexpr int tensor_type = std::is_same_v<T, BFloat16>
                                  ? TensorProto_DataType_BFLOAT16
                                  : TensorProto_DataType_FLOAT16;
  auto& graph = *numerical_model.mutable_graph();
  for (auto* values : {graph.mutable_input(), graph.mutable_output(), graph.mutable_value_info()}) {
    for (auto& value : *values) {
      auto* type = value.mutable_type()->mutable_tensor_type();
      if (type->elem_type() == TensorProto_DataType_FLOAT16) {
        type->set_elem_type(tensor_type);
      }
    }
  }
  for (auto& tensor : *graph.mutable_initializer()) {
    tensor.set_data_type(tensor_type);
    std::vector<T> weights(static_cast<size_t>(kExperts * kWidth * kWidth), T(0.0f));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      for (int64_t column = 0; column < kWidth; ++column) {
        weights[static_cast<size_t>(expert * kWidth * kWidth + column * kWidth + column)] =
            T(tensor.name() == "w1" ? 1.0f : static_cast<float>(expert + 1));
      }
    }
    tensor.set_raw_data(weights.data(), weights.size() * sizeof(T));
  }
  for (auto& node : *numerical_model.mutable_graph()->mutable_node()) {
    auto* attribute = node.add_attribute();
    attribute->set_name("normalize_routing_weights");
    attribute->set_type(AttributeProto_AttributeType_INT);
    attribute->set_i(1);
  }
  const auto model = numerical_model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  const auto* first_kernel = session.GetSessionState().GetKernel(0);
  const auto* second_kernel = session.GetSessionState().GetKernel(1);
  ASSERT_EQ(CudaExperts(*state, first_kernel), (InlinedVector<int>{0, 1}));
  ASSERT_EQ(CudaExperts(*state, second_kernel), (InlinedVector<int>{0, 1}));

  const auto run_and_validate = [&]() {
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(ExecuteCountingModelTyped<T>(session, outputs));
    ASSERT_EQ(outputs.size(), 1U);
    const auto values = outputs[0].Get<Tensor>().DataAsSpan<T>();
    const std::array<float, 3> expected{1.0f, 1.0f, 9.0f};
    constexpr float tolerance = std::is_same_v<T, BFloat16> ? 0.02f : 0.01f;
    for (size_t row = 0; row < expected.size(); ++row) {
      for (size_t column = 0; column < static_cast<size_t>(kWidth); ++column) {
        EXPECT_NEAR(values[row * static_cast<size_t>(kWidth) + column].ToFloat(), expected[row], tolerance);
      }
    }
  };

  run_and_validate();
  EXPECT_EQ(CudaExperts(*state, first_kernel), (InlinedVector<int>{0, 1}));
  EXPECT_EQ(CudaExperts(*state, second_kernel), (InlinedVector<int>{0, 1}));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(CudaExperts(*state, first_kernel), (InlinedVector<int>{0, 1}));
  EXPECT_EQ(CudaExperts(*state, second_kernel), (InlinedVector<int>{0, 1}));

  run_and_validate();
  EXPECT_EQ(CudaExperts(*state, first_kernel), (InlinedVector<int>{0, 2}));
  EXPECT_EQ(CudaExperts(*state, second_kernel), (InlinedVector<int>{0, 2}));
}

TEST(MoeExpertCountingTest, AdaptiveCpuOffloadPublishesCompletedFp16SwapsAtNextInvocation) {
  RunAdaptiveCpuOffloadPublishesCompletedSwapsAtNextInvocation<MLFloat16>();
}

TEST(MoeExpertCountingTest, AdaptiveCpuOffloadPublishesCompletedBf16SwapsAtNextInvocation) {
  RunAdaptiveCpuOffloadPublishesCompletedSwapsAtNextInvocation<BFloat16>();
}

TEST(MoeExpertCountingTest, StaticCpuOffloadPreservesAliasedInputDuringHostCopy) {
  if (!HasCudaEnvironment(700)) {
    GTEST_SKIP() << "CUDA device with compute capability 7.0 or newer is required.";
  }
  auto provider = DefaultCudaExecutionProvider();
  ASSERT_NE(provider, nullptr);
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  constexpr int64_t rows = 4096;
  ModelProto numerical_model;
  ASSERT_TRUE(numerical_model.ParseFromString(MakeCountingModel(false, true, false, rows)));
  auto& graph = *numerical_model.mutable_graph();
  for (auto& tensor : *graph.mutable_initializer()) {
    std::vector<MLFloat16> weights(static_cast<size_t>(kExperts * kWidth * kWidth), MLFloat16(0.0f));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      for (int64_t column = 0; column < kWidth; ++column) {
        weights[static_cast<size_t>(expert * kWidth * kWidth + column * kWidth + column)] =
            MLFloat16(tensor.name() == "w1" ? 1.0f : 2.0f);
      }
    }
    tensor.set_raw_data(weights.data(), weights.size() * sizeof(MLFloat16));
  }
  for (auto& node : *graph.mutable_node()) {
    auto* attribute = node.add_attribute();
    attribute->set_name("normalize_routing_weights");
    attribute->set_type(AttributeProto_AttributeType_INT);
    attribute->set_i(1);
  }
  // Keep the second MoE output internal so MayInplace can reuse its input allocation.
  graph.mutable_node(1)->set_output(0, "mixed_output");
  SetValue(*graph.add_value_info(), "mixed_output", TensorProto_DataType_FLOAT16, {rows, kWidth});
  auto* identity = graph.add_node();
  identity->set_op_type("Identity");
  identity->add_input("mixed_output");
  identity->add_output("output");

  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "4"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = numerical_model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const auto& state = session.GetSessionState();
  auto* expert_state = state.GetMoeExpertState();
  ASSERT_NE(expert_state, nullptr);
  for (int node = 0; node < 2; ++node) {
    EXPECT_EQ(CudaExperts(*expert_state, state.GetKernel(node)), (InlinedVector<int>{0, 1}));
  }
  int input_index = -1;
  int output_index = -1;
  ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("intermediate", input_index));
  ASSERT_STATUS_OK(state.GetOrtValueNameIdxMap().GetIdx("mixed_output", output_index));
  const auto* plan = state.GetExecutionPlan();
  ASSERT_NE(plan, nullptr);
  const auto& output_allocation = plan->allocation_plan[static_cast<size_t>(output_index)];
  ASSERT_EQ(output_allocation.alloc_kind, AllocKind::kReuse);
  ASSERT_EQ(output_allocation.reused_buffer, input_index);

  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  std::vector<MLFloat16> values(static_cast<size_t>(rows * kWidth));
  std::vector<MLFloat16> routing(static_cast<size_t>(rows * kExperts), MLFloat16(0.0f));
  for (int64_t row = 0; row < rows; ++row) {
    routing[static_cast<size_t>(row * kExperts + (row % 2 == 0 ? 0 : 3))] = MLFloat16(9.0f);
    for (int64_t column = 0; column < kWidth; ++column) {
      values[static_cast<size_t>(row * kWidth + column)] = MLFloat16(static_cast<float>(1 + column % 4));
    }
  }
  OrtValue input, router;
  CreateMLValue<MLFloat16>(allocator, {rows, kWidth}, values, &input);
  CreateMLValue<MLFloat16>(allocator, {rows, kExperts}, routing, &router);
  NameMLValMap feeds{{"input", input}, {"router", router}};
  const std::array<std::string, 1> output_names{"output"};
  for (int iteration = 0; iteration < 3; ++iteration) {
    SCOPED_TRACE(iteration);
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &outputs));
    ASSERT_EQ(outputs.size(), 1U);
    const auto result = outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>();
    ASSERT_EQ(result.size(), values.size());
    for (size_t index = 0; index < result.size(); ++index) {
      ASSERT_NEAR(result[index].ToFloat(), 4.0f * values[index].ToFloat(), 0.01f) << "index=" << index;
    }
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadRequiresEnabledPrepacking) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  const auto model = MakeCountingModel(false, true);
  for (const char* offload_count : {"0", "1"}) {
    for (const char* disable_prepacking : {"0", "1"}) {
      SCOPED_TRACE(MakeString("offload_count=", offload_count, ", disable_prepacking=", disable_prepacking));
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                             offload_count));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigDisablePrepacking,
                                                             disable_prepacking));
      InferenceSessionWrapper session(options, GetEnvironment());
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
      ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
      const auto status = session.Initialize();
      if (offload_count[0] == '1' && disable_prepacking[0] == '1') {
        ASSERT_FALSE(status.IsOK());
        EXPECT_NE(status.ErrorMessage().find(
                      MakeString(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                 " requires prepacking to remain enabled.")),
                  std::string::npos);
      } else {
        ASSERT_STATUS_OK(status);
      }
    }
  }
}

TEST(MoeExpertCountingTest, StaticCpuOffloadRejectsCountAboveEligibleExperts) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "CUDA execution provider is unavailable.";
  }
  if (provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "MoE CPU offload is not supported by the CUDA plugin execution provider.";
  }

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "9"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = MakeCountingModel(false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  const auto status = session.Initialize();
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("contain only 8 experts"), std::string::npos);
}
#endif

TEST(MoeExpertCountingTest, InvalidConfigurationFailsInitialization) {
  for (const auto& entry : {
           std::pair{kOrtSessionOptionsConfigEnableMoeExpertCounting, "true"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterStateFile, "missing.txt"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "0.5"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterBeta, "2"},
           std::pair{kOrtSessionOptionsConfigMoeExpertSwapEpsilon, "0.1"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadExperts, "-1"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadExperts, "invalid"}}) {
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first, entry.second));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }
  for (const char* path : {"", "missing_moe_expert_counter_state.txt"}) {
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }
  for (const auto& entry : {
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "-0.1"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "1.1"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "0.95"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "nan"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterBeta, "-0.1"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterBeta, "0.2"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterBeta, "inf"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterBeta, "invalid"}}) {
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first, entry.second));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }
}
#endif
}  // namespace onnxruntime::test
