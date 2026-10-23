// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <fstream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/framework/customregistry.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test/capturing_sink.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/file_util.h"

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_provider_options.h"
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
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterAlpha, "0.9"));
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

template <typename T>
ModelProto MakeDemandSwapModel(int64_t rows, int64_t top_k, int64_t bits, bool swiglu) {
  ModelProto model;
  ORT_ENFORCE(model.ParseFromString(MakeCountingModel(true, true, false, rows)));
  auto& graph = *model.mutable_graph();
  graph.mutable_node()->RemoveLast();
  auto& node = *graph.mutable_node(0);
  node.set_output(0, "output");
  node.set_input(4, "b1");
  node.add_input("b2");
  for (auto& attribute : *node.mutable_attribute()) {
    if (attribute.name() == "k") attribute.set_i(top_k);
    if (attribute.name() == "activation_type") attribute.set_s(swiglu ? "swiglu" : "relu");
  }
  for (const auto& [name, value] : {std::pair{"expert_weight_bits", bits},
                                    std::pair{"weights_prepacked", int64_t{0}},
                                    std::pair{"block_size", int64_t{32}},
                                    std::pair{"normalize_routing_weights", int64_t{1}}}) {
    auto* attribute = node.add_attribute();
    attribute->set_name(name);
    attribute->set_type(AttributeProto_AttributeType_INT);
    attribute->set_i(value);
  }
  constexpr bool bf16 = std::is_same_v<T, BFloat16>;
  const int type = bf16 ? TensorProto_DataType_BFLOAT16 : TensorProto_DataType_FLOAT16;
  for (auto& value : *graph.mutable_input()) value.mutable_type()->mutable_tensor_type()->set_elem_type(type);
  graph.mutable_output(0)->mutable_type()->mutable_tensor_type()->set_elem_type(type);
  const int64_t fc1_rows = swiglu ? 2 * kWidth : kWidth;
  for (auto& tensor : *graph.mutable_initializer()) {
    const bool first = tensor.name() == "w1" || tensor.name() == "s1";
    const bool weight = tensor.name() == "w1" || tensor.name() == "w2";
    const int64_t outputs = first ? fc1_rows : kWidth;
    tensor.clear_dims();
    for (auto dim : {kExperts, outputs, weight ? kWidth / (8 / bits) : kWidth / 32}) tensor.add_dims(dim);
    const size_t count = static_cast<size_t>(outputs * tensor.dims(2));
    if (weight) {
      std::string values(static_cast<size_t>(kExperts) * count, '\0');
      for (int64_t expert = 0; expert < kExperts; ++expert) {
        std::fill_n(values.begin() + expert * count, count,
                    static_cast<char>(bits == 4 ? (9 + expert) * 17 : 1 + expert));
      }
      tensor.set_raw_data(values);
    } else {
      tensor.set_data_type(type);
      const std::vector<T> scales(static_cast<size_t>(kExperts) * count, T(0.0078125f));
      tensor.set_raw_data(scales.data(), scales.size() * sizeof(T));
    }
  }
  for (const auto& [name, width] : {std::pair{"b1", fc1_rows}, std::pair{"b2", kWidth}}) {
    AddZeroInitializer(graph, name, type, {kExperts, width}, sizeof(T));
    const std::vector<T> bias(static_cast<size_t>(kExperts * width), T(0.0009765625f));
    graph.mutable_initializer(graph.initializer_size() - 1)->set_raw_data(bias.data(), bias.size() * sizeof(T));
  }
  return model;
}

template <typename T>
NameMLValMap DemandSwapFeeds(int64_t rows, gsl::span<const int> experts, bool different_last_row = false) {
  auto allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtValue input, router;
  CreateMLValue<T>(allocator, {rows, kWidth}, std::vector<T>(static_cast<size_t>(rows * kWidth), T(0.125f)), &input);
  std::vector<T> routing(static_cast<size_t>(rows * kExperts), T(0.f));
  for (int64_t row = 0; row < rows; ++row) {
    for (size_t i = 0; i < experts.size(); ++i) {
      const int expert = different_last_row ? (row + 1 == rows ? experts.back() : experts.front()) : experts[i];
      routing[static_cast<size_t>(row * kExperts + expert)] = T(9.f - static_cast<float>(i));
    }
  }
  CreateMLValue<T>(allocator, {rows, kExperts}, routing, &router);
  return {{"input", input}, {"router", router}};
}

template <typename T>
void TestDemandSwaps() {
  const std::vector<std::string> output_names{"output"};
  struct Case {
    int64_t rows, top_k, cpu, threshold;
    InlinedVector<int> selected, expected;
    bool different_last_row{false};
  };
  for (const char* strategy : {"adaptive", "layered"}) {
    SCOPED_TRACE(strategy);
    for (bool swiglu : {false, true}) {
      for (int64_t bits : {4, 8}) {
        if (swiglu && bits == 8) continue;
        for (const auto& test : {
                 Case{1, 1, 1, 1, {3}, {1, 2, 3}},
                 Case{3, 1, 1, 1, {3}, {1, 2, 3}},
                 Case{3, 2, 2, 2, {2, 3}, {2, 3}},
                 Case{3, 2, 2, 1, {2, 3}, {0, 1}},
                 Case{3, 2, 3, 1, {0, 3}, {0}},
                 Case{1, 1, 4, 1, {3}, {}},
                 Case{1, 1, 1, 0, {3}, {0, 1, 2}},
                 Case{3, 2, 1, 1, {0, 2}, {0, 1, 2}},
                 Case{257, 1, 2, 1, {2, 3}, {0, 1}, true},
                 Case{257, 1, 2, 2, {2, 3}, {2, 3}, true}}) {
          SCOPED_TRACE(MakeString("rows=", test.rows, ", top_k=", test.top_k, ", cpu=", test.cpu,
                                  ", threshold=", test.threshold, ", bits=", bits, ", swiglu=", swiglu));
          const auto model = MakeDemandSwapModel<T>(test.rows, test.top_k, bits, swiglu).SerializeAsString();
          auto options = CountingOptions();
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
          InferenceSessionWrapper reference(options, GetEnvironment());
          ASSERT_STATUS_OK(reference.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
          ASSERT_STATUS_OK(reference.Load(model.data(), static_cast<int>(model.size())));
          ASSERT_STATUS_OK(reference.Initialize());
          const auto feeds = DemandSwapFeeds<T>(test.rows, test.selected, test.different_last_row);
          std::vector<OrtValue> expected;
          ASSERT_STATUS_OK(reference.Run(RunOptions{}, feeds, output_names, &expected));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                                 std::to_string(test.cpu).c_str()));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold,
                                                                 std::to_string(test.threshold).c_str()));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "0"));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, strategy));
          if (std::string_view(strategy) == "layered") {
            ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadGpuRatio, "1"));
          }
          InferenceSessionWrapper session(options, GetEnvironment());
          ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
          ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
          ASSERT_STATUS_OK(session.Initialize());
          auto* state = session.GetSessionState().GetMoeExpertState();
          const auto* kernel = session.GetSessionState().GetKernel(0);
          ASSERT_NE(state, nullptr);
          ASSERT_NE(kernel, nullptr);
          for (int run = 0; run < 2; ++run) {
            std::vector<OrtValue> actual;
            ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &actual));
            ASSERT_EQ(actual.size(), 1U);
            EXPECT_EQ(CudaExperts(*state, kernel), test.expected);
            const auto values = actual[0].Get<Tensor>().DataAsSpan<T>();
            const auto reference_values = expected[0].Get<Tensor>().DataAsSpan<T>();
            ASSERT_EQ(values.size(), reference_values.size());
            for (size_t i = 0; i < values.size(); ++i) {
              EXPECT_GT(reference_values[i].ToFloat(), 0.f);
              // CPU accumulation uses FP32; CUDA rounds intermediate BF16 activations.
              const float tolerance = std::is_same_v<T, BFloat16>
                                          ? 0.0001f + 0.008f * std::abs(reference_values[i].ToFloat())
                                          : 0.003f;
              EXPECT_NEAR(values[i].ToFloat(), reference_values[i].ToFloat(), tolerance);
            }
          }
        }
      }
    }
  }
}
#endif

#if defined(USE_CUDA)
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
#endif

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
  for (int configuration = 0; configuration < 5; ++configuration) {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    if (configuration == 1) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertCounting, "0"));
    } else if (configuration == 2) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertCounting, "1"));
    } else if (configuration == 3) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
    } else if (configuration == 4) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
          kOrtSessionOptionsConfigMoeExpertPopularityFile, "unused_moe_popularity.json"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(
          kOrtSessionOptionsConfigMoeExpertCounterStateFile, "missing_moe_expert_counter_state.txt"));
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
  ASSERT_EQ(state, nullptr);
  const int64_t width = use_packed_int2_gemv ? kPackedInt2GemvWidth : kWidth;
  for (int run = 1; run <= 2; ++run) {
    RunCountingModel(session, false, true, rows, width);
  }
}
void TestPopularityOnly(bool quantized, bool cuda) {
  const std::string path = MakeString("moe_popularity_only_", cuda ? "cuda_" : "cpu_", quantized ? "qmoe" : "moe", ".json");
  auto cleanup = gsl::finally([&]() { std::remove(path.c_str()); });
  {
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertPopularityFile, path.c_str()));
    if (cuda) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    }
    InferenceSessionWrapper session(options, GetEnvironment());
    if (cuda) {
      auto provider = DefaultCudaExecutionProvider();
      if (!provider || GetCudaArchitecture() < 800) {
        GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
      }
      if (provider->GetOrtEp() != nullptr) {
        GTEST_SKIP() << "MoE expert counting is not supported by the CUDA plugin execution provider.";
      }
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    }
    const auto model = MakeCountingModel(quantized, cuda);
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    ASSERT_EQ(session.GetSessionState().GetMoeExpertState(), nullptr);
    RunCountingModel(session);
    RunCountingModel(session);
  }
  std::ifstream input(path);
  EXPECT_FALSE(input.is_open());
}
}  // namespace

TEST(MoeExpertCountingTest, CpuMoENoCountersWithoutOffload) { TestCounting(false, false); }
TEST(MoeExpertCountingTest, CpuQMoENoCountersWithoutOffload) { TestCounting(true, false); }
TEST(MoeExpertCountingTest, CpuMoEDisabledDoesNotRecordUsage) { TestDisabledRecording(false); }
TEST(MoeExpertCountingTest, CpuQMoEDisabledDoesNotRecordUsage) { TestDisabledRecording(true); }
TEST(MoeExpertCountingTest, CpuMoEPopularityOutputRequiresOffload) { TestPopularityOnly(false, false); }
TEST(MoeExpertCountingTest, CpuQMoEPopularityOutputRequiresOffload) { TestPopularityOnly(true, false); }

TEST(MoeExpertCountingTest, ContextHasNoCollectorWithoutOffload) {
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  const auto model = MakeCountingModel();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto& session_state = session.GetSessionState();
  auto* state = session_state.GetMoeExpertState();
  ASSERT_EQ(state, nullptr);
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
  EXPECT_EQ(pilot, nullptr);
  ASSERT_STATUS_OK(context.RecordKernelUsage());
}

TEST(MoeExpertCountingTest, CustomKernelHasNoCollectorWithoutOffload) {
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
  EXPECT_NE(status.ErrorMessage().find("Missing test kernel collector."), std::string::npos);
  EXPECT_EQ(session.GetSessionState().GetMoeExpertState(), nullptr);
}

#if defined(USE_CUDA)
TEST(MoeExpertCountingTest, WeightedOffloadCountersExcludeCpuKernels) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Built-in CUDA execution provider with compute capability 8.0 or newer is required.";
  }
  auto options = CountingOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "0.5"));
  InferenceSessionWrapper session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto model = MakeCountingModel(false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  session.GetMutableGraph().GetNode(0)->SetExecutionProviderType(kCpuExecutionProvider);
  ASSERT_STATUS_OK(session.Initialize());
  const auto& session_state = session.GetSessionState();
  auto* state = session_state.GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(session_state.GetKernel(0)->Node().GetExecutionProviderType(), kCpuExecutionProvider);
  ASSERT_EQ(session_state.GetKernel(1)->Node().GetExecutionProviderType(), kCudaExecutionProvider);
  EXPECT_EQ(state->TotalExpertCount(), static_cast<size_t>(kExperts));
  EXPECT_EQ(session_state.GetKernelPilot(session_state.GetKernel(0)), nullptr);
  ASSERT_NE(session_state.GetKernelPilot(session_state.GetKernel(1)), nullptr);
  RunCountingModel(session);
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state->GetCounters(session_state.GetKernel(1), counters));
  ASSERT_EQ(counters.size(), static_cast<size_t>(kExperts));
  EXPECT_GT(counters[0], 0.0);
  EXPECT_DOUBLE_EQ(counters[1], 0.0);
  EXPECT_GT(counters[2], 0.0);
  EXPECT_DOUBLE_EQ(counters[3], 0.0);
}

TEST(MoeExpertCountingTest, WeightedOffloadCountersExcludeFloatCudaKernels) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Built-in CUDA execution provider with compute capability 8.0 or newer is required.";
  }
  auto sink = std::make_unique<CapturingSink>();
  const auto* captured = sink.get();
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::move(sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  std::unique_ptr<Environment> environment;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), environment));
  auto options = CountingOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "0.5"));
  ModelProto model;
  ASSERT_TRUE(model.ParseFromString(MakeCountingModel(false, true)));
  auto& graph = *model.mutable_graph();
  SetValue(*graph.add_input(), "float_input", TensorProto_DataType_FLOAT, {3, kWidth});
  SetValue(*graph.add_input(), "float_router", TensorProto_DataType_FLOAT, {3, kExperts});
  SetValue(*graph.add_output(), "float_output", TensorProto_DataType_FLOAT, {3, kWidth});
  AddZeroInitializer(graph, "float_w1", TensorProto_DataType_FLOAT, {kExperts, kWidth, kWidth}, sizeof(float));
  AddZeroInitializer(graph, "float_w2", TensorProto_DataType_FLOAT, {kExperts, kWidth, kWidth}, sizeof(float));
  auto* float_node = graph.mutable_node(0);
  float_node->set_input(0, "float_input");
  float_node->set_input(1, "float_router");
  float_node->set_input(2, "float_w1");
  float_node->set_input(4, "float_w2");
  float_node->set_output(0, "float_output");
  graph.mutable_node(1)->set_input(0, "input");

  InferenceSessionWrapper session(options, *environment);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  const auto serialized = model.SerializeAsString();
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto& session_state = session.GetSessionState();
  auto* state = session_state.GetMoeExpertState();
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(session_state.GetKernel(0)->Node().GetExecutionProviderType(), kCudaExecutionProvider);
  ASSERT_EQ(session_state.GetKernel(1)->Node().GetExecutionProviderType(), kCudaExecutionProvider);
  EXPECT_EQ(state->TotalExpertCount(), static_cast<size_t>(kExperts));
  EXPECT_EQ(session_state.GetKernelPilot(session_state.GetKernel(0)), nullptr);
  ASSERT_NE(session_state.GetKernelPilot(session_state.GetKernel(1)), nullptr);

  auto feeds = CountingFeeds();
  auto float_feeds = CountingFeedsTyped<float>();
  feeds.emplace("float_input", float_feeds.at("input"));
  feeds.emplace("float_router", float_feeds.at("router"));
  RunOptions run_options;
  run_options.run_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  std::vector<OrtValue> outputs;
  const std::array<std::string, 2> output_names{"output", "float_output"};
  ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &outputs));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state->GetCounters(session_state.GetKernel(1), counters));
  ASSERT_EQ(counters.size(), static_cast<size_t>(kExperts));
  EXPECT_GT(counters[0], 0.0);
  EXPECT_DOUBLE_EQ(counters[1], 0.0);
  EXPECT_GT(counters[2], 0.0);
  EXPECT_DOUBLE_EQ(counters[3], 0.0);
  size_t counter_events = 0;
  for (const auto& message : captured->Messages()) {
    if (message.find("moe_expert_counters ") != std::string::npos) {
      EXPECT_NE(message.find("\"node_index\":1"), std::string::npos);
      EXPECT_NE(message.find("\"moe_count\":1"), std::string::npos);
      EXPECT_NE(message.find("\"total_expert_count\":4"), std::string::npos);
      ++counter_events;
    }
  }
  EXPECT_EQ(counter_events, 1U);
}

TEST(MoeExpertCountingTest, CpuOffloadSkipsPrepackedQMoE) {
  for (int64_t bits : {4, 8}) {
    for (int64_t prepacked_mode : {-1, 1}) {
      SCOPED_TRACE(MakeString("bits=", bits, ", weights_prepacked=", prepacked_mode));
      auto provider = DefaultCudaExecutionProvider();
      if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
        GTEST_SKIP() << "Built-in CUDA execution provider with compute capability 8.0 or newer is required.";
      }
      auto options = CountingOptions();
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "0.5"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      ModelProto model;
      ASSERT_TRUE(model.ParseFromString(MakeCountingModel(true, true)));
      auto& graph = *model.mutable_graph();
      for (auto& initializer : *graph.mutable_initializer()) {
        if (initializer.data_type() == TensorProto_DataType_UINT8) {
          const int64_t packed_width = kWidth / (8 / bits);
          initializer.set_dims(2, packed_width);
          initializer.set_raw_data(std::string(kExperts * kWidth * packed_width, '\0'));
        }
      }
      for (int i = 0; i < graph.node_size(); ++i) {
        auto& node = *graph.mutable_node(i);
        auto* attribute = node.add_attribute();
        attribute->set_name("expert_weight_bits");
        attribute->set_type(AttributeProto_AttributeType_INT);
        attribute->set_i(bits);
        if (i == 0 || prepacked_mode != -1) {
          attribute = node.add_attribute();
          attribute->set_name("weights_prepacked");
          attribute->set_type(AttributeProto_AttributeType_INT);
          attribute->set_i(i == 0 ? 0 : prepacked_mode);
        }
      }
      InferenceSessionWrapper session(options, GetEnvironment());
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
      const auto serialized = model.SerializeAsString();
      ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
      ASSERT_STATUS_OK(session.Initialize());
      const auto& session_state = session.GetSessionState();
      auto* state = session_state.GetMoeExpertState();
      ASSERT_NE(state, nullptr);
      const auto* canonical = session_state.GetKernel(0);
      const auto* prepacked = session_state.GetKernel(1);
      ASSERT_NE(canonical, nullptr);
      ASSERT_NE(prepacked, nullptr);
      ASSERT_EQ(canonical->Node().GetExecutionProviderType(), kCudaExecutionProvider);
      ASSERT_EQ(prepacked->Node().GetExecutionProviderType(), kCudaExecutionProvider);
      EXPECT_EQ(state->TotalExpertCount(), static_cast<size_t>(kExperts));
      ASSERT_NE(session_state.GetKernelPilot(canonical), nullptr);
      EXPECT_EQ(session_state.GetKernelPilot(prepacked), nullptr);
      EXPECT_EQ(CudaExperts(*state, canonical), InlinedVector<int>({0}));
      RunCountingModel(session);
      InlinedVector<double> counters;
      ASSERT_STATUS_OK(state->GetCounters(canonical, counters));
      ASSERT_EQ(counters.size(), static_cast<size_t>(kExperts));
      EXPECT_GT(counters[0], 0.0);
      EXPECT_DOUBLE_EQ(counters[1], 0.0);
      EXPECT_GT(counters[2], 0.0);
      EXPECT_DOUBLE_EQ(counters[3], 0.0);
    }
  }
}

#if !defined(ORT_NO_EXCEPTIONS)
TEST(MoeExpertCountingTest, FailedRunKeepsLoggerAliveForDeferredUsage) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }

  class FailingRunEndProvider final : public CPUExecutionProvider {
   public:
    FailingRunEndProvider() : CPUExecutionProvider(CPUExecutionProviderInfo{}) {}

    Status OnRunEnd(bool, const RunOptions&) override {
      ORT_ENFORCE(!fail, "Intentional failure before deferred MoE usage is committed.");
      return Status::OK();
    }

    bool fail{true};
  };

  auto sink = std::make_unique<CapturingSink>();
  const auto* captured = sink.get();
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::move(sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  std::unique_ptr<Environment> environment;
  ASSERT_STATUS_OK(Environment::Create(std::move(logging_manager), environment));
  auto options = CountingOptions();
  options.session_logid = "deferred_moe_logger";
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigEnableMoeExpertStatistics, "1"));
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
  InferenceSessionWrapper session(options, *environment);
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
  auto failing_provider = std::make_unique<FailingRunEndProvider>();
  auto* failure_control = failing_provider.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(failing_provider)));
  const auto model = MakeCountingModel(false, true);
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  RunOptions run_options;
  run_options.run_tag = "failed_run";
  run_options.run_log_severity_level = static_cast<int>(logging::Severity::kINFO);
  std::vector<OrtValue> outputs;
  const std::array<std::string, 1> output_names{"output"};
  const auto status = session.Run(run_options, CountingFeeds(), output_names, &outputs);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("Intentional failure before deferred MoE usage"), std::string::npos);
  size_t counter_events = 0;
  for (const auto& message : captured->Messages()) {
    if (message.find("moe_expert_counters ") != std::string::npos) {
      EXPECT_NE(message.find("deferred_moe_logger:failed_run"), std::string::npos);
      ++counter_events;
    }
  }
  EXPECT_EQ(counter_events, 2U);
  for (const auto& [kernel, counters] : CountersByKernel(*session.GetSessionState().GetMoeExpertState())) {
    EXPECT_EQ(counters, (InlinedVector<double>{1.0 - 0.9, 0, 1.0 - 0.9, 0}));
  }

  failure_control->fail = false;
  RunCountingModel(session);
}
#endif

TEST(MoeExpertCountingTest, CudaMoENoCountersWithoutOffload) { TestCounting(false, true); }
TEST(MoeExpertCountingTest, CudaQMoENoCountersWithoutOffload) { TestCounting(true, true); }
TEST(MoeExpertCountingTest, CudaQMoETiledNoCountersWithoutOffload) { TestCounting(true, true, true); }
TEST(MoeExpertCountingTest, CudaQMoESingleTokenNoCountersWithoutOffload) { TestCounting(true, true, false, 1); }
TEST(MoeExpertCountingTest, CudaMoEPopularityOutputRequiresOffload) { TestPopularityOnly(false, true); }
TEST(MoeExpertCountingTest, CudaQMoEPopularityOutputRequiresOffload) { TestPopularityOnly(true, true); }
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
  ASSERT_EQ(state, nullptr);
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
  EXPECT_EQ(first.GetSessionState().GetMoeExpertState(), nullptr);
  EXPECT_EQ(second.GetSessionState().GetMoeExpertState(), nullptr);
}

TEST(MoeExpertCountingTest, SubgraphsHaveNoCountersWithoutOffload) {
  const auto model = MakeCountingModel(false, false, true);
  InferenceSessionWrapper session(CountingOptions(), GetEnvironment());
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  const auto* state = session.GetSessionState().GetMoeExpertState();
  ASSERT_EQ(state, nullptr);
  for (const auto& [node, subgraphs] : session.GetSessionState().GetSubgraphSessionStateMap()) {
    for (const auto& [attribute, subgraph] : subgraphs) {
      EXPECT_EQ(subgraph->GetMoeExpertState(), state);
    }
  }
  RunCountingModel(session, true, true);
  RunCountingModel(session, true, false);
}

TEST(MoeExpertCountingTest, CounterAlphaDoesNotEnableCountersWithoutOffload) {
  const auto model = MakeCountingModel();
  for (const char* alpha : {"0.5", "0", "1"}) {
    SCOPED_TRACE(MakeString("alpha=", alpha));
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterAlpha, alpha));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    RunCountingModel(session);
    RunCountingModel(session);
    EXPECT_EQ(session.GetSessionState().GetMoeExpertState(), nullptr);
  }
}

TEST(MoeExpertCountingTest, DoesNotLoadStateFileWithoutOffload) {
  const char* path = "moe_expert_counting_initial_state.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0.9,"epsilon":0.01,"experts":[{"moe":0,"expert":0,"popularity":3.5}]})";
    ASSERT_TRUE(file.good());
  }
  auto options = CountingOptions();
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
  InferenceSessionWrapper session(options, GetEnvironment());
  const auto model = MakeCountingModel();
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());
  RunCountingModel(session);
  EXPECT_EQ(session.GetSessionState().GetMoeExpertState(), nullptr);
}

#if defined(USE_CUDA)
TEST(MoeExpertCountingTest, CudaCpuOffloadUsesLoadedAlphaForMoEAndQMoE) {
  if (GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "CUDA device with compute capability 8.0 or newer is required.";
  }
  const char* path = "moe_cpu_offload_loaded_alpha.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0.4,"epsilon":0.2,"experts":[)"
            R"({"moe":0,"expert":0,"popularity":0.25},)"
            R"({"moe":1,"expert":0,"popularity":0.25}]})";
    ASSERT_TRUE(file.good());
  }
  for (bool quantized : {false, true}) {
    SCOPED_TRACE(quantized ? "QMoE" : "MoE");
    auto provider = DefaultCudaExecutionProvider();
    if (!provider || provider->GetOrtEp() != nullptr) {
      GTEST_SKIP() << "Built-in CUDA execution provider is required.";
    }
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "0"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    ModelProto counting_model;
    ASSERT_TRUE(counting_model.ParseFromString(MakeCountingModel(quantized, true)));
    if (quantized) {
      for (auto& node : *counting_model.mutable_graph()->mutable_node()) {
        for (const auto& [name, value] :
             {std::pair{"expert_weight_bits", int64_t{4}}, std::pair{"weights_prepacked", int64_t{0}}}) {
          auto* attribute = node.add_attribute();
          attribute->set_name(name);
          attribute->set_type(AttributeProto_AttributeType_INT);
          attribute->set_i(value);
        }
      }
    }
    const auto model = counting_model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    auto* state = session.GetSessionState().GetMoeExpertState();
    ASSERT_NE(state, nullptr);
    for (size_t node_index : {size_t{0}, size_t{1}}) {
      const auto* kernel = session.GetSessionState().GetKernel(node_index);
      ASSERT_NE(kernel, nullptr);
      EXPECT_EQ(kernel->Node().GetExecutionProviderType(), kCudaExecutionProvider);
      EXPECT_DOUBLE_EQ(state->GetKernelPilot(kernel)->MoeExpertCounterAlpha(), 0.4);
    }
    for (int run = 1; run <= 2; ++run) {
      RunCountingModel(session);
      for (const auto& [kernel, counters] : CountersByKernel(*state)) {
        ASSERT_EQ(counters.size(), 4U);
        EXPECT_NEAR(counters[0], 0.25 * std::pow(0.4, run) + 1.0 - std::pow(0.4, run), 1e-12);
        EXPECT_DOUBLE_EQ(counters[1], 0.0);
        EXPECT_NEAR(counters[2], 1.0 - std::pow(0.4, run), 1e-12);
        EXPECT_DOUBLE_EQ(counters[3], 0.0);
      }
    }
  }
}

TEST(MoeExpertCountingTest, QMoESynchronousDemandSwaps) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Built-in CUDA on SM80 or newer is required.";
  }
  TestDemandSwaps<MLFloat16>();
}

#if defined(ENABLE_BF16) && !defined(ORT_QUICK_BUILD)
TEST(MoeExpertCountingTest, QMoESynchronousDemandSwapsBFloat16) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Built-in CUDA on SM80 or newer is required.";
  }
  TestDemandSwaps<BFloat16>();
}
#endif

TEST(MoeExpertCountingTest, QMoEDemandSwapsFinishBackgroundSwapAndPreserveCpuWeights) {
  const std::vector<std::string> output_names{"output"};
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr || GetCudaArchitecture() < 800) {
    GTEST_SKIP() << "Built-in CUDA on SM80 or newer is required.";
  }
  for (bool swiglu : {false, true}) {
    const auto model = MakeDemandSwapModel<MLFloat16>(3, 1, 4, swiglu).SerializeAsString();
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    InferenceSessionWrapper reference(options, GetEnvironment());
    ASSERT_STATUS_OK(reference.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(reference.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(reference.Initialize());
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "2"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "1"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    ASSERT_STATUS_OK(session.Initialize());
    auto* state = session.GetSessionState().GetMoeExpertState();
    const auto* kernel = session.GetSessionState().GetKernel(0);
    ASSERT_NE(state, nullptr);
    for (const auto& experts : {InlinedVector<int>{2, 3}, InlinedVector<int>{3}, InlinedVector<int>{0, 1}}) {
      const auto feeds = DemandSwapFeeds<MLFloat16>(3, experts, experts.size() > 1);
      std::vector<OrtValue> expected, actual;
      ASSERT_STATUS_OK(reference.Run(RunOptions{}, feeds, output_names, &expected));
      ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, output_names, &actual));
      const auto values = actual[0].Get<Tensor>().DataAsSpan<MLFloat16>();
      const auto reference_values = expected[0].Get<Tensor>().DataAsSpan<MLFloat16>();
      for (size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(values[i].ToFloat(), reference_values[i].ToFloat(), 0.003f);
      }
      if (experts.size() == 1) {
        EXPECT_EQ(CudaExperts(*state, kernel), (InlinedVector<int>{2, 3}));
      } else {
        EXPECT_EQ(CudaExperts(*state, kernel),
                  experts.front() == 2 ? (InlinedVector<int>{0, 1}) : (InlinedVector<int>{2, 3}));
      }
    }
  }
}

TEST(MoeExpertCountingTest, DemandSwapsRejectBasicAndDenseMoe) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  for (bool basic : {false, true}) {
    SCOPED_TRACE(basic);
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy,
                                                           basic ? "basic" : "adaptive"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    const auto model = MakeCountingModel(basic, true);
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    const auto status = session.Initialize();
    EXPECT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(basic ? "requires CPU offload with the adaptive or layered strategy"
                                               : "supports CUDA QMoE only"),
              std::string::npos)
        << status.ErrorMessage();
  }
}

TEST(MoeExpertCountingTest, BasicQMoEOffloadFillsEarlierNodesAndNeverSwaps) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  const char* path = "qmoe_basic_cpu_offload_initial_state.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0.9,"epsilon":0,"experts":[{"moe":1,"expert":2,"popularity":100}]})";
    ASSERT_TRUE(file.good());
  }
  for (bool reverse_node_indices : {false, true}) {
    for (bool load_counters : {false, true}) {
      for (size_t cpu_count : {1U, 3U, 4U, 5U, 8U}) {
        SCOPED_TRACE(MakeString("reverse=", reverse_node_indices, ", state=", load_counters,
                                ", cpu_count=", cpu_count));
        auto options = CountingOptions();
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, "basic"));
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                               std::to_string(cpu_count).c_str()));
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "5"));
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
        if (load_counters) {
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
        }
        ModelProto model;
        ASSERT_TRUE(model.ParseFromString(MakeCountingModel(true, true)));
        for (auto& node : *model.mutable_graph()->mutable_node()) {
          for (const auto& [name, value] :
               {std::pair{"expert_weight_bits", int64_t{4}}, std::pair{"weights_prepacked", int64_t{0}}}) {
            auto* attribute = node.add_attribute();
            attribute->set_name(name);
            attribute->set_type(AttributeProto_AttributeType_INT);
            attribute->set_i(value);
          }
        }
        if (reverse_node_indices) {
          model.mutable_graph()->mutable_node()->SwapElements(0, 1);
        }
        InferenceSessionWrapper session(options, GetEnvironment());
        ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
        const auto serialized = model.SerializeAsString();
        ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
        ASSERT_STATUS_OK(session.Initialize());
        auto* state = session.GetSessionState().GetMoeExpertState();
        ASSERT_NE(state, nullptr);
        const auto* first = session.GetSessionState().GetKernel(reverse_node_indices ? 1 : 0);
        const auto* last = session.GetSessionState().GetKernel(reverse_node_indices ? 0 : 1);
        ASSERT_EQ(first->Node().Name(), "moe0");
        ASSERT_EQ(last->Node().Name(), "moe1");
        const size_t cuda_count = 2 * kExperts - cpu_count;
        const size_t first_count = std::min(cuda_count, static_cast<size_t>(kExperts));
        InlinedVector<int> expected_first, expected_last;
        for (size_t i = 0; i < first_count; ++i) expected_first.push_back(static_cast<int>(i));
        for (size_t i = 0; i < cuda_count - first_count; ++i) expected_last.push_back(static_cast<int>(i));
        EXPECT_EQ(CudaExperts(*state, first), expected_first);
        EXPECT_EQ(CudaExperts(*state, last), expected_last);
        for (int run = 0; run < 3; ++run) {
          RunCountingModel(session);
          ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
          EXPECT_EQ(CudaExperts(*state, first), expected_first);
          EXPECT_EQ(CudaExperts(*state, last), expected_last);
        }
      }
    }
  }
}

TEST(MoeExpertCountingTest, LayeredQMoEOffloadSwapsInEarlyAndBoundaryNodesOnly) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  const char* path = "qmoe_layered_cpu_offload_initial_state.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0,"epsilon":0,"experts":[)"
            R"({"moe":0,"expert":3,"popularity":100},)"
            R"({"moe":1,"expert":3,"popularity":100}]})";
    ASSERT_TRUE(file.good());
  }
  for (bool reverse_node_indices : {false, true}) {
    for (size_t cpu_count : {3U, 5U}) {
      SCOPED_TRACE(MakeString("reverse=", reverse_node_indices, ", cpu_count=", cpu_count));
      auto options = CountingOptions();
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, "layered"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadGpuRatio, "0.8"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                             std::to_string(cpu_count).c_str()));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "5"));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterStateFile, path));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      ModelProto model;
      ASSERT_TRUE(model.ParseFromString(MakeCountingModel(true, true)));
      for (auto& node : *model.mutable_graph()->mutable_node()) {
        for (const auto& [name, value] :
             {std::pair{"expert_weight_bits", int64_t{4}}, std::pair{"weights_prepacked", int64_t{0}}}) {
          auto* attribute = node.add_attribute();
          attribute->set_name(name);
          attribute->set_type(AttributeProto_AttributeType_INT);
          attribute->set_i(value);
        }
      }
      if (reverse_node_indices) {
        model.mutable_graph()->mutable_node()->SwapElements(0, 1);
      }
      InferenceSessionWrapper session(options, GetEnvironment());
      ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
      const auto serialized = model.SerializeAsString();
      ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
      ASSERT_STATUS_OK(session.Initialize());
      auto* state = session.GetSessionState().GetMoeExpertState();
      ASSERT_NE(state, nullptr);
      const auto* first = session.GetSessionState().GetKernel(reverse_node_indices ? 1 : 0);
      const auto* last = session.GetSessionState().GetKernel(reverse_node_indices ? 0 : 1);
      EXPECT_EQ(first->Node().GetExecutionProviderType(), kCudaExecutionProvider);
      EXPECT_EQ(last->Node().GetExecutionProviderType(), kCudaExecutionProvider);
      EXPECT_EQ(CudaExperts(*state, first), (InlinedVector<int>{0, 1, 3}));
      EXPECT_EQ(CudaExperts(*state, last), cpu_count == 3 ? (InlinedVector<int>{0, 3}) : InlinedVector<int>{});
      for (int run = 0; run < 3; ++run) {
        RunCountingModel(session);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      }
      EXPECT_EQ(CudaExperts(*state, first), (InlinedVector<int>{0, 2, 3}));
      EXPECT_EQ(CudaExperts(*state, last), cpu_count == 3 ? (InlinedVector<int>{0, 2}) : InlinedVector<int>{});
    }
  }
}

TEST(MoeExpertCountingTest, BasicAndLayeredQMoEHonorFirstLastAndMiddleLayerOrder) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  for (size_t node_count : {1U, 5U, 8U}) {
    ModelProto model;
    ASSERT_TRUE(model.ParseFromString(MakeCountingModel(true, true)));
    auto prototype = model.graph().node(0);
    for (const auto& [name, value] :
         {std::pair{"expert_weight_bits", int64_t{4}}, std::pair{"weights_prepacked", int64_t{0}}}) {
      auto* attribute = prototype.add_attribute();
      attribute->set_name(name);
      attribute->set_type(AttributeProto_AttributeType_INT);
      attribute->set_i(value);
    }
    auto* nodes = model.mutable_graph()->mutable_node();
    nodes->Clear();
    for (size_t i = 0; i < node_count; ++i) {
      auto* node = nodes->Add();
      *node = prototype;
      node->set_name(MakeString("moe", i));
      node->set_input(0, i == 0 ? "input" : MakeString("intermediate", i - 1));
      node->set_output(0, i + 1 == node_count ? "output" : MakeString("intermediate", i));
    }
    for (size_t i = 0; i < node_count / 2; ++i) {
      nodes->SwapElements(static_cast<int>(i), static_cast<int>(node_count - 1 - i));
    }
    const auto serialized = model.SerializeAsString();
    for (const char* strategy : {"basic", "layered"}) {
      const size_t capacity = std::string_view(strategy) == "basic" ? 4 : 3;
      for (const char* order : {"first", "last", "middle"}) {
        InlinedVector<size_t> priority;
        if (std::string_view(order) == "middle") {
          priority = node_count == 1   ? InlinedVector<size_t>{0}
                     : node_count == 5 ? InlinedVector<size_t>{2, 3, 1, 4, 0}
                                       : InlinedVector<size_t>{3, 4, 2, 5, 1, 6, 0, 7};
        } else {
          for (size_t i = 0; i < node_count; ++i) {
            priority.push_back(std::string_view(order) == "first" ? i : node_count - 1 - i);
          }
        }
        for (size_t filled = 0; filled < node_count; ++filled) {
          SCOPED_TRACE(MakeString("nodes=", node_count, ", strategy=", strategy, ", order=", order, ", filled=", filled));
          const size_t cuda_count = filled * capacity + 1;
          auto options = CountingOptions();
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, strategy));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadLayerOrder, order));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                                 std::to_string(node_count * 4 - cuda_count).c_str()));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "0"));
          ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
          InferenceSessionWrapper session(options, GetEnvironment());
          ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
          ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
          ASSERT_STATUS_OK(session.Initialize());
          auto* state = session.GetSessionState().GetMoeExpertState();
          ASSERT_NE(state, nullptr);
          for (int run = 0; run < 2; ++run) {
            for (size_t position = 0; position < node_count; ++position) {
              const size_t layer = priority[position];
              const auto* kernel = session.GetSessionState().GetKernel(node_count - 1 - layer);
              ASSERT_NE(kernel, nullptr);
              EXPECT_EQ(kernel->Node().Name(), MakeString("moe", layer));
              EXPECT_EQ(kernel->Node().GetExecutionProviderType(), kCudaExecutionProvider);
              InlinedVector<int> expected;
              const size_t count = position < filled ? capacity : position == filled ? 1
                                                                                     : 0;
              for (size_t expert = 0; expert < count; ++expert) expected.push_back(static_cast<int>(expert));
              EXPECT_EQ(CudaExperts(*state, kernel), expected);
            }
            RunCountingModel(session);
          }
        }
      }
    }
  }
}

TEST(MoeExpertCountingTest, LayeredQMoEOffloadAllowsFewerGpuExpertsThanTopK) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  for (size_t cpu_count : {5U, 7U}) {
    SCOPED_TRACE(cpu_count);
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, "layered"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts,
                                                           std::to_string(cpu_count).c_str()));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "0"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    ModelProto model;
    ASSERT_TRUE(model.ParseFromString(MakeCountingModel(true, true)));
    for (auto& node : *model.mutable_graph()->mutable_node()) {
      for (auto& attribute : *node.mutable_attribute()) {
        if (attribute.name() == "k") attribute.set_i(kExperts);
      }
      for (const auto& [name, value] :
           {std::pair{"expert_weight_bits", int64_t{4}}, std::pair{"weights_prepacked", int64_t{0}}}) {
        auto* attribute = node.add_attribute();
        attribute->set_name(name);
        attribute->set_type(AttributeProto_AttributeType_INT);
        attribute->set_i(value);
      }
    }
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    const auto serialized = model.SerializeAsString();
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());
    auto* state = session.GetSessionState().GetMoeExpertState();
    ASSERT_NE(state, nullptr);
    const auto* first = session.GetSessionState().GetKernel(0);
    EXPECT_EQ(CudaExperts(*state, first).size(), 8 - cpu_count);
    RunCountingModel(session);
    RunCountingModel(session);
  }
}

TEST(MoeExpertCountingTest, QMoEOnlyCpuOffloadStrategiesRejectDenseMoE) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  for (const char* strategy : {"basic", "layered"}) {
    SCOPED_TRACE(strategy);
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, strategy));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
    const auto model = MakeCountingModel(false, true);
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    const auto status = session.Initialize();
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find("supports QMoE only"), std::string::npos);
  }
}

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

TEST(MoeExpertCountingTest, CpuOffloadRetainsExternalInitializersUntilCompactStoresAreReady) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider || provider->GetOrtEp() != nullptr) {
    GTEST_SKIP() << "Built-in CUDA execution provider is unavailable.";
  }
  ModelProto model;
  ASSERT_TRUE(model.ParseFromString(MakeCountingModel(false, true)));
  for (auto& node : *model.mutable_graph()->mutable_node()) {
    auto* attribute = node.add_attribute();
    attribute->set_name("normalize_routing_weights");
    attribute->set_type(AttributeProto_AttributeType_INT);
    attribute->set_i(1);
  }
  FILE* file = nullptr;
  PathString path = ORT_TSTR("moe_external_weights_XXXXXX");
  CreateTestFile(file, path);
  ASSERT_NE(file, nullptr);
  ScopedFileDeleter delete_file{path};
  size_t offset = 0;
  for (auto& tensor : *model.mutable_graph()->mutable_initializer()) {
    std::vector<MLFloat16> weights(static_cast<size_t>(kExperts * kWidth * kWidth), MLFloat16(0.f));
    for (int64_t expert = 0; expert < kExperts; ++expert) {
      for (int64_t column = 0; column < kWidth; ++column) {
        weights[static_cast<size_t>(expert * kWidth * kWidth + column * kWidth + column)] =
            MLFloat16(tensor.name() == "w1" ? static_cast<float>(expert + 1) : 1.f);
      }
    }
    const size_t bytes = weights.size() * sizeof(MLFloat16);
    ASSERT_EQ(fwrite(weights.data(), 1, bytes, file), bytes);
    tensor.clear_raw_data();
    tensor.set_data_location(TensorProto_DataLocation_EXTERNAL);
    for (const auto& [key, value] : {
             std::pair{std::string{"location"}, ToUTF8String(path)},
             std::pair{std::string{"offset"}, std::to_string(offset)},
             std::pair{std::string{"length"}, std::to_string(bytes)}}) {
      auto* entry = tensor.add_external_data();
      entry->set_key(key);
      entry->set_value(value);
    }
    offset += bytes;
  }
  ASSERT_EQ(fclose(file), 0);
  const auto serialized = model.SerializeAsString();
  for (size_t readers : {size_t{0}, size_t{4}}) {
    SCOPED_TRACE(readers);
    OrtCUDAProviderOptionsV2 cuda_options{};
    cuda_options.external_data_loader_reading_threads = readers;
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadExperts, "3"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "0"));
    InferenceSessionWrapper session(options, GetEnvironment());
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(CudaExecutionProviderWithOptions(&cuda_options)));
    ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
    ASSERT_STATUS_OK(session.Initialize());
    std::vector<OrtValue> outputs;
    ASSERT_STATUS_OK(ExecuteCountingModel(session, outputs));
    ASSERT_EQ(outputs.size(), 1U);
    const auto result = outputs[0].Get<Tensor>().DataAsSpan<MLFloat16>();
    for (int64_t row = 0; row < 3; ++row) {
      for (int64_t column = 0; column < kWidth; ++column) {
        EXPECT_NEAR(result[static_cast<size_t>(row * kWidth + column)].ToFloat(), row == 2 ? 9.f : 1.f, 0.01f);
      }
    }
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

  const char* path = "moe_static_cpu_offload_all_cuda_node_state.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0.75,"epsilon":0.01,"experts":[)"
            R"({"moe":0,"expert":0,"popularity":10},)"
            R"({"moe":0,"expert":1,"popularity":9},)"
            R"({"moe":0,"expert":2,"popularity":8},)"
            R"({"moe":0,"expert":3,"popularity":7}]})";
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

  const char* path = "moe_static_cpu_offload_initial_state.json";
  auto cleanup = gsl::finally([path]() { std::remove(path); });
  {
    std::ofstream file(path);
    file << R"({"alpha":0.75,"epsilon":0.01,"experts":[)"
            R"({"moe":0,"expert":3,"popularity":10},)"
            R"({"moe":1,"expert":2,"popularity":9},)"
            R"({"moe":1,"expert":0,"popularity":8},)"
            R"({"moe":0,"expert":1,"popularity":7},)"
            R"({"moe":1,"expert":3,"popularity":6}]})";
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
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "1"));
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
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "0.5"},
           std::pair{kOrtSessionOptionsConfigMoeExpertSwapEpsilon, "0.1"},
           std::pair{kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "2"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadExperts, "-1"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadStrategy, "invalid"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadGpuRatio, "0.8"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadLayerOrder, "last"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, "1"},
           std::pair{kOrtSessionOptionsConfigMoeCpuOffloadExperts, "invalid"}}) {
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first, entry.second));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }

  for (const char* path : {""}) {
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
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "nan"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "inf"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterAlpha, "invalid"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "-0.1"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "1.1"},
           std::pair{kOrtSessionOptionsConfigMoeExpertCounterProbabilityWeight, "invalid"},
           std::pair{kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "-1"},
           std::pair{kOrtSessionOptionsConfigMoeMaxInFlightSwapsPerDevice, "invalid"}}) {
    auto options = CountingOptions();
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(entry.first, entry.second));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_FALSE(session.Initialize().IsOK());
  }
}

TEST(MoeExpertCountingTest, DemandSwapThresholdConfigurationIsValidated) {
  for (const char* value : {"-1", "invalid", "", "1.5", "9223372036854775808", "0", "1"}) {
    for (const char* strategy : {"adaptive", "basic", "layered"}) {
      SCOPED_TRACE(MakeString("threshold=", value, ", strategy=", strategy));
      auto options = CountingOptions();
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, strategy));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadOnDemandThreshold, value));
      InferenceSessionWrapper session(options, GetEnvironment());
      const auto model = MakeCountingModel();
      ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
      const bool valid = std::string_view(value) == "0";
      EXPECT_EQ(session.Initialize().IsOK(), valid);
    }
  }
}
TEST(MoeExpertCountingTest, LayeredGpuRatioConfigurationIsValidatedWithoutTracking) {
  for (const char* ratio : {"0", "-0.1", "1.1", "nan", "inf", "invalid", "", "0.8", "1"}) {
    SCOPED_TRACE(ratio);
    SessionOptions options;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, "layered"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadGpuRatio, ratio));
    InferenceSessionWrapper session(options, GetEnvironment());
    const auto model = MakeCountingModel();
    ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
    EXPECT_EQ(session.Initialize().IsOK(), std::string_view(ratio) == "0.8" || std::string_view(ratio) == "1");
  }
}
TEST(MoeExpertCountingTest, LayerOrderConfigurationIsValidatedWithoutTracking) {
  for (const char* strategy : {"adaptive", "basic", "layered"}) {
    for (const char* order : {"first", "last", "middle", "", "invalid"}) {
      SCOPED_TRACE(MakeString("strategy=", strategy, ", order=", order));
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadStrategy, strategy));
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsConfigMoeCpuOffloadLayerOrder, order));
      InferenceSessionWrapper session(options, GetEnvironment());
      const auto model = MakeCountingModel();
      ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
      const bool valid = std::string_view(strategy) != "adaptive" &&
                         (std::string_view(order) == "first" || std::string_view(order) == "last" ||
                          std::string_view(order) == "middle");
      EXPECT_EQ(session.Initialize().IsOK(), valid);
    }
  }
}
#endif
}  // namespace onnxruntime::test
