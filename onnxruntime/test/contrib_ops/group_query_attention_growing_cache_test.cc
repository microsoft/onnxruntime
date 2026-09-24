// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GroupQueryAttention with a growing KV cache whose past_key/past_value are planner-owned values.
// Shape inference used to copy the past length onto present, symbolic or fixed, although present
// grows with the runtime total_sequence_length. The allocation planner then gave present a
// past-sized buffer (MayInplace alias or free list) and every growing step failed with
// "Shape mismatch attempting to re-use buffer".

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr int64_t kBatch = 1;
constexpr int64_t kSeqLen = 1;
constexpr int64_t kNumHeads = 4;
constexpr int64_t kKvNumHeads = 2;
constexpr int64_t kHeadSize = 32;
constexpr int64_t kLayers = 2;

using Dim = std::variant<int64_t, std::string>;

// How the planner gets hold of a past-sized buffer to give to present.
enum class PastSizedBuffer {
  // Gather(past_keys, layer) -> GroupQueryAttention. past_key dies at the node, so a
  // MayInplace(past_key -> present_key) alias on the kernel def is honoured.
  kPastItself,
  // Gather -> Mul -> GroupQueryAttention. The Gather output is dead and on the planner's free list
  // by the time the present outputs are planned. No kernel def alias is involved.
  kFreeList,
};

struct ModelOptions {
  PastSizedBuffer kind = PastSizedBuffer::kPastItself;
  Dim past_len = std::string("past_len");
  // total_sequence_length as an initializer instead of a graph input.
  std::optional<int32_t> constant_total_seq_len;
  bool sliding_window_cache = false;
};

struct KvCacheModel {
  std::string data;
  // query, key, value, past_keys, past_values, seqlens_k, total_seq_len (empty when constant).
  std::array<std::string, 7> input_names;
  std::vector<std::string> output_names;
};

// A KV cache held as one tensor for all layers and sliced per layer.
KvCacheModel BuildKvCacheModel(const ModelOptions& options) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("GqaGrowingKvCache", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* query = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kNumHeads * kHeadSize});
  auto* key = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* value = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* past_keys = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, options.past_len, kHeadSize});
  auto* past_values = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, options.past_len, kHeadSize});
  auto* seqlens_k = builder.MakeSymbolicInput<int32_t>({kBatch});
  auto* total_seq_len = options.constant_total_seq_len
                            ? builder.MakeScalarInitializer<int32_t>(*options.constant_total_seq_len)
                            : builder.MakeSymbolicInput<int32_t>({});

  auto* layer_index = builder.MakeScalarInitializer<int64_t>(0);
  auto* layer_axis = builder.Make1DInitializer<int64_t>({0});

  auto slice_layer = [&](NodeArg* cache) {
    auto* gathered = builder.MakeIntermediate();
    builder.AddNode("Gather", {cache, layer_index}, {gathered}).AddAttribute("axis", static_cast<int64_t>(0));
    if (options.kind == PastSizedBuffer::kPastItself) {
      return gathered;
    }
    // Both Mul inputs are the same value, so Mul cannot run in place and `gathered` is freed.
    auto* squared = builder.MakeIntermediate();
    builder.AddNode("Mul", {gathered, gathered}, {squared});
    return squared;
  };
  auto* past_key = slice_layer(past_keys);
  auto* past_value = slice_layer(past_values);

  auto* attn_out = builder.MakeOutput();
  auto* present_key = builder.MakeIntermediate();
  auto* present_value = builder.MakeIntermediate();
  Node& gqa = builder.AddNode("GroupQueryAttention",
                              {query, key, value, past_key, past_value, seqlens_k, total_seq_len},
                              {attn_out, present_key, present_value}, kMSDomain);
  gqa.AddAttribute("num_heads", kNumHeads);
  gqa.AddAttribute("kv_num_heads", kKvNumHeads);
  if (options.sliding_window_cache) {
    gqa.AddAttribute("sliding_window_cache", static_cast<int64_t>(1));
    gqa.AddAttribute("local_window_size", static_cast<int64_t>(8));
  }

  auto* new_keys = builder.MakeOutput();
  auto* new_values = builder.MakeOutput();
  builder.AddNode("Unsqueeze", {present_key, layer_axis}, {new_keys});
  builder.AddNode("Unsqueeze", {present_value, layer_axis}, {new_values});

  builder.SetGraphOutputs();
  EXPECT_STATUS_OK(graph.Resolve());

  KvCacheModel result;
  model.ToProto().SerializeToString(&result.data);
  result.input_names = {query->Name(), key->Name(), value->Name(), past_keys->Name(), past_values->Name(),
                        seqlens_k->Name(), options.constant_total_seq_len ? "" : total_seq_len->Name()};
  result.output_names = {attn_out->Name(), new_keys->Name(), new_values->Name()};
  return result;
}

// Small, exactly representable values keep the cross-EP comparison well conditioned.
std::vector<float> SmallValues(size_t count, int seed) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = static_cast<float>((static_cast<int>(i) * 5 + seed) % 17 - 8) / 16.0f;
  }
  return values;
}

template <typename T>
void AddFeed(NameMLValMap& feeds, const std::string& name,
             const std::vector<int64_t>& shape, const std::vector<T>& data) {
  OrtValue value;
  CreateMLValue<T>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, data, &value);
  feeds.insert({name, std::move(value)});
}

struct Step {
  // Length of the KV buffer that reaches the node.
  int64_t past_len;
  int32_t seqlens_k;
  // How long the present cache has to be once this step's key and value are appended.
  int32_t total_seq_len;
};

// Fed to a model with a symbolic past length. The first three steps grow present beyond the past
// buffer; the last two do not.
constexpr Step kSymbolicSteps[] = {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}, {1, 0, 1}, {4, 3, 4}};

// Each fed to a model whose declared past length equals the step's past length. A fixed declared
// length does not make the cache static: total_sequence_length is still a runtime input.
constexpr Step kFixedSteps[] = {{1, 1, 2}, {2, 2, 3}, {16, 16, 17}};

std::string Describe(const Step& step) {
  return "past_len=" + std::to_string(step.past_len) + " total_seq_len=" + std::to_string(step.total_seq_len);
}

NameMLValMap MakeFeeds(const KvCacheModel& model, const Step& step) {
  const size_t q_size = static_cast<size_t>(kBatch * kSeqLen * kNumHeads * kHeadSize);
  const size_t kv_size = static_cast<size_t>(kBatch * kSeqLen * kKvNumHeads * kHeadSize);
  const size_t past_size = static_cast<size_t>(kLayers * kBatch * kKvNumHeads * step.past_len * kHeadSize);
  const std::vector<int64_t> past_shape{kLayers, kBatch, kKvNumHeads, step.past_len, kHeadSize};

  NameMLValMap feeds;
  AddFeed<float>(feeds, model.input_names[0], {kBatch, kSeqLen, kNumHeads * kHeadSize}, SmallValues(q_size, 0));
  AddFeed<float>(feeds, model.input_names[1], {kBatch, kSeqLen, kKvNumHeads * kHeadSize}, SmallValues(kv_size, 3));
  AddFeed<float>(feeds, model.input_names[2], {kBatch, kSeqLen, kKvNumHeads * kHeadSize}, SmallValues(kv_size, 7));
  AddFeed<float>(feeds, model.input_names[3], past_shape, SmallValues(past_size, 11));
  AddFeed<float>(feeds, model.input_names[4], past_shape, SmallValues(past_size, 13));
  AddFeed<int32_t>(feeds, model.input_names[5], {kBatch}, {step.seqlens_k});
  AddFeed<int32_t>(feeds, model.input_names[6], {}, {step.total_seq_len});
  return feeds;
}

// The present_key sequence dimension inferred for the model's GroupQueryAttention node, read back
// from a fresh load with the shapes recorded at build time dropped.
ONNX_NAMESPACE::TensorShapeProto_Dimension InferredPresentLength(const KvCacheModel& model) {
  ONNX_NAMESPACE::ModelProto proto;
  EXPECT_TRUE(proto.ParseFromString(model.data));
  proto.mutable_graph()->clear_value_info();
  std::shared_ptr<Model> loaded;
  EXPECT_STATUS_OK(Model::Load(std::move(proto), loaded, nullptr, DefaultLoggingManager().DefaultLogger()));
  for (const Node& node : loaded->MainGraph().Nodes()) {
    if (node.OpType() == "GroupQueryAttention") {
      const auto* shape = node.OutputDefs()[1]->Shape();
      EXPECT_NE(shape, nullptr);
      EXPECT_EQ(shape->dim_size(), 4);
      EXPECT_EQ(shape->dim(1).dim_value(), kKvNumHeads);
      EXPECT_EQ(shape->dim(3).dim_value(), kHeadSize);
      return shape->dim(2);
    }
  }
  ADD_FAILURE() << "no GroupQueryAttention node";
  return {};
}

void ExpectDynamic(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim) {
  EXPECT_FALSE(dim.has_dim_value()) << dim.dim_value();
  EXPECT_FALSE(dim.has_dim_param()) << dim.dim_param();
}

void RunOnCpu(const KvCacheModel& model, const Step& step, bool enable_mem_reuse, std::vector<OrtValue>& fetches) {
  SessionOptions so;
  so.session_logid = "GroupQueryAttentionGrowingCache";
  so.enable_mem_reuse = enable_mem_reuse;
  InferenceSession session{so, GetEnvironment()};
  ASSERT_STATUS_OK(session.Load(model.data.data(), static_cast<int>(model.data.size())));
  ASSERT_STATUS_OK(session.Initialize());
  ASSERT_STATUS_OK(session.Run(RunOptions{}, MakeFeeds(model, step), model.output_names, &fetches));
}

// The CPU kernel declares no alias, so this is the free-list route alone; a session with memory
// reuse disabled is the reference.
void ExpectCpuReuseMatchesNoReuse(const KvCacheModel& model, const Step& step) {
  std::vector<OrtValue> expected;
  std::vector<OrtValue> actual;
  ASSERT_NO_FATAL_FAILURE(RunOnCpu(model, step, /*enable_mem_reuse*/ false, expected));
  ASSERT_NO_FATAL_FAILURE(RunOnCpu(model, step, /*enable_mem_reuse*/ true, actual));

  ASSERT_EQ(actual.size(), expected.size());
  const TensorShape new_keys_shape{1, kBatch, kKvNumHeads, std::max<int64_t>(step.past_len, step.total_seq_len),
                                   kHeadSize};
  EXPECT_EQ(actual[1].Get<Tensor>().Shape(), new_keys_shape);
  for (size_t i = 0; i < actual.size(); ++i) {
    VerifyOutput(model.output_names[i], expected[i].Get<Tensor>(), actual[i].Get<Tensor>(), 1e-5f);
  }
}

void ExpectWebGpuMatchesCpu(const KvCacheModel& model, const Step& step, const char* log_id) {
  const gsl::span<const std::byte> model_bytes{reinterpret_cast<const std::byte*>(model.data.data()),
                                               model.data.size()};
  // Gather alone would satisfy ExpectedEPNodeAssignment::Some.
  const std::function<void(const Graph&)> gqa_runs_on_webgpu = [](const Graph& graph) {
    for (const Node& node : graph.Nodes()) {
      if (node.OpType() == "GroupQueryAttention") {
        EXPECT_EQ(node.GetExecutionProviderType(), kWebGpuExecutionProvider);
      }
    }
  };
  EPVerificationParams params;
  params.ep_node_assignment = ExpectedEPNodeAssignment::Some;
  params.fp32_abs_err = 1e-4f;
  params.graph_verifier = &gqa_runs_on_webgpu;
  RunAndVerifyOutputsWithEP(model_bytes, log_id, DefaultWebGpuExecutionProvider(), MakeFeeds(model, step), params);
}

void RunOnWebGpu(PastSizedBuffer kind, const char* log_id) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  const KvCacheModel symbolic = BuildKvCacheModel({kind});
  for (const Step& step : kSymbolicSteps) {
    SCOPED_TRACE("symbolic " + Describe(step));
    ExpectWebGpuMatchesCpu(symbolic, step, log_id);
  }
  for (const Step& step : kFixedSteps) {
    SCOPED_TRACE("fixed " + Describe(step));
    ExpectWebGpuMatchesCpu(BuildKvCacheModel({kind, Dim{step.past_len}}), step, log_id);
  }
}

}  // namespace

TEST(GroupQueryAttentionGrowingCacheTest, SymbolicPastLengthLeavesPresentDynamic) {
  ExpectDynamic(InferredPresentLength(BuildKvCacheModel({})));
}

TEST(GroupQueryAttentionGrowingCacheTest, FixedPastLengthLeavesPresentDynamic) {
  ExpectDynamic(InferredPresentLength(BuildKvCacheModel({PastSizedBuffer::kPastItself, Dim{16}})));
}

TEST(GroupQueryAttentionGrowingCacheTest, ConstantTotalLengthGivesExactPresentLength) {
  ModelOptions grows{PastSizedBuffer::kPastItself, Dim{16}, 17};
  EXPECT_EQ(InferredPresentLength(BuildKvCacheModel(grows)).dim_value(), 17);
  ModelOptions fits{PastSizedBuffer::kPastItself, Dim{16}, 8};
  EXPECT_EQ(InferredPresentLength(BuildKvCacheModel(fits)).dim_value(), 16);
}

// A windowed cache is capacity-sized and evicts internally, so present keeps the past length.
TEST(GroupQueryAttentionGrowingCacheTest, SlidingWindowCacheKeepsPastLength) {
  ModelOptions windowed{PastSizedBuffer::kPastItself, Dim{16}, std::nullopt, /*sliding_window_cache*/ true};
  EXPECT_EQ(InferredPresentLength(BuildKvCacheModel(windowed)).dim_value(), 16);
}

TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondFreedPastSizedBuffer) {
  const KvCacheModel symbolic = BuildKvCacheModel({PastSizedBuffer::kFreeList});
  for (const Step& step : kSymbolicSteps) {
    SCOPED_TRACE("symbolic " + Describe(step));
    ExpectCpuReuseMatchesNoReuse(symbolic, step);
  }
  for (const Step& step : kFixedSteps) {
    SCOPED_TRACE("fixed " + Describe(step));
    ExpectCpuReuseMatchesNoReuse(BuildKvCacheModel({PastSizedBuffer::kFreeList, Dim{step.past_len}}), step);
  }
}

TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondPlannerOwnedPast_WebGPU) {
  RunOnWebGpu(PastSizedBuffer::kPastItself, "GroupQueryAttentionGrowingCacheTest.PlannerOwnedPast_WebGPU");
}

TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondFreedPastSizedBuffer_WebGPU) {
  RunOnWebGpu(PastSizedBuffer::kFreeList, "GroupQueryAttentionGrowingCacheTest.FreedPastSizedBuffer_WebGPU");
}

}  // namespace test
}  // namespace onnxruntime
