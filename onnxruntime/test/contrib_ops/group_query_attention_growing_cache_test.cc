// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GroupQueryAttention with a growing KV cache whose past_key/past_value are planner-owned values.
// Shape inference used to copy a symbolic past length onto present; the allocation planner compares
// dim_params by name, so it gave present a past-sized buffer (MayInplace alias or free list) and
// every growing step failed with "Shape mismatch attempting to re-use buffer".

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
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

struct KvCacheModel {
  std::string data;
  // query, key, value, past_keys, past_values, seqlens_k, total_seq_len.
  std::array<std::string, 7> input_names;
  std::vector<std::string> output_names;
};

// How the planner gets hold of a past-sized buffer to give to present.
enum class PastSizedBuffer {
  // Gather(past_keys, layer) -> GroupQueryAttention. past_key dies at the node, so a
  // MayInplace(past_key -> present_key) alias on the kernel def is honoured.
  kPastItself,
  // Gather -> Mul -> GroupQueryAttention. The Gather output is dead and on the planner's free list
  // by the time the present outputs are planned. No kernel def alias is involved.
  kFreeList,
};

// A KV cache held as one tensor for all layers and sliced per layer.
KvCacheModel BuildKvCacheModel(PastSizedBuffer kind, const Dim& past_len = Dim{"past_len"}) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("GqaGrowingKvCache", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* query = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kNumHeads * kHeadSize});
  auto* key = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* value = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* past_keys = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, past_len, kHeadSize});
  auto* past_values = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, past_len, kHeadSize});
  auto* seqlens_k = builder.MakeSymbolicInput<int32_t>({kBatch});
  auto* total_seq_len = builder.MakeSymbolicInput<int32_t>({});

  auto* layer_index = builder.MakeScalarInitializer<int64_t>(0);
  auto* layer_axis = builder.Make1DInitializer<int64_t>({0});

  auto slice_layer = [&](NodeArg* cache) {
    auto* gathered = builder.MakeIntermediate();
    builder.AddNode("Gather", {cache, layer_index}, {gathered}).AddAttribute("axis", static_cast<int64_t>(0));
    if (kind == PastSizedBuffer::kPastItself) {
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

  auto* new_keys = builder.MakeOutput();
  auto* new_values = builder.MakeOutput();
  builder.AddNode("Unsqueeze", {present_key, layer_axis}, {new_keys});
  builder.AddNode("Unsqueeze", {present_value, layer_axis}, {new_values});

  builder.SetGraphOutputs();
  EXPECT_STATUS_OK(graph.Resolve());

  KvCacheModel result;
  model.ToProto().SerializeToString(&result.data);
  result.input_names = {query->Name(), key->Name(), value->Name(), past_keys->Name(),
                        past_values->Name(), seqlens_k->Name(), total_seq_len->Name()};
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

// The first three steps grow present beyond the past buffer; the last two do not.
constexpr Step kSteps[] = {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}, {1, 0, 1}, {4, 3, 4}};

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

const ONNX_NAMESPACE::TensorShapeProto* InferredPresentKeyShape(const Graph& graph) {
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "GroupQueryAttention") {
      return node.OutputDefs()[1]->Shape();
    }
  }
  return nullptr;
}

void LoadModel(const KvCacheModel& model, std::shared_ptr<Model>& loaded) {
  ONNX_NAMESPACE::ModelProto proto;
  ASSERT_TRUE(proto.ParseFromString(model.data));
  // Drop the shapes recorded when the model was built, so that what is read back is inferred here.
  proto.mutable_graph()->clear_value_info();
  ASSERT_STATUS_OK(Model::Load(std::move(proto), loaded, nullptr, DefaultLoggingManager().DefaultLogger()));
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

void RunOnWebGpu(PastSizedBuffer kind, const char* log_id) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const KvCacheModel model = BuildKvCacheModel(kind);
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

  for (const Step& step : kSteps) {
    SCOPED_TRACE(Describe(step));
    EPVerificationParams params;
    params.ep_node_assignment = ExpectedEPNodeAssignment::Some;
    params.fp32_abs_err = 1e-4f;
    params.graph_verifier = &gqa_runs_on_webgpu;
    RunAndVerifyOutputsWithEP(model_bytes, log_id, DefaultWebGpuExecutionProvider(), MakeFeeds(model, step), params);
  }
}

}  // namespace

TEST(GroupQueryAttentionGrowingCacheTest, SymbolicPastLengthIsNotInferredForPresent) {
  std::shared_ptr<Model> model;
  ASSERT_NO_FATAL_FAILURE(LoadModel(BuildKvCacheModel(PastSizedBuffer::kPastItself), model));

  const auto* present_shape = InferredPresentKeyShape(model->MainGraph());
  ASSERT_NE(present_shape, nullptr);
  ASSERT_EQ(present_shape->dim_size(), 4);
  EXPECT_EQ(present_shape->dim(1).dim_value(), kKvNumHeads);
  EXPECT_FALSE(present_shape->dim(2).has_dim_param()) << present_shape->dim(2).dim_param();
  EXPECT_FALSE(present_shape->dim(2).has_dim_value());
  EXPECT_EQ(present_shape->dim(3).dim_value(), kHeadSize);
}

// A fixed-length past is a static cache shared with present, so its length still carries over.
TEST(GroupQueryAttentionGrowingCacheTest, StaticPastLengthIsInferredForPresent) {
  constexpr int64_t kMaxLen = 16;
  std::shared_ptr<Model> model;
  ASSERT_NO_FATAL_FAILURE(LoadModel(BuildKvCacheModel(PastSizedBuffer::kPastItself, Dim{kMaxLen}), model));

  const auto* present_shape = InferredPresentKeyShape(model->MainGraph());
  ASSERT_NE(present_shape, nullptr);
  ASSERT_EQ(present_shape->dim_size(), 4);
  EXPECT_EQ(present_shape->dim(2).dim_value(), kMaxLen);
}

// The CPU kernel declares no alias, so this is the free-list route alone; a session with memory
// reuse disabled is the reference.
TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondFreedPastSizedBuffer) {
  const KvCacheModel model = BuildKvCacheModel(PastSizedBuffer::kFreeList);

  for (const Step& step : kSteps) {
    SCOPED_TRACE(Describe(step));
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
}

TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondPlannerOwnedPast_WebGPU) {
  RunOnWebGpu(PastSizedBuffer::kPastItself, "GroupQueryAttentionGrowingCacheTest.PlannerOwnedPast_WebGPU");
}

TEST(GroupQueryAttentionGrowingCacheTest, PresentKvMayGrowBeyondFreedPastSizedBuffer_WebGPU) {
  RunOnWebGpu(PastSizedBuffer::kFreeList, "GroupQueryAttentionGrowingCacheTest.FreedPastSizedBuffer_WebGPU");
}

}  // namespace test
}  // namespace onnxruntime
