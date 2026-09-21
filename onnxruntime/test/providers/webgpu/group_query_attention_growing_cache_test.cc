// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
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

struct GatheredKvCacheModel {
  std::string data;
  // query, key, value, past_keys, past_values, seqlens_k, total_seq_len.
  std::array<std::string, 7> input_names;
};

// A KV cache held as one tensor for all layers and sliced per layer. past_key is then a value the
// memory planner owns, which is the only situation in which it honours a MayInplace alias, and the
// symbolic past sequence dimension keeps shape inference from seeing that present has to grow.
GatheredKvCacheModel BuildGatheredKvCacheModel() {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("GqaGrowingKvCache", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              {{kOnnxDomain, 21}, {kMSDomain, 1}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* query = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kNumHeads * kHeadSize});
  auto* key = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* value = builder.MakeSymbolicInput<float>({kBatch, kSeqLen, kKvNumHeads * kHeadSize});
  auto* past_keys = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, "past_len", kHeadSize});
  auto* past_values = builder.MakeSymbolicInput<float>({kLayers, kBatch, kKvNumHeads, "past_len", kHeadSize});
  auto* seqlens_k = builder.MakeSymbolicInput<int32_t>({kBatch});
  auto* total_seq_len = builder.MakeSymbolicInput<int32_t>({});

  auto* layer_index = builder.MakeScalarInitializer<int64_t>(0);
  auto* layer_axis = builder.Make1DInitializer<int64_t>({0});

  auto* past_key = builder.MakeIntermediate();
  auto* past_value = builder.MakeIntermediate();
  builder.AddNode("Gather", {past_keys, layer_index}, {past_key}).AddAttribute("axis", static_cast<int64_t>(0));
  builder.AddNode("Gather", {past_values, layer_index}, {past_value}).AddAttribute("axis", static_cast<int64_t>(0));

  auto* attn_out = builder.MakeOutput();
  auto* present_key = builder.MakeIntermediate();
  auto* present_value = builder.MakeIntermediate();
  Node& gqa = builder.AddNode("GroupQueryAttention",
                              {query, key, value, past_key, past_value, seqlens_k, total_seq_len},
                              {attn_out, present_key, present_value}, kMSDomain);
  gqa.AddAttribute("num_heads", kNumHeads);
  gqa.AddAttribute("kv_num_heads", kKvNumHeads);

  // Consuming the present outputs keeps them ordinary graph values, as in a real cache write-back.
  auto* new_keys = builder.MakeOutput();
  auto* new_values = builder.MakeOutput();
  builder.AddNode("Unsqueeze", {present_key, layer_axis}, {new_keys});
  builder.AddNode("Unsqueeze", {present_value, layer_axis}, {new_values});

  builder.SetGraphOutputs();
  EXPECT_STATUS_OK(graph.Resolve());

  GatheredKvCacheModel result;
  model.ToProto().SerializeToString(&result.data);
  result.input_names = {query->Name(), key->Name(), value->Name(), past_keys->Name(),
                        past_values->Name(), seqlens_k->Name(), total_seq_len->Name()};
  return result;
}

// Exactly representable and small: WebGPU and the CPU reference accumulate in a different order.
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

// `past_len` is the KV buffer that reaches the node; `total_seq_len` is how long present must be.
NameMLValMap MakeFeeds(const GatheredKvCacheModel& model, int64_t past_len,
                       int32_t seqlens_k, int32_t total_seq_len) {
  const size_t q_size = static_cast<size_t>(kBatch * kSeqLen * kNumHeads * kHeadSize);
  const size_t kv_size = static_cast<size_t>(kBatch * kSeqLen * kKvNumHeads * kHeadSize);
  const size_t past_size = static_cast<size_t>(kLayers * kBatch * kKvNumHeads * past_len * kHeadSize);
  const std::vector<int64_t> past_shape{kLayers, kBatch, kKvNumHeads, past_len, kHeadSize};

  NameMLValMap feeds;
  AddFeed<float>(feeds, model.input_names[0], {kBatch, kSeqLen, kNumHeads * kHeadSize}, SmallValues(q_size, 0));
  AddFeed<float>(feeds, model.input_names[1], {kBatch, kSeqLen, kKvNumHeads * kHeadSize}, SmallValues(kv_size, 3));
  AddFeed<float>(feeds, model.input_names[2], {kBatch, kSeqLen, kKvNumHeads * kHeadSize}, SmallValues(kv_size, 7));
  AddFeed<float>(feeds, model.input_names[3], past_shape, SmallValues(past_size, 11));
  AddFeed<float>(feeds, model.input_names[4], past_shape, SmallValues(past_size, 13));
  AddFeed<int32_t>(feeds, model.input_names[5], {kBatch}, {seqlens_k});
  AddFeed<int32_t>(feeds, model.input_names[6], {}, {total_seq_len});
  return feeds;
}

}  // namespace

// The kernel def used to declare MayInplace(past_key -> present_key) unconditionally. Once the
// planner honoured the alias, a step needing a longer present aborted in OpKernelContext::Output
// with "Shape mismatch attempting to re-use buffer" -- every step of a grow-by-concat KV cache.
TEST(GroupQueryAttention_WebGPU, PresentKvMayGrowBeyondPlannerOwnedPast) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const GatheredKvCacheModel model = BuildGatheredKvCacheModel();
  const gsl::span<const std::byte> model_bytes{reinterpret_cast<const std::byte*>(model.data.data()),
                                               model.data.size()};

  struct Step {
    int64_t past_len;
    int32_t seqlens_k;
    int32_t total_seq_len;
  };
  // The first three grow present beyond the buffer they were handed; the last two never did.
  constexpr Step kSteps[] = {{0, 0, 1}, {1, 1, 2}, {2, 2, 3}, {1, 0, 1}, {4, 3, 4}};

  for (const Step& step : kSteps) {
    SCOPED_TRACE("past_len=" + std::to_string(step.past_len) +
                 " total_seq_len=" + std::to_string(step.total_seq_len));
    EPVerificationParams params;
    params.ep_node_assignment = ExpectedEPNodeAssignment::Some;
    params.fp32_abs_err = 1e-4f;
    RunAndVerifyOutputsWithEP(model_bytes, "GroupQueryAttention_WebGPU.PresentKvMayGrowBeyondPlannerOwnedPast",
                              DefaultWebGpuExecutionProvider(),
                              MakeFeeds(model, step.past_len, step.seqlens_k, step.total_seq_len),
                              params);
  }
}

}  // namespace test
}  // namespace onnxruntime
