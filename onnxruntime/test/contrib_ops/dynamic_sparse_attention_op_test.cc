// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#ifdef USE_CUDA
#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA
namespace {

struct DynamicSparseAttentionCase {
  int64_t batch_size = 1;
  int64_t sequence_length = 1;
  int64_t num_heads = 1;
  int64_t kv_num_heads = 1;
  int64_t head_size = 8;
  int64_t cache_sequence_length = 1;
  int64_t auxiliary_sequence_length = 0;
  int64_t max_selected = 1;
  int64_t rotary_cache_length = 0;
  int64_t rotary_half_dim = 0;

  std::string attention_mode = "selected_only";
  std::string selected_kv_source = "main";
  float scale = 1.0f;
  float qk_norm_epsilon = 1e-6f;
  int64_t local_window_size = -1;
  int64_t auxiliary_kv_shared = 0;
  int64_t do_rotary = 0;
  int64_t rotary_interleaved = 0;
  int64_t rotary_offset = 0;
  int64_t smooth_softmax = 0;
  bool packed_qkv = false;

  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> past_key;
  std::vector<float> past_value;
  std::vector<float> auxiliary_key;
  std::vector<float> auxiliary_value;
  std::vector<int32_t> selected_indices;
  std::vector<int64_t> selected_indices_shape;
  std::vector<int32_t> selected_counts;
  std::vector<int64_t> selected_counts_shape;
  std::vector<int32_t> seqlens_k;
  int32_t total_sequence_length = 1;
  std::vector<float> cos_cache;
  std::vector<float> sin_cache;
  std::vector<int64_t> position_ids;
  std::vector<float> q_norm_weight;
  std::vector<float> k_norm_weight;
  std::vector<float> head_sink;

  std::vector<float> expected_output;
  std::vector<float> expected_present_key;
  std::vector<float> expected_present_value;
};

template <typename T>
std::vector<T> ToTensorData(const std::vector<float>& values) {
  if constexpr (std::is_same_v<T, float>) {
    return values;
  } else {
    std::vector<T> converted;
    converted.reserve(values.size());
    for (float value : values) {
      converted.emplace_back(value);
    }
    return converted;
  }
}

template <typename T = MLFloat16>
void RunDynamicSparseAttentionCase(
    const DynamicSparseAttentionCase& c,
    std::unique_ptr<IExecutionProvider> cuda_ep,
    OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
    const std::string& expected_error = "") {
  ASSERT_NE(cuda_ep, nullptr);

  const int64_t hidden_size = c.num_heads * c.head_size;
  const int64_t kv_hidden_size = c.kv_num_heads * c.head_size;
  const int64_t query_count = c.batch_size * c.sequence_length;
  const int64_t query_input_size =
      c.packed_qkv ? hidden_size + 2 * kv_hidden_size : hidden_size;

  ASSERT_EQ(c.query.size(), static_cast<size_t>(query_count * query_input_size));
  if (!c.packed_qkv) {
    ASSERT_EQ(c.key.size(), static_cast<size_t>(query_count * kv_hidden_size));
    ASSERT_EQ(c.value.size(), static_cast<size_t>(query_count * kv_hidden_size));
  }
  const size_t cache_elements =
      static_cast<size_t>(c.batch_size * c.kv_num_heads * c.cache_sequence_length * c.head_size);
  ASSERT_EQ(c.past_key.size(), cache_elements);
  ASSERT_EQ(c.past_value.size(), cache_elements);

  OpTester tester("DynamicSparseAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", c.num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  tester.AddAttribute<float>("scale", c.scale);
  tester.AddAttribute<int64_t>("is_causal", 1);
  tester.AddAttribute<int64_t>("local_window_size", c.local_window_size);
  tester.AddAttribute<std::string>("attention_mode", c.attention_mode);
  tester.AddAttribute<std::string>("selected_kv_source", c.selected_kv_source);
  tester.AddAttribute<int64_t>("do_rotary", c.do_rotary);
  tester.AddAttribute<int64_t>("rotary_interleaved", c.rotary_interleaved);
  tester.AddAttribute<int64_t>("rotary_offset", c.rotary_offset);
  tester.AddAttribute<float>("qk_norm_epsilon", c.qk_norm_epsilon);
  tester.AddAttribute<int64_t>("smooth_softmax", c.smooth_softmax);
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", c.auxiliary_kv_shared);

  tester.AddInput<T>("query", {c.batch_size, c.sequence_length, query_input_size},
                     ToTensorData<T>(c.query));
  if (c.packed_qkv) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("key", {c.batch_size, c.sequence_length, kv_hidden_size},
                       ToTensorData<T>(c.key));
    tester.AddInput<T>("value", {c.batch_size, c.sequence_length, kv_hidden_size},
                       ToTensorData<T>(c.value));
  }
  tester.AddInput<T>("past_key",
                     {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                     ToTensorData<T>(c.past_key));
  tester.AddInput<T>("past_value",
                     {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                     ToTensorData<T>(c.past_value));

  if (c.auxiliary_sequence_length > 0) {
    tester.AddInput<T>("auxiliary_key",
                       {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                       ToTensorData<T>(c.auxiliary_key));
  } else {
    tester.AddOptionalInputEdge<T>();
  }

  if (!c.auxiliary_value.empty()) {
    tester.AddInput<T>("auxiliary_value",
                       {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                       ToTensorData<T>(c.auxiliary_value));
  } else {
    tester.AddOptionalInputEdge<T>();
  }

  const std::vector<int64_t> selected_indices_shape =
      c.selected_indices_shape.empty()
          ? std::vector<int64_t>{query_count, c.max_selected}
          : c.selected_indices_shape;
  const std::vector<int64_t> selected_counts_shape =
      c.selected_counts_shape.empty()
          ? std::vector<int64_t>{query_count}
          : c.selected_counts_shape;
  tester.AddInput<int32_t>("selected_indices", selected_indices_shape, c.selected_indices);
  tester.AddInput<int32_t>("selected_counts", selected_counts_shape, c.selected_counts);
  tester.AddInput<int32_t>("seqlens_k", {c.batch_size}, c.seqlens_k);
  tester.AddInput<int32_t>("total_sequence_length", {}, {c.total_sequence_length});

  if (c.cos_cache.empty()) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("cos_cache", {c.rotary_cache_length, c.rotary_half_dim},
                       ToTensorData<T>(c.cos_cache));
    tester.AddInput<T>("sin_cache", {c.rotary_cache_length, c.rotary_half_dim},
                       ToTensorData<T>(c.sin_cache));
  }
  if (c.position_ids.empty()) {
    tester.AddOptionalInputEdge<int64_t>();
  } else {
    tester.AddInput<int64_t>("position_ids", {c.batch_size, c.sequence_length}, c.position_ids);
  }
  if (c.q_norm_weight.empty()) {
    tester.AddOptionalInputEdge<T>();
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("q_norm_weight", {c.head_size}, ToTensorData<T>(c.q_norm_weight));
    tester.AddInput<T>("k_norm_weight", {c.head_size}, ToTensorData<T>(c.k_norm_weight));
  }
  if (c.head_sink.empty()) {
    tester.AddOptionalInputEdge<T>();
  } else {
    tester.AddInput<T>("head_sink", {c.num_heads}, ToTensorData<T>(c.head_sink));
  }

  tester.AddOutput<T>("output", {c.batch_size, c.sequence_length, hidden_size},
                      ToTensorData<T>(c.expected_output));
  tester.AddOutput<T>("present_key",
                      {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                      ToTensorData<T>(c.expected_present_key));
  tester.AddOutput<T>("present_value",
                      {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                      ToTensorData<T>(c.expected_present_value));
  tester.SetOutputTolerance(0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(expected_result, expected_error, {}, nullptr, &execution_providers);
}

DynamicSparseAttentionCase MakeSingleTokenSelectedOnlyCase() {
  DynamicSparseAttentionCase c;
  c.query.assign(8, 1.0f);
  c.key.assign(8, 3.0f);
  c.value.assign(8, 9.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.selected_indices = {-1, -1};
  c.selected_counts = {0};
  c.max_selected = 2;
  c.seqlens_k = {0};
  c.expected_output.assign(8, 0.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;
  return c;
}

DynamicSparseAttentionCase MakeSingleTokenSelectedValueCase(float value = 9.0f) {
  auto c = MakeSingleTokenSelectedOnlyCase();
  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, value);
  c.selected_indices = {0, -1};
  c.selected_counts = {1};
  c.expected_output.assign(8, value);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;
  return c;
}

}  // namespace

TEST(DynamicSparseAttentionTest, SelectedOnlyMainVariableCountsGqaAndCacheAppend_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.sequence_length = 2;
  c.num_heads = 2;
  c.cache_sequence_length = 3;
  c.max_selected = 3;
  c.total_sequence_length = 3;

  c.query.assign(2 * 2 * 8, 0.0f);
  c.key.assign(2 * 8, 0.0f);
  c.value.insert(c.value.end(), 8, 3.0f);
  c.value.insert(c.value.end(), 8, 5.0f);
  c.past_key.assign(3 * 8, 0.0f);
  c.past_value.assign(3 * 8, 0.0f);
  std::fill_n(c.past_value.begin(), 8, 1.0f);
  c.selected_indices = {0, -1, -1,
                        0, 2, -1};
  c.selected_counts = {1, 2};
  c.seqlens_k = {2};

  c.expected_output.insert(c.expected_output.end(), 2 * 8, 1.0f);
  c.expected_output.insert(c.expected_output.end(), 2 * 8, 3.0f);
  c.expected_present_key.assign(3 * 8, 0.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 1.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 3.0f);
  c.expected_present_value.insert(c.expected_present_value.end(), 8, 5.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, LocalPlusSelectedAuxiliaryJointSoftmaxSinkSharedKv_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  c.auxiliary_kv_shared = 1;
  c.auxiliary_sequence_length = 2;

  c.query.assign(8, 0.0f);
  c.key.assign(8, 0.0f);
  c.value.assign(8, 2.0f);
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 4.0f);
  c.auxiliary_key.insert(c.auxiliary_key.end(), 8, 8.0f);
  c.selected_indices = {1};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.head_sink = {0.0f};

  // Q is zero, so the local, selected auxiliary, and sink logits are all zero.
  // They share one denominator: (2 + 8 + 0) / 3.
  c.expected_output.assign(8, 10.0f / 3.0f);
  c.expected_present_key = c.key;
  c.expected_present_value = c.value;

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyEmptySetProducesZeroAndAppendsCache_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  RunDynamicSparseAttentionCase(MakeSingleTokenSelectedOnlyCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, PackedQkvSelectedOnlyGqa_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.packed_qkv = true;
  c.num_heads = 2;
  c.query.assign(2 * 8, 0.0f);             // Q
  c.query.insert(c.query.end(), 8, 0.0f);  // K
  c.query.insert(c.query.end(), 8, 6.0f);  // V
  c.past_key.assign(8, 0.0f);
  c.past_value.assign(8, 0.0f);
  c.selected_indices = {0};
  c.selected_counts = {1};
  c.seqlens_k = {0};
  c.expected_output.assign(2 * 8, 6.0f);
  c.expected_present_key.assign(8, 0.0f);
  c.expected_present_value.assign(8, 6.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyFloat32_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  RunDynamicSparseAttentionCase<float>(
      MakeSingleTokenSelectedValueCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyBFloat16_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "CUDA device does not support BFloat16.";
  }

  RunDynamicSparseAttentionCase<BFloat16>(
      MakeSingleTokenSelectedValueCase(), std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, QkRmsNormPartialInterleavedRotary_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  DynamicSparseAttentionCase c;
  c.head_size = 16;
  c.cache_sequence_length = 2;
  c.max_selected = 2;
  c.total_sequence_length = 2;
  c.scale = 0.1f;
  c.do_rotary = 1;
  c.rotary_interleaved = 1;
  c.rotary_offset = 8;
  c.rotary_cache_length = 2;
  c.rotary_half_dim = 4;

  c.query.assign(16, 0.0f);
  c.key.assign(16, 0.0f);
  std::fill_n(c.query.begin(), 8, 1.0f);
  std::fill_n(c.key.begin(), 8, 1.0f);
  for (int i = 8; i < 16; i += 2) {
    c.query[i] = 1.0f;
    c.key[i] = 1.0f;
  }
  c.value.assign(16, 3.0f);

  c.past_key.assign(2 * 16, 0.0f);
  for (int i = 8; i < 16; i += 2) {
    c.past_key[i] = 2.0f;
  }
  c.past_value.assign(2 * 16, 0.0f);
  std::fill_n(c.past_value.begin(), 16, 1.0f);

  c.selected_indices = {0, 1};
  c.selected_counts = {2};
  c.seqlens_k = {1};
  c.position_ids = {1};
  c.q_norm_weight.assign(16, 1.0f);
  c.k_norm_weight.assign(16, 1.0f);
  c.cos_cache = {1.0f, 1.0f, 1.0f, 1.0f,
                 0.0f, 0.0f, 0.0f, 0.0f};
  c.sin_cache = {0.0f, 0.0f, 0.0f, 0.0f,
                 1.0f, 1.0f, 1.0f, 1.0f};

  const float inverse_rms = 1.0f / std::sqrt(0.75f + c.qk_norm_epsilon);
  const float current_logit = c.scale * 12.0f * inverse_rms * inverse_rms;
  const float current_weight = std::exp(current_logit);
  const float expected_value = (1.0f + 3.0f * current_weight) / (1.0f + current_weight);
  c.expected_output.assign(16, expected_value);

  c.expected_present_key = c.past_key;
  for (int i = 0; i < 8; ++i) {
    c.expected_present_key[16 + i] = inverse_rms;
  }
  for (int i = 8; i < 16; i += 2) {
    c.expected_present_key[16 + i] = 0.0f;
    c.expected_present_key[16 + i + 1] = inverse_rms;
  }
  c.expected_present_value = c.past_value;
  std::fill(c.expected_present_value.begin() + 16,
            c.expected_present_value.end(), 3.0f);

  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, SelectedOnlyExplicitSinkSharesSoftmax_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedValueCase(4.0f);
  c.head_sink = {0.0f};
  // The selected value and zero-valued sink have equal logits.
  c.expected_output.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(c, std::move(cuda_ep));
}

TEST(DynamicSparseAttentionTest, PrefillAndTokenDecodeFixedCapacityCache_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  {
    SCOPED_TRACE("prefill");
    DynamicSparseAttentionCase c;
    c.sequence_length = 2;
    c.cache_sequence_length = 2;
    c.max_selected = 2;
    c.total_sequence_length = 2;
    c.query.assign(2 * 8, 0.0f);
    c.key.assign(2 * 8, 0.0f);
    c.value.insert(c.value.end(), 8, 2.0f);
    c.value.insert(c.value.end(), 8, 4.0f);
    c.past_key.assign(2 * 8, 0.0f);
    c.past_value.assign(2 * 8, 0.0f);
    c.selected_indices = {0, -1,
                          1, -1};
    c.selected_counts = {1, 1};
    c.seqlens_k = {1};
    c.expected_output = c.value;
    c.expected_present_key = c.key;
    c.expected_present_value = c.value;
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider());
  }

  {
    SCOPED_TRACE("token decode");
    DynamicSparseAttentionCase c;
    c.cache_sequence_length = 3;
    c.max_selected = 2;
    c.total_sequence_length = 3;
    c.query.assign(8, 0.0f);
    c.key.assign(8, 0.0f);
    c.value.assign(8, 6.0f);
    c.past_key.assign(3 * 8, 0.0f);
    c.past_value.assign(3 * 8, 0.0f);
    std::fill_n(c.past_value.begin(), 8, 2.0f);
    std::fill_n(c.past_value.begin() + 8, 8, 4.0f);
    c.selected_indices = {0, 2};
    c.selected_counts = {2};
    c.seqlens_k = {2};
    c.expected_output.assign(8, 4.0f);
    c.expected_present_key.assign(3 * 8, 0.0f);
    c.expected_present_value = c.past_value;
    std::fill(c.expected_present_value.begin() + 2 * 8,
              c.expected_present_value.end(), 6.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider());
  }
}

TEST(DynamicSparseAttentionTest, TokenDecodeWithAliasedCache_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_length = 1;
  constexpr int64_t num_heads = 1;
  constexpr int64_t kv_num_heads = 1;
  constexpr int64_t head_size = 8;
  constexpr int64_t cache_capacity = 2;

  Model model("dynamic_sparse_attention_cache_alias", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 17}, {kMSDomain, 1}},
              {}, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto fp16_type;
  fp16_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  ONNX_NAMESPACE::TypeProto int32_type;
  int32_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);

  std::vector<NodeArg*> inputs{
      &graph.GetOrCreateNodeArg("query", &fp16_type),
      &graph.GetOrCreateNodeArg("key", &fp16_type),
      &graph.GetOrCreateNodeArg("value", &fp16_type),
      &graph.GetOrCreateNodeArg("past_key", &fp16_type),
      &graph.GetOrCreateNodeArg("past_value", &fp16_type),
      &empty,
      &empty,
      &graph.GetOrCreateNodeArg("selected_indices", &int32_type),
      &graph.GetOrCreateNodeArg("selected_counts", &int32_type),
      &graph.GetOrCreateNodeArg("seqlens_k", &int32_type),
      &graph.GetOrCreateNodeArg("total_sequence_length", &int32_type),
      &empty,
      &empty,
      &empty,
      &empty,
      &empty,
      &empty};
  std::vector<NodeArg*> outputs{
      &graph.GetOrCreateNodeArg("output", &fp16_type),
      &graph.GetOrCreateNodeArg("present_key", &fp16_type),
      &graph.GetOrCreateNodeArg("present_value", &fp16_type)};
  auto& node = graph.AddNode("dynamic_sparse_attention", "DynamicSparseAttention", "",
                             inputs, outputs, nullptr, kMSDomain);
  node.AddAttribute("num_heads", num_heads);
  node.AddAttribute("kv_num_heads", kv_num_heads);
  node.AddAttribute("scale", 1.0f);
  node.AddAttribute("attention_mode", std::string{"selected_only"});
  node.AddAttribute("selected_kv_source", std::string{"main"});
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));
  SessionOptions options;
  InferenceSession session(options, GetEnvironment());
  IExecutionProvider* cuda_ep_ptr = cuda_ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(cuda_ep)));
  std::istringstream model_stream(model_data);
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());

  auto gpu_allocators = cuda_ep_ptr->CreatePreferredAllocators();
  auto gpu_allocator = std::find_if(gpu_allocators.begin(), gpu_allocators.end(), [](const auto& allocator) {
    return allocator->Info().device.Type() == OrtDevice::GPU &&
           allocator->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(gpu_allocator, gpu_allocators.end());
  auto allocator = session.GetAllocator((*gpu_allocator)->Info());
  ASSERT_NE(allocator, nullptr);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu_value = [&](const auto& values, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(values)>::value_type;
    Tensor cpu_tensor(DataTypeImpl::GetType<Element>(), shape,
                      const_cast<Element*>(values.data()), cpu_allocator->Info());
    Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, allocator);
    ORT_THROW_IF_ERROR(cuda_ep_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu_tensor), result);
    return result;
  };

  const TensorShape qkv_shape{batch_size, sequence_length, head_size};
  const TensorShape cache_shape{batch_size, kv_num_heads, cache_capacity, head_size};
  auto query_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{0.0f}), qkv_shape);
  auto key_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{0.0f}), qkv_shape);
  auto value_value = make_gpu_value(std::vector<MLFloat16>(head_size, MLFloat16{5.0f}), qkv_shape);
  auto past_key_value =
      make_gpu_value(std::vector<MLFloat16>(cache_capacity * head_size, MLFloat16{0.0f}), cache_shape);
  std::vector<MLFloat16> past_value_data(cache_capacity * head_size, MLFloat16{0.0f});
  std::fill_n(past_value_data.begin(), head_size, MLFloat16{1.0f});
  auto past_value_value = make_gpu_value(past_value_data, cache_shape);
  auto selected_indices_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1, 1});
  auto selected_counts_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1});
  auto seqlens_value = make_gpu_value(std::vector<int32_t>{1}, TensorShape{1});

  std::vector<int32_t> total_length_data{2};
  OrtValue total_length_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), TensorShape{},
                       total_length_data.data(), cpu_allocator->Info(), total_length_value);
  Tensor output_tensor(DataTypeImpl::GetType<MLFloat16>(), qkv_shape, allocator);
  OrtValue output_value;
  Tensor::InitOrtValue(std::move(output_tensor), output_value);

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("query", query_value));
  ASSERT_STATUS_OK(binding->BindInput("key", key_value));
  ASSERT_STATUS_OK(binding->BindInput("value", value_value));
  ASSERT_STATUS_OK(binding->BindInput("past_key", past_key_value));
  ASSERT_STATUS_OK(binding->BindInput("past_value", past_value_value));
  ASSERT_STATUS_OK(binding->BindInput("selected_indices", selected_indices_value));
  ASSERT_STATUS_OK(binding->BindInput("selected_counts", selected_counts_value));
  ASSERT_STATUS_OK(binding->BindInput("seqlens_k", seqlens_value));
  ASSERT_STATUS_OK(binding->BindInput("total_sequence_length", total_length_value));
  ASSERT_STATUS_OK(binding->BindOutput("output", output_value));
  ASSERT_STATUS_OK(binding->BindOutput("present_key", past_key_value));
  ASSERT_STATUS_OK(binding->BindOutput("present_value", past_value_value));
  ASSERT_STATUS_OK(binding->SynchronizeInputs());
  ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
  ASSERT_STATUS_OK(binding->SynchronizeOutputs());

  const auto& results = binding->GetOutputs();
  ASSERT_EQ(results.size(), 3u);
  EXPECT_EQ(results[1].Get<Tensor>().Data<MLFloat16>(),
            past_key_value.Get<Tensor>().Data<MLFloat16>());
  EXPECT_EQ(results[2].Get<Tensor>().Data<MLFloat16>(),
            past_value_value.Get<Tensor>().Data<MLFloat16>());

  Tensor cpu_output(DataTypeImpl::GetType<MLFloat16>(), qkv_shape, cpu_allocator);
  ASSERT_STATUS_OK(cuda_ep_ptr->GetDataTransfer()->CopyTensor(results[0].Get<Tensor>(), cpu_output));
  for (MLFloat16 value : cpu_output.DataAsSpan<MLFloat16>()) {
    EXPECT_FLOAT_EQ(value.ToFloat(), 5.0f);
  }
  Tensor cpu_present_value(DataTypeImpl::GetType<MLFloat16>(), cache_shape, cpu_allocator);
  ASSERT_STATUS_OK(cuda_ep_ptr->GetDataTransfer()->CopyTensor(results[2].Get<Tensor>(), cpu_present_value));
  for (int64_t i = 0; i < head_size; ++i) {
    EXPECT_FLOAT_EQ(cpu_present_value.Data<MLFloat16>()[i].ToFloat(), 1.0f);
    EXPECT_FLOAT_EQ(cpu_present_value.Data<MLFloat16>()[head_size + i].ToFloat(), 5.0f);
  }
}

TEST(DynamicSparseAttentionTest, RejectsAuxiliaryInputsInMainMode_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.auxiliary_sequence_length = 1;
  c.auxiliary_key.assign(8, 1.0f);
  c.auxiliary_value.assign(8, 2.0f);
  RunDynamicSparseAttentionCase(
      c, std::move(cuda_ep), OpTester::ExpectResult::kExpectFailure,
      "auxiliary KV inputs are not allowed when selected_kv_source is main");
}

TEST(DynamicSparseAttentionTest, RejectsMissingAuxiliaryInputInAuxiliaryMode_CUDA) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto c = MakeSingleTokenSelectedOnlyCase();
  c.attention_mode = "local_plus_selected";
  c.selected_kv_source = "auxiliary";
  c.local_window_size = 1;
  RunDynamicSparseAttentionCase(
      c, std::move(cuda_ep), OpTester::ExpectResult::kExpectFailure,
      "auxiliary KV inputs are required when selected_kv_source is auxiliary");
}

TEST(DynamicSparseAttentionTest, RejectsInvalidSelectionMetadata_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  {
    SCOPED_TRACE("selected_counts shape");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_counts = {0, 0};
    c.selected_counts_shape = {2};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected count exceeds row capacity");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_counts = {3};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("non-padding entry after selected count");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {0, -1};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("duplicate selected entry");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {0, 0};
    c.selected_counts = {2};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected entry outside main sequence");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_indices = {1, -1};
    c.selected_counts = {1};
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected entry is non-causal");
    DynamicSparseAttentionCase c;
    c.sequence_length = 2;
    c.cache_sequence_length = 2;
    c.max_selected = 1;
    c.total_sequence_length = 2;
    c.query.assign(2 * 8, 0.0f);
    c.key.assign(2 * 8, 0.0f);
    c.value.assign(2 * 8, 1.0f);
    c.past_key.assign(2 * 8, 0.0f);
    c.past_value.assign(2 * 8, 0.0f);
    c.selected_indices = {1, 1};
    c.selected_counts = {1, 1};
    c.seqlens_k = {1};
    c.expected_output.assign(2 * 8, 0.0f);
    c.expected_present_key.assign(2 * 8, 0.0f);
    c.expected_present_value.assign(2 * 8, 1.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure,
                                  "must not refer to a future key");
  }
  {
    SCOPED_TRACE("per-batch length exceeds declared total length");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.cache_sequence_length = 3;
    c.total_sequence_length = 2;
    c.past_key.assign(3 * 8, 0.0f);
    c.past_value.assign(3 * 8, 0.0f);
    c.seqlens_k = {2};
    c.expected_present_key.assign(3 * 8, 0.0f);
    c.expected_present_value.assign(3 * 8, 0.0f);
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure,
                                  "incompatible with the current sequence");
  }
}

TEST(DynamicSparseAttentionTest, RejectsUnknownModeAndSource_CUDA) {
  auto cuda_ep_probe = DefaultCudaExecutionProvider();
  if (!cuda_ep_probe) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  cuda_ep_probe.reset();

  {
    SCOPED_TRACE("attention_mode");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.attention_mode = "dense";
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
  {
    SCOPED_TRACE("selected_kv_source");
    auto c = MakeSingleTokenSelectedOnlyCase();
    c.selected_kv_source = "paged";
    RunDynamicSparseAttentionCase(c, DefaultCudaExecutionProvider(), OpTester::ExpectResult::kExpectFailure);
  }
}

#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
