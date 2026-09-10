// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

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

  std::string attention_mode = "selected_only";
  std::string selected_kv_source = "main";
  int64_t local_window_size = -1;
  int64_t auxiliary_kv_shared = 0;

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
  std::vector<float> head_sink;

  std::vector<float> expected_output;
  std::vector<float> expected_present_key;
  std::vector<float> expected_present_value;
};

void RunDynamicSparseAttentionCase(
    const DynamicSparseAttentionCase& c,
    std::unique_ptr<IExecutionProvider> cuda_ep,
    OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess) {
  ASSERT_NE(cuda_ep, nullptr);

  const int64_t hidden_size = c.num_heads * c.head_size;
  const int64_t kv_hidden_size = c.kv_num_heads * c.head_size;
  const int64_t query_count = c.batch_size * c.sequence_length;

  ASSERT_EQ(c.query.size(), static_cast<size_t>(query_count * hidden_size));
  ASSERT_EQ(c.key.size(), static_cast<size_t>(query_count * kv_hidden_size));
  ASSERT_EQ(c.value.size(), static_cast<size_t>(query_count * kv_hidden_size));
  const size_t cache_elements =
      static_cast<size_t>(c.batch_size * c.kv_num_heads * c.cache_sequence_length * c.head_size);
  ASSERT_EQ(c.past_key.size(), cache_elements);
  ASSERT_EQ(c.past_value.size(), cache_elements);

  OpTester tester("DynamicSparseAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", c.num_heads);
  tester.AddAttribute<int64_t>("kv_num_heads", c.kv_num_heads);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<int64_t>("is_causal", 1);
  tester.AddAttribute<int64_t>("local_window_size", c.local_window_size);
  tester.AddAttribute<std::string>("attention_mode", c.attention_mode);
  tester.AddAttribute<std::string>("selected_kv_source", c.selected_kv_source);
  tester.AddAttribute<int64_t>("do_rotary", 0);
  tester.AddAttribute<int64_t>("rotary_interleaved", 0);
  tester.AddAttribute<int64_t>("rotary_offset", 0);
  tester.AddAttribute<float>("qk_norm_epsilon", 1e-6f);
  tester.AddAttribute<int64_t>("smooth_softmax", 0);
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", c.auxiliary_kv_shared);

  tester.AddInput<MLFloat16>("query", {c.batch_size, c.sequence_length, hidden_size},
                             FloatsToMLFloat16s(c.query));
  tester.AddInput<MLFloat16>("key", {c.batch_size, c.sequence_length, kv_hidden_size},
                             FloatsToMLFloat16s(c.key));
  tester.AddInput<MLFloat16>("value", {c.batch_size, c.sequence_length, kv_hidden_size},
                             FloatsToMLFloat16s(c.value));
  tester.AddInput<MLFloat16>("past_key",
                             {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                             FloatsToMLFloat16s(c.past_key));
  tester.AddInput<MLFloat16>("past_value",
                             {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                             FloatsToMLFloat16s(c.past_value));

  if (c.auxiliary_sequence_length > 0) {
    tester.AddInput<MLFloat16>("auxiliary_key",
                               {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                               FloatsToMLFloat16s(c.auxiliary_key));
  } else {
    tester.AddOptionalInputEdge<MLFloat16>();
  }

  if (!c.auxiliary_value.empty()) {
    tester.AddInput<MLFloat16>("auxiliary_value",
                               {c.batch_size, c.kv_num_heads, c.auxiliary_sequence_length, c.head_size},
                               FloatsToMLFloat16s(c.auxiliary_value));
  } else {
    tester.AddOptionalInputEdge<MLFloat16>();
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

  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<int64_t>();    // position_ids
  tester.AddOptionalInputEdge<MLFloat16>();  // q_norm_weight
  tester.AddOptionalInputEdge<MLFloat16>();  // k_norm_weight
  if (c.head_sink.empty()) {
    tester.AddOptionalInputEdge<MLFloat16>();
  } else {
    tester.AddInput<MLFloat16>("head_sink", {c.num_heads}, FloatsToMLFloat16s(c.head_sink));
  }

  tester.AddOutput<MLFloat16>("output", {c.batch_size, c.sequence_length, hidden_size},
                              FloatsToMLFloat16s(c.expected_output));
  tester.AddOutput<MLFloat16>("present_key",
                              {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                              FloatsToMLFloat16s(c.expected_present_key));
  tester.AddOutput<MLFloat16>("present_value",
                              {c.batch_size, c.kv_num_heads, c.cache_sequence_length, c.head_size},
                              FloatsToMLFloat16s(c.expected_present_value));
  tester.SetOutputTolerance(0.005f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(expected_result, "", {}, nullptr, &execution_providers);
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
