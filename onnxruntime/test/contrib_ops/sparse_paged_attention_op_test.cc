// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr int kHeadSize = 8;
constexpr int kBlockSize = 16;
constexpr int kCacheElementCount = kBlockSize * kHeadSize;

std::vector<MLFloat16> HalfVector(float value, int count = kHeadSize) {
  return std::vector<MLFloat16>(count, MLFloat16(value));
}

void AddAttributes(OpTester& tester) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
}

void AddSingleTokenPrefix(OpTester& tester,
                          const std::vector<MLFloat16>& query,
                          const std::vector<MLFloat16>& key,
                          const std::vector<MLFloat16>& value,
                          const std::vector<int32_t>& selected_indices,
                          int32_t selected_count,
                          int32_t past_seqlen = 0,
                          int32_t block_id = 0,
                          int32_t slot = 0,
                          const std::vector<MLFloat16>& key_cache = HalfVector(0.0f, kCacheElementCount),
                          const std::vector<MLFloat16>& value_cache = HalfVector(0.0f, kCacheElementCount)) {
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {1, kHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {1, kHeadSize}, value);
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {past_seqlen});
  tester.AddInput<int32_t>("block_table", {1, 1}, {block_id});
  tester.AddInput<int32_t>("slot_mapping", {1}, {slot});
  tester.AddInput<int32_t>("selected_indices", {1, static_cast<int64_t>(selected_indices.size())},
                           selected_indices);
  tester.AddInput<int32_t>("selected_counts", {1}, {selected_count});
}

std::vector<MLFloat16> AddQuantizedInputs(OpTester& tester,
                                          const std::string& k_quant_type,
                                          const std::string& v_quant_type,
                                          const std::vector<float>& k_scales,
                                          const std::vector<float>& v_scales) {
  AddAttributes(tester);
  tester.AddAttribute<std::string>("k_quant_type", k_quant_type);
  tester.AddAttribute<std::string>("v_quant_type", v_quant_type);
  tester.AddAttribute<float>("scale", 1.0f);
  auto query = HalfVector(0.0f, 2 * kHeadSize);
  query[kHeadSize + 1] = MLFloat16(1.0f);
  auto key = HalfVector(0.0f, 2 * kHeadSize);
  key[kHeadSize + 1] = MLFloat16(1.5f);
  auto value = HalfVector(0.0f, 2 * kHeadSize);
  std::vector<MLFloat16> current_value;
  current_value.reserve(kHeadSize);
  for (int i = 0; i < kHeadSize; ++i) {
    const float scale = v_quant_type == "PER_CHANNEL" ? v_scales[i] : v_scales[0];
    current_value.emplace_back(scale * static_cast<float>(i + 1));
  }
  std::copy(current_value.begin(), current_value.end(), value.begin() + kHeadSize);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, value);
  tester.AddInput<int8_t>("key_cache", {1, kBlockSize, 1, kHeadSize},
                          std::vector<int8_t>(kCacheElementCount, 0));
  tester.AddInput<int8_t>("value_cache", {1, kBlockSize, 1, kHeadSize},
                          std::vector<int8_t>(kCacheElementCount, 0));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {0, 1});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {0, -1, 0, 1});
  tester.AddInput<int32_t>("selected_counts", {2}, {1, 2});
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_key
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddOptionalInputEdge<int32_t>();    // auxiliary_lengths
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  tester.AddOptionalInputEdge<MLFloat16>();  // q_norm_weight
  tester.AddOptionalInputEdge<MLFloat16>();  // k_norm_weight
  tester.AddInput<float>("k_scale", {static_cast<int64_t>(k_scales.size())}, k_scales);
  tester.AddInput<float>("v_scale", {static_cast<int64_t>(v_scales.size())}, v_scales);

  const float current_weight = std::exp(1.5f) / (1.0f + std::exp(1.5f));
  auto expected = HalfVector(0.0f);
  for (const MLFloat16 element : current_value) {
    expected.emplace_back(element.ToFloat() * current_weight);
  }
  return expected;
}

void RunCuda(OpTester& tester) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(SparsePagedAttention, Cuda_SelectedMainWritesAndReadsPagedCache) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(2.0f), {0, -1}, 1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(2.0f));
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_LocalAndSharedAuxiliaryUseJointSoftmax) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  std::vector<MLFloat16> query = HalfVector(0.0f);
  query[0] = MLFloat16(1.0f);
  std::vector<MLFloat16> auxiliary_key = HalfVector(0.0f);
  auxiliary_key[0] = MLFloat16(std::log(3.0f));

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, query, HalfVector(0.0f), HalfVector(1.0f), {0}, 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize}, auxiliary_key);
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  auto expected = HalfVector(0.25f);
  expected[0] = MLFloat16((1.0f + 3.0f * std::log(3.0f)) / 4.0f);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, expected, false, 0.002f, 0.002f);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_InvalidSelectedMainIndicesAreIgnored) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  for (const auto& test_case : std::vector<std::pair<int32_t, int32_t>>{{-1, 0}, {kBlockSize, 0}, {0, -1}}) {
    OpTester tester("SparsePagedAttention", 1, kMSDomain);
    AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(7.0f),
                         {test_case.first}, 1, 0, test_case.second);
    tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(0.0f));
    RunCuda(tester);
  }
}

TEST(SparsePagedAttention, Cuda_NonCausalSelectedMainIndexIsIgnored) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  auto values = HalfVector(1.0f);
  const auto future_values = HalfVector(9.0f);
  values.insert(values.end(), future_values.begin(), future_values.end());
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, values);
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {0, 1});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {1, 0, 1, -1});
  tester.AddInput<int32_t>("selected_counts", {2}, {2, 1});
  auto expected = HalfVector(1.0f);
  expected.insert(expected.end(), future_values.begin(), future_values.end());
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize}, expected);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_LocalWindowAndSelectedMainUseSetUnion) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto cached_values = HalfVector(0.0f, kCacheElementCount);
  std::fill_n(cached_values.begin(), kHeadSize, MLFloat16(3.0f));
  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(1.0f),
                       {1, 0}, 2, 1, 0, 1, HalfVector(0.0f, kCacheElementCount), cached_values);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(2.0f));
  RunCuda(tester);
}

class SparsePagedAttentionInt8Test : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(SparsePagedAttentionInt8Test, Cuda_QuantizedMainCache) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  const auto& [k_quant_type, v_quant_type] = GetParam();
  std::vector<float> k_scales(k_quant_type == "PER_CHANNEL" ? kHeadSize : 1, 0.5f);
  std::vector<float> v_scales(v_quant_type == "PER_CHANNEL" ? kHeadSize : 1, 0.5f);
  if (k_quant_type == "PER_CHANNEL") {
    for (int i = 0; i < kHeadSize; ++i) {
      k_scales[i] = 0.25f * static_cast<float>(i + 1);
    }
  }
  if (v_quant_type == "PER_CHANNEL") {
    for (int i = 0; i < kHeadSize; ++i) {
      v_scales[i] = 0.25f * static_cast<float>(i + 1);
    }
  }
  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  const auto expected = AddQuantizedInputs(tester, k_quant_type, v_quant_type, k_scales, v_scales);
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize}, expected, false, 0.002f, 0.002f);
  RunCuda(tester);
}

INSTANTIATE_TEST_SUITE_P(
    QuantizationGranularity, SparsePagedAttentionInt8Test,
    ::testing::Values(std::make_pair("PER_TENSOR", "PER_TENSOR"),
                      std::make_pair("PER_CHANNEL", "PER_CHANNEL"),
                      std::make_pair("PER_CHANNEL", "PER_TENSOR")));

}  // namespace test
}  // namespace onnxruntime
