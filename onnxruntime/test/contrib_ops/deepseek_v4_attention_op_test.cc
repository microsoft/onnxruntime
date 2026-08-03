// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

void AddBaseDeepSeekV4Inputs(OpTester& tester) {
  constexpr int64_t batch_size = 1;
  constexpr int64_t sequence_length = 1;
  constexpr int64_t hidden_size = 4;
  constexpr int64_t head_size = 4;
  constexpr int64_t q_lora_rank = 2;
  constexpr int64_t o_groups = 1;
  constexpr int64_t o_lora_rank = 2;
  constexpr int64_t kv_width = 8;

  tester.AddInput<float>("hidden_states", {batch_size, sequence_length, hidden_size}, std::vector<float>(4, 0.1f));
  tester.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, {0});
  tester.AddOptionalInputEdge<float>();  // attention_bias
  tester.AddInput<float>("past_key", {batch_size, 1, 1, head_size}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_value", {batch_size, 1, 1, head_size}, std::vector<float>(4, 0.0f));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddInput<float>("cos_cache", {2, head_size / 2}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("sin_cache", {2, head_size / 2}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("q_a_weight", {hidden_size, q_lora_rank}, std::vector<float>(8, 0.01f));
  tester.AddInput<float>("q_a_norm_weight", {q_lora_rank}, std::vector<float>(2, 1.0f));
  tester.AddInput<float>("q_b_weight", {q_lora_rank, hidden_size}, std::vector<float>(8, 0.01f));
  tester.AddInput<float>("kv_weight", {hidden_size, kv_width}, std::vector<float>(32, 0.01f));
  tester.AddInput<float>("kv_norm_weight", {kv_width}, std::vector<float>(8, 1.0f));
  tester.AddInput<float>("o_a_weight", {hidden_size, o_groups * o_lora_rank}, std::vector<float>(8, 0.01f));
  tester.AddInput<float>("o_b_weight", {o_groups * o_lora_rank, hidden_size}, std::vector<float>(8, 0.01f));
  tester.AddInput<float>("head_sink", {1}, {0.0f});
}

void AddBaseDeepSeekV4Attributes(OpTester& tester) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("head_size", 4);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<int64_t>("q_lora_rank", 2);
  tester.AddAttribute<int64_t>("o_groups", 1);
  tester.AddAttribute<int64_t>("o_lora_rank", 2);
  tester.AddAttribute<int64_t>("rotary_dim", 4);
  tester.AddAttribute<int64_t>("rotary_interleaved", 1);
  tester.AddAttribute<int64_t>("rotary_trailing", 1);
  tester.AddAttribute<int64_t>("do_output_derotate", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddAttribute<float>("rms_norm_epsilon", 1e-6f);
  tester.AddAttribute<std::string>("attention_mode", "sliding");
}

std::vector<std::unique_ptr<IExecutionProvider>> CpuEpOnly() {
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  return execution_providers;
}

}  // namespace

TEST(DeepSeekV4AttentionTest, SlidingModeContractPlaceholder) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  AddBaseDeepSeekV4Inputs(tester);

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "frontend contract placeholder and is not implemented yet",
             {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4AttentionTest, CsaModeRequiresCompressRate) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  tester.AddAttribute<std::string>("attention_mode", "csa");
  tester.AddAttribute<int64_t>("index_topk", 1);
  tester.AddAttribute<int64_t>("index_num_heads", 1);
  tester.AddAttribute<int64_t>("index_head_dim", 4);
  AddBaseDeepSeekV4Inputs(tester);

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "compress_rate must be provided and > 0 for csa/hca mode",
             {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4AttentionTest, SlidingModeRejectsCompressorInputs) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  AddBaseDeepSeekV4Inputs(tester);

  for (int i = 17; i < 24; ++i) {
    tester.AddOptionalInputEdge<float>();
  }

  tester.AddInput<float>("csa_kv_weight", {4, 4}, std::vector<float>(16, 0.01f));

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "compressor inputs are not allowed",
             {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
