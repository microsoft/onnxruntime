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

// Dimensions shared by all test cases.
constexpr int64_t kBatchSize      = 1;
constexpr int64_t kSeqLen         = 1;
constexpr int64_t kHiddenSize     = 4;
constexpr int64_t kHeadSize       = 4;
constexpr int64_t kQLorRank       = 2;
constexpr int64_t kOGroups        = 1;
constexpr int64_t kOLoraRank      = 2;
constexpr int64_t kKVWidth        = 8;

void AddBaseDeepSeekV4Inputs(OpTester& tester) {
  constexpr int64_t batch_size = kBatchSize;
  constexpr int64_t sequence_length = kSeqLen;
  constexpr int64_t hidden_size = kHiddenSize;
  constexpr int64_t head_size = kHeadSize;
  constexpr int64_t q_lora_rank = kQLorRank;
  constexpr int64_t o_groups = kOGroups;
  constexpr int64_t o_lora_rank = kOLoraRank;
  constexpr int64_t kv_width = kKVWidth;

  tester.AddInput<float>("hidden_states", {batch_size, sequence_length, hidden_size}, std::vector<float>(4, 0.1f));
  tester.AddInput<int64_t>("position_ids", {batch_size, sequence_length}, {0});
  tester.AddOptionalInputEdge<float>();  // attention_bias
  tester.AddInput<float>("past_key", {batch_size, 1, 1, head_size}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_value", {batch_size, 1, 1, head_size}, std::vector<float>(4, 0.0f));
  tester.AddInput<int32_t>("seqlens_k", {batch_size}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddInput<float>("cos_cache", {2, head_size / 2}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("sin_cache", {2, head_size / 2}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("q_a_weight", {hidden_size, q_lora_rank}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("q_a_norm_weight", {q_lora_rank}, std::vector<float>(2, 1.0f));
  tester.AddInput<float>("q_b_weight", {q_lora_rank, hidden_size}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("kv_weight", {hidden_size, kv_width}, std::vector<float>(32, 0.0f));
  tester.AddInput<float>("kv_norm_weight", {kv_width}, std::vector<float>(8, 1.0f));
  tester.AddInput<float>("o_a_weight", {hidden_size, o_groups * o_lora_rank}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("o_b_weight", {o_groups * o_lora_rank, hidden_size}, std::vector<float>(8, 0.0f));
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

// Helper to add inputs as MLFloat16 so the CUDA EP (which registers MLFloat16) accepts them.
void AddBaseDeepSeekV4InputsHalf(OpTester& tester) {
  auto make_half = [](int n, float val) {
    std::vector<MLFloat16> v(n);
    std::fill(v.begin(), v.end(), MLFloat16(val));
    return v;
  };
  constexpr int64_t B  = kBatchSize;
  constexpr int64_t S  = kSeqLen;
  constexpr int64_t H  = kHiddenSize;
  constexpr int64_t HS = kHeadSize;
  constexpr int64_t Rq = kQLorRank;
  constexpr int64_t Go = kOGroups;
  constexpr int64_t Ro = kOLoraRank;
  constexpr int64_t Kw = kKVWidth;

  tester.AddInput<MLFloat16>("hidden_states", {B, S, H}, make_half(static_cast<int>(B * S * H), 0.1f));
  tester.AddInput<int64_t>("position_ids", {B, S}, {0});
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddInput<MLFloat16>("past_key",   {B, 1, 1, HS}, make_half(static_cast<int>(B * HS), 0.0f));
  tester.AddInput<MLFloat16>("past_value", {B, 1, 1, HS}, make_half(static_cast<int>(B * HS), 0.0f));
  tester.AddInput<int32_t>("seqlens_k", {B}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddInput<MLFloat16>("cos_cache", {2, HS / 2}, make_half(static_cast<int>(HS), 1.0f));
  tester.AddInput<MLFloat16>("sin_cache", {2, HS / 2}, make_half(static_cast<int>(HS), 0.0f));
  tester.AddInput<MLFloat16>("q_a_weight",      {H, Rq},       make_half(static_cast<int>(H * Rq), 0.0f));
  tester.AddInput<MLFloat16>("q_a_norm_weight", {Rq},          make_half(static_cast<int>(Rq), 1.0f));
  tester.AddInput<MLFloat16>("q_b_weight",      {Rq, H},       make_half(static_cast<int>(Rq * H), 0.0f));
  tester.AddInput<MLFloat16>("kv_weight",       {H, Kw},       make_half(static_cast<int>(H * Kw), 0.0f));
  tester.AddInput<MLFloat16>("kv_norm_weight",  {Kw},          make_half(static_cast<int>(Kw), 1.0f));
  tester.AddInput<MLFloat16>("o_a_weight",      {H, Go * Ro},  make_half(static_cast<int>(H * Go * Ro), 0.0f));
  tester.AddInput<MLFloat16>("o_b_weight",      {Go * Ro, H},  make_half(static_cast<int>(Go * Ro * H), 0.0f));
  tester.AddInput<MLFloat16>("head_sink", {1}, make_half(1, 0.0f));
}

void AddBaseDeepSeekV4AttributesHalf(OpTester& tester) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("head_size", kHeadSize);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<int64_t>("q_lora_rank", kQLorRank);
  tester.AddAttribute<int64_t>("o_groups", kOGroups);
  tester.AddAttribute<int64_t>("o_lora_rank", kOLoraRank);
  tester.AddAttribute<int64_t>("rotary_dim", kHeadSize);
  tester.AddAttribute<int64_t>("rotary_interleaved", 1);
  tester.AddAttribute<int64_t>("rotary_trailing", 1);
  tester.AddAttribute<int64_t>("do_output_derotate", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddAttribute<float>("rms_norm_epsilon", 1e-6f);
  tester.AddAttribute<std::string>("attention_mode", "sliding");
}

}  // namespace

TEST(DeepSeekV4AttentionTest, SlidingModeRunsMathPath) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  AddBaseDeepSeekV4Inputs(tester);

  tester.AddOutput<float>("output", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_key", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_value", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
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

#ifdef USE_CUDA
TEST(DeepSeekV4AttentionTest, CudaSlidingModeRunsMathPath) {
  // Run the sliding-mode math path on the CUDA EP using MLFloat16 tensors
  // (the CUDA kernel registers only MLFloat16 and BFloat16).
  // All weights are zero so all outputs should also be zero.
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4AttributesHalf(tester);
  AddBaseDeepSeekV4InputsHalf(tester);

  auto make_half_zero = [](int n) {
    std::vector<MLFloat16> v(n);
    std::fill(v.begin(), v.end(), MLFloat16(0.0f));
    return v;
  };

  tester.AddOutput<MLFloat16>("output", {kBatchSize, kSeqLen, kHiddenSize},
                              make_half_zero(static_cast<int>(kBatchSize * kSeqLen * kHiddenSize)));
  tester.AddOutput<MLFloat16>("present_key",   {kBatchSize, 1, 1, kHeadSize},
                              make_half_zero(static_cast<int>(kBatchSize * kHeadSize)));
  tester.AddOutput<MLFloat16>("present_value", {kBatchSize, 1, 1, kHeadSize},
                              make_half_zero(static_cast<int>(kBatchSize * kHeadSize)));

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_eps;
  cuda_eps.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_eps);
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
