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

#ifdef USE_CUDA
std::vector<MLFloat16> MakeHalfValues(int count, float value) {
  return std::vector<MLFloat16>(static_cast<size_t>(count), MLFloat16(value));
}

// Helper to add inputs as MLFloat16 so the CUDA EP (which registers MLFloat16) accepts them.
void AddBaseDeepSeekV4InputsHalf(OpTester& tester) {
  constexpr int64_t B  = kBatchSize;
  constexpr int64_t S  = kSeqLen;
  constexpr int64_t H  = kHiddenSize;
  constexpr int64_t HS = kHeadSize;
  constexpr int64_t Rq = kQLorRank;
  constexpr int64_t Go = kOGroups;
  constexpr int64_t Ro = kOLoraRank;
  constexpr int64_t Kw = kKVWidth;

  tester.AddInput<MLFloat16>("hidden_states", {B, S, H}, MakeHalfValues(static_cast<int>(B * S * H), 0.1f));
  tester.AddInput<int64_t>("position_ids", {B, S}, {0});
  tester.AddOptionalInputEdge<MLFloat16>();  // attention_bias
  tester.AddInput<MLFloat16>("past_key",   {B, 1, 1, HS}, MakeHalfValues(static_cast<int>(B * HS), 0.0f));
  tester.AddInput<MLFloat16>("past_value", {B, 1, 1, HS}, MakeHalfValues(static_cast<int>(B * HS), 0.0f));
  tester.AddInput<int32_t>("seqlens_k", {B}, {0});
  tester.AddInput<int32_t>("total_sequence_length", {1}, {1});
  tester.AddInput<MLFloat16>("cos_cache", {2, HS / 2}, MakeHalfValues(static_cast<int>(HS), 1.0f));
  tester.AddInput<MLFloat16>("sin_cache", {2, HS / 2}, MakeHalfValues(static_cast<int>(HS), 0.0f));
  tester.AddInput<MLFloat16>("q_a_weight",      {H, Rq},       MakeHalfValues(static_cast<int>(H * Rq), 0.0f));
  tester.AddInput<MLFloat16>("q_a_norm_weight", {Rq},          MakeHalfValues(static_cast<int>(Rq), 1.0f));
  tester.AddInput<MLFloat16>("q_b_weight",      {Rq, H},       MakeHalfValues(static_cast<int>(Rq * H), 0.0f));
  tester.AddInput<MLFloat16>("kv_weight",       {H, Kw},       MakeHalfValues(static_cast<int>(H * Kw), 0.0f));
  tester.AddInput<MLFloat16>("kv_norm_weight",  {Kw},          MakeHalfValues(static_cast<int>(Kw), 1.0f));
  tester.AddInput<MLFloat16>("o_a_weight",      {H, Go * Ro},  MakeHalfValues(static_cast<int>(H * Go * Ro), 0.0f));
  tester.AddInput<MLFloat16>("o_b_weight",      {Go * Ro, H},  MakeHalfValues(static_cast<int>(Go * Ro * H), 0.0f));
  tester.AddInput<MLFloat16>("head_sink", {1}, MakeHalfValues(1, 0.0f));
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
#endif  // USE_CUDA

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

TEST(DeepSeekV4AttentionTest, HcaModeUpdatesCompressorState) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  tester.AddAttribute<std::string>("attention_mode", "hca");
  tester.AddAttribute<float>("compress_rate", 1.0f);
  AddBaseDeepSeekV4Inputs(tester);

  tester.AddInput<float>("hca_kv_weight", {kHiddenSize, kHeadSize}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("hca_gate_weight", {kHiddenSize, kHeadSize}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("hca_position_bias", {1, kHeadSize}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("hca_kv_norm_weight", {kHeadSize}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("past_hca_pending_kv", {kBatchSize, 0, kHeadSize}, {});
  tester.AddInput<float>("past_hca_pending_gate", {kBatchSize, 0, kHeadSize}, {});
  tester.AddInput<float>("past_hca_entries", {kBatchSize, 1, 0, kHeadSize}, {});

  tester.AddOutput<float>("output", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_key", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_value", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_hca_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<float>("present_hca_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<float>("present_hca_entries", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4AttentionTest, CsaModeUpdatesCompressorState) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4Attributes(tester);
  tester.AddAttribute<std::string>("attention_mode", "csa");
  tester.AddAttribute<float>("compress_rate", 1.0f);
  tester.AddAttribute<int64_t>("index_topk", 1);
  tester.AddAttribute<int64_t>("index_num_heads", 1);
  tester.AddAttribute<int64_t>("index_head_dim", kHeadSize);
  AddBaseDeepSeekV4Inputs(tester);
  for (int i = 17; i < 24; ++i) {
    tester.AddOptionalInputEdge<float>();
  }

  tester.AddInput<float>("csa_kv_weight", {kHiddenSize, 2 * kHeadSize}, std::vector<float>(32, 0.0f));
  tester.AddInput<float>("csa_gate_weight", {kHiddenSize, 2 * kHeadSize}, std::vector<float>(32, 0.0f));
  tester.AddInput<float>("csa_position_bias", {1, 2 * kHeadSize}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("csa_kv_norm_weight", {kHeadSize}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("index_kv_weight", {kHiddenSize, 2 * kHeadSize}, std::vector<float>(32, 0.0f));
  tester.AddInput<float>("index_gate_weight", {kHiddenSize, 2 * kHeadSize}, std::vector<float>(32, 0.0f));
  tester.AddInput<float>("index_position_bias", {1, 2 * kHeadSize}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("index_kv_norm_weight", {kHeadSize}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("index_q_b_weight", {kQLorRank, kHeadSize}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("index_weights_proj_weight", {kHiddenSize, 1}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_csa_pending_kv", {kBatchSize, 0, 2 * kHeadSize}, {});
  tester.AddInput<float>("past_csa_pending_gate", {kBatchSize, 0, 2 * kHeadSize}, {});
  tester.AddInput<float>("past_csa_entries", {kBatchSize, 1, 0, kHeadSize}, {});
  tester.AddInput<float>("past_csa_overlap_kv", {kBatchSize, 1, kHeadSize}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_csa_overlap_gate", {kBatchSize, 1, kHeadSize}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_index_pending_kv", {kBatchSize, 0, 2 * kHeadSize}, {});
  tester.AddInput<float>("past_index_pending_gate", {kBatchSize, 0, 2 * kHeadSize}, {});
  tester.AddInput<float>("past_index_entries", {kBatchSize, 1, 0, kHeadSize}, {});
  tester.AddInput<float>("past_index_overlap_kv", {kBatchSize, 1, kHeadSize}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("past_index_overlap_gate", {kBatchSize, 1, kHeadSize}, std::vector<float>(4, 0.0f));

  tester.AddOutput<float>("output", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_key", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_value", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOptionalOutputEdge<float>();
  tester.AddOptionalOutputEdge<float>();
  tester.AddOptionalOutputEdge<float>();
  tester.AddOutput<float>("present_csa_pending_kv", {1, 0, 8}, {});
  tester.AddOutput<float>("present_csa_pending_gate", {1, 0, 8}, {});
  tester.AddOutput<float>("present_csa_entries", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_csa_overlap_kv", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_csa_overlap_gate", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_index_pending_kv", {1, 0, 8}, {});
  tester.AddOutput<float>("present_index_pending_gate", {1, 0, 8}, {});
  tester.AddOutput<float>("present_index_entries", {1, 1, 0, 4}, {});
  tester.AddOutput<float>("present_index_overlap_kv", {1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("present_index_overlap_gate", {1, 1, 4}, std::vector<float>(4, 0.0f));

  auto execution_providers = CpuEpOnly();
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#ifdef USE_CUDA
TEST(DeepSeekV4AttentionTest, CudaSlidingModeRunsMathPath) {
  // Run the sliding-mode math path on the CUDA EP using MLFloat16 tensors
  // (the CUDA kernel registers only MLFloat16 and BFloat16).
  // All weights are zero so all outputs should also be zero.
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4AttributesHalf(tester);
  AddBaseDeepSeekV4InputsHalf(tester);

  tester.AddOutput<MLFloat16>("output", {kBatchSize, kSeqLen, kHiddenSize},
                              MakeHalfValues(static_cast<int>(kBatchSize * kSeqLen * kHiddenSize), 0.0f));
  tester.AddOutput<MLFloat16>("present_key",   {kBatchSize, 1, 1, kHeadSize},
                              MakeHalfValues(static_cast<int>(kBatchSize * kHeadSize), 0.0f));
  tester.AddOutput<MLFloat16>("present_value", {kBatchSize, 1, 1, kHeadSize},
                              MakeHalfValues(static_cast<int>(kBatchSize * kHeadSize), 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_eps;
  cuda_eps.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_eps);
}

TEST(DeepSeekV4AttentionTest, CudaHcaModeUpdatesCompressorState) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4AttributesHalf(tester);
  tester.AddAttribute<std::string>("attention_mode", "hca");
  tester.AddAttribute<float>("compress_rate", 1.0f);
  AddBaseDeepSeekV4InputsHalf(tester);

  tester.AddInput<MLFloat16>("hca_kv_weight", {kHiddenSize, kHeadSize}, MakeHalfValues(16, 0.0f));
  tester.AddInput<MLFloat16>("hca_gate_weight", {kHiddenSize, kHeadSize}, MakeHalfValues(16, 0.0f));
  tester.AddInput<MLFloat16>("hca_position_bias", {1, kHeadSize}, MakeHalfValues(4, 0.0f));
  tester.AddInput<MLFloat16>("hca_kv_norm_weight", {kHeadSize}, MakeHalfValues(4, 1.0f));
  tester.AddInput<MLFloat16>("past_hca_pending_kv", {kBatchSize, 0, kHeadSize}, {});
  tester.AddInput<MLFloat16>("past_hca_pending_gate", {kBatchSize, 0, kHeadSize}, {});
  tester.AddInput<MLFloat16>("past_hca_entries", {kBatchSize, 1, 0, kHeadSize}, {});

  tester.AddOutput<MLFloat16>("output", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_key", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_value", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_hca_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_hca_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_hca_entries", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_eps;
  cuda_eps.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_eps);
}

TEST(DeepSeekV4AttentionTest, CudaCsaModeUpdatesCompressorState) {
  OpTester tester("DeepSeekV4Attention", 1, onnxruntime::kMSDomain);
  AddBaseDeepSeekV4AttributesHalf(tester);
  tester.AddAttribute<std::string>("attention_mode", "csa");
  tester.AddAttribute<float>("compress_rate", 1.0f);
  tester.AddAttribute<int64_t>("index_topk", 1);
  tester.AddAttribute<int64_t>("index_num_heads", 1);
  tester.AddAttribute<int64_t>("index_head_dim", kHeadSize);
  AddBaseDeepSeekV4InputsHalf(tester);
  for (int index = 17; index < 24; ++index) {
    tester.AddOptionalInputEdge<MLFloat16>();
  }

  tester.AddInput<MLFloat16>("csa_kv_weight", {kHiddenSize, 2 * kHeadSize}, MakeHalfValues(32, 0.0f));
  tester.AddInput<MLFloat16>("csa_gate_weight", {kHiddenSize, 2 * kHeadSize}, MakeHalfValues(32, 0.0f));
  tester.AddInput<MLFloat16>("csa_position_bias", {1, 2 * kHeadSize}, MakeHalfValues(8, 0.0f));
  tester.AddInput<MLFloat16>("csa_kv_norm_weight", {kHeadSize}, MakeHalfValues(4, 1.0f));
  tester.AddInput<MLFloat16>("index_kv_weight", {kHiddenSize, 2 * kHeadSize}, MakeHalfValues(32, 0.0f));
  tester.AddInput<MLFloat16>("index_gate_weight", {kHiddenSize, 2 * kHeadSize}, MakeHalfValues(32, 0.0f));
  tester.AddInput<MLFloat16>("index_position_bias", {1, 2 * kHeadSize}, MakeHalfValues(8, 0.0f));
  tester.AddInput<MLFloat16>("index_kv_norm_weight", {kHeadSize}, MakeHalfValues(4, 1.0f));
  tester.AddInput<MLFloat16>("index_q_b_weight", {kQLorRank, kHeadSize}, MakeHalfValues(8, 0.0f));
  tester.AddInput<MLFloat16>("index_weights_proj_weight", {kHiddenSize, 1}, MakeHalfValues(4, 0.0f));
  tester.AddInput<MLFloat16>("past_csa_pending_kv", {1, 0, 8}, {});
  tester.AddInput<MLFloat16>("past_csa_pending_gate", {1, 0, 8}, {});
  tester.AddInput<MLFloat16>("past_csa_entries", {1, 1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_csa_overlap_kv", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddInput<MLFloat16>("past_csa_overlap_gate", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddInput<MLFloat16>("past_index_pending_kv", {1, 0, 8}, {});
  tester.AddInput<MLFloat16>("past_index_pending_gate", {1, 0, 8}, {});
  tester.AddInput<MLFloat16>("past_index_entries", {1, 1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_index_overlap_kv", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddInput<MLFloat16>("past_index_overlap_gate", {1, 1, 4}, MakeHalfValues(4, 0.0f));

  tester.AddOutput<MLFloat16>("output", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_key", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_value", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOptionalOutputEdge<MLFloat16>();
  tester.AddOptionalOutputEdge<MLFloat16>();
  tester.AddOptionalOutputEdge<MLFloat16>();
  tester.AddOutput<MLFloat16>("present_csa_pending_kv", {1, 0, 8}, {});
  tester.AddOutput<MLFloat16>("present_csa_pending_gate", {1, 0, 8}, {});
  tester.AddOutput<MLFloat16>("present_csa_entries", {1, 1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_csa_overlap_kv", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_csa_overlap_gate", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_index_pending_kv", {1, 0, 8}, {});
  tester.AddOutput<MLFloat16>("present_index_pending_gate", {1, 0, 8}, {});
  tester.AddOutput<MLFloat16>("present_index_entries", {1, 1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_index_overlap_kv", {1, 1, 4}, MakeHalfValues(4, 0.0f));
  tester.AddOutput<MLFloat16>("present_index_overlap_gate", {1, 1, 4}, MakeHalfValues(4, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_eps;
  cuda_eps.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_eps);
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
