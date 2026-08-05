// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(DeepSeekV4StandaloneOpsTest, HeavilyCompressedAttentionUpdatesState) {
  OpTester tester("HeavilyCompressedAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 1);
  tester.AddAttribute<int64_t>("rotary_dim", 4);
  tester.AddAttribute<float>("rms_norm_epsilon", 1e-6f);
  tester.AddInput<float>("hidden_states", {1, 1, 4}, std::vector<float>(4, 0.1f));
  tester.AddInput<int64_t>("position_ids", {1, 1}, {0});
  tester.AddInput<float>("cos_cache", {2, 2}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("sin_cache", {2, 2}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("kv_weight", {4, 4}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("gate_weight", {4, 4}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("position_bias", {1, 4}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("norm_weight", {4}, std::vector<float>(4, 1.0f));
  tester.AddInput<float>("past_pending_kv", {1, 0, 4}, {});
  tester.AddInput<float>("past_pending_gate", {1, 0, 4}, {});
  tester.AddInput<float>("past_entries", {1, 1, 0, 4}, {});

  tester.AddOutput<float>("compressed_kv", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddOutput<float>("block_bias", {1, 1, 1, 1}, {0.0f});
  tester.AddOutput<float>("present_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<float>("present_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<float>("present_entries", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CompressedSparseAttentionPreservesCaCbOverlap) {
  OpTester tester("CompressedSparseAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 1);
  tester.AddAttribute<int64_t>("rotary_dim", 2);
  tester.AddAttribute<float>("rms_norm_epsilon", 1e-6f);
  tester.AddInput<float>("hidden_states", {1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  tester.AddInput<int64_t>("position_ids", {1, 2}, {0, 1});
  tester.AddInput<float>("cos_cache", {2, 1}, {1.0f, 1.0f});
  tester.AddInput<float>("sin_cache", {2, 1}, {0.0f, 0.0f});
  tester.AddInput<float>("kv_weight", {2, 4}, {2.0f, 0.0f, 4.0f, 0.0f,
                                                        0.0f, 2.0f, 0.0f, 4.0f});
  tester.AddInput<float>("gate_weight", {2, 4}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("position_bias", {1, 4}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("norm_weight", {2}, {1.0f, 1.0f});
  tester.AddInput<float>("past_pending_kv", {1, 0, 4}, {});
  tester.AddInput<float>("past_pending_gate", {1, 0, 4}, {});
  tester.AddInput<float>("past_entries", {1, 1, 0, 2}, {});
  tester.AddInput<float>("past_overlap_kv", {1, 1, 2}, {6.0f, 0.0f});
  tester.AddInput<float>("past_overlap_gate", {1, 1, 2}, {0.0f, 0.0f});

  const float first = 5.0f / std::sqrt(12.5f + 1e-6f);
  const float second_scale = 1.0f / std::sqrt(2.5f + 1e-6f);
  tester.AddOutput<float>("compressed_kv", {1, 1, 2, 2}, {first, 0.0f, second_scale, 2.0f * second_scale});
  tester.AddOutput<float>("present_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<float>("present_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<float>("present_entries", {1, 1, 2, 2}, {first, 0.0f, second_scale, 2.0f * second_scale});
  tester.AddOutput<float>("present_overlap_kv", {1, 1, 2}, {0.0f, 2.0f});
  tester.AddOutput<float>("present_overlap_gate", {1, 1, 2}, {0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4StandaloneOpsTest, LightningIndexerSelectsCausalTopKAndPadsWithNegativeOne) {
  OpTester tester("LightningIndexer", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 1);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("head_size", 2);
  tester.AddAttribute<int64_t>("index_topk", 2);
  tester.AddAttribute<int64_t>("rotary_dim", 2);
  tester.AddAttribute<float>("rms_norm_epsilon", 1e-6f);
  tester.AddInput<float>("hidden_states", {1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  tester.AddInput<float>("q_residual", {1, 2, 1}, {1.0f, 1.0f});
  tester.AddInput<int64_t>("position_ids", {1, 2}, {0, 1});
  tester.AddInput<float>("cos_cache", {2, 1}, {1.0f, 1.0f});
  tester.AddInput<float>("sin_cache", {2, 1}, {0.0f, 0.0f});
  tester.AddInput<float>("kv_weight", {2, 4}, {1.0f, 0.0f, 1.0f, 0.0f,
                                                        0.0f, 1.0f, 0.0f, 1.0f});
  tester.AddInput<float>("gate_weight", {2, 4}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("position_bias", {1, 4}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("norm_weight", {2}, {1.0f, 1.0f});
  tester.AddInput<float>("q_weight", {1, 2}, {1.0f, 0.0f});
  tester.AddInput<float>("score_weight", {2, 1}, {1.0f, 1.0f});
  tester.AddInput<float>("past_pending_kv", {1, 0, 4}, {});
  tester.AddInput<float>("past_pending_gate", {1, 0, 4}, {});
  tester.AddInput<float>("past_entries", {1, 0, 2}, {});
  tester.AddInput<float>("past_overlap_kv", {1, 1, 2}, {1.0f, 0.0f});
  tester.AddInput<float>("past_overlap_gate", {1, 1, 2}, {0.0f, 0.0f});

  const float first = 1.0f / std::sqrt(0.5f + 1e-6f);
  const float second = 0.5f / std::sqrt(0.25f + 1e-6f);
  tester.AddOutput<int64_t>("selected_indices", {1, 2, 2}, {0, -1, 0, 1});
  tester.AddOutput<float>("present_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<float>("present_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<float>("present_entries", {1, 2, 2}, {first, 0.0f, second, second});
  tester.AddOutput<float>("present_overlap_kv", {1, 1, 2}, {0.0f, 1.0f});
  tester.AddOutput<float>("present_overlap_gate", {1, 1, 2}, {0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CompressedAttentionUsesOneSinkInclusiveSoftmax) {
  OpTester tester("CompressedAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddInput<float>("query", {1, 1, 1, 2}, {0.0f, 0.0f});
  tester.AddInput<float>("local_kv", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 2.0f});
  tester.AddInput<float>("compressed_kv", {1, 1, 2, 2}, {10.0f, 10.0f, 3.0f, 0.0f});
  tester.AddInput<float>("attention_bias", {1, 1, 1, 4}, std::vector<float>(4, 0.0f));
  tester.AddInput<int64_t>("selected_indices", {1, 1, 2}, {1, -1});
  tester.AddInput<float>("head_sink", {}, {0.0f});
  tester.AddOutput<float>("output", {1, 1, 1, 2}, {1.0f, 0.5f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(DeepSeekV4StandaloneOpsTest, HashRouterGathersNormalizesAndScales) {
  for (const std::string& score_function : {std::string("sigmoid"), std::string("sqrtsoftplus")}) {
    OpTester tester("HashRouter", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<std::string>("score_function", score_function);
    tester.AddAttribute<float>("routed_scaling_factor", 2.0f);
    tester.AddInput<float>("hidden_states", {1, 1}, {1.0f});
    tester.AddInput<int64_t>("input_ids", {1}, {1});
    tester.AddInput<float>("gate_weight", {3, 1}, {0.0f, 1.0f, 2.0f});
    tester.AddInput<int64_t>("token_to_expert", {2, 2}, {0, 1, 2, 0});
    tester.AddOutput<float>("logits", {1, 3}, {0.0f, 1.0f, 2.0f});
    const auto activate = [&](float value) {
      return score_function == "sigmoid" ? 1.0f / (1.0f + std::exp(-value))
                                          : std::sqrt(std::log1p(std::exp(value)));
    };
    const float first = activate(2.0f);
    const float second = activate(0.0f);
    tester.AddOutput<float>("routing_weights", {1, 2},
                            {2.0f * first / (first + second), 2.0f * second / (first + second)});
    tester.AddOutput<int64_t>("expert_indices", {1, 2}, {2, 0});

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(DeepSeekV4StandaloneOpsTest, HyperConnectionComputesThreeModuleOutputs) {
  OpTester tester("HyperConnection", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddAttribute<int64_t>("sinkhorn_iterations", 1);
  tester.AddInput<float>("hidden_streams", {1, 1, 2, 1}, {1.0f, 3.0f});
  tester.AddInput<float>("projection_weight", {8, 2}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("projection_bias", {8}, {std::log(3.0f), std::log(3.0f), 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0.0f});
  tester.AddInput<float>("projection_scale", {3}, {2.0f, 2.0f, 2.0f});
  tester.AddOutput<float>("post", {1, 1, 2}, {1.0f, 1.0f});
  tester.AddOutput<float>("comb", {1, 1, 2, 2}, std::vector<float>(4, 0.5f));
  tester.AddOutput<float>("collapsed", {1, 1, 1}, {3.000004f});
  tester.SetOutputAbsErr("comb", 1e-5f);
  tester.SetOutputAbsErr("collapsed", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, HyperHeadCollapsesStreams) {
  OpTester tester("HyperHead", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddInput<float>("hidden_streams", {1, 1, 2, 1}, {1.0f, 3.0f});
  tester.AddInput<float>("projection_weight", {2, 2}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("projection_bias", {2}, {std::log(3.0f), std::log(3.0f)});
  tester.AddInput<float>("projection_scale", {1}, {2.0f});
  tester.AddOutput<float>("output", {1, 1, 1}, {3.000004f});
  tester.SetOutputAbsErr("output", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, SinkhornNormalizeStartsWithColumns) {
  OpTester tester("SinkhornNormalize", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("iterations", 1);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddInput<float>("X", {1, 2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
  tester.AddOutput<float>("Y", {1, 2, 2},
                          {1.0f / 3.000001f, 3.0f / 7.000001f,
                           2.0f / 3.000001f, 4.0f / 7.000001f});
  tester.SetOutputAbsErr("Y", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, HyperConnectionMixFusesAdjacentBoundary) {
  OpTester tester("HyperConnectionMix", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("sinkhorn_iterations", 1);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddAttribute<float>("hc_epsilon", 1e-6f);
  tester.AddAttribute<float>("sinkhorn_epsilon", 1e-6f);
  tester.AddAttribute<float>("post_alpha", 2.0f);
  tester.AddInput<float>("x", {1, 1, 1}, {2.0f});
  tester.AddInput<float>("residual", {1, 1, 2, 1}, {1.0f, 3.0f});
  tester.AddInput<float>("post_mix", {1, 1, 2}, {1.0f, 2.0f});
  tester.AddInput<float>("comb_mix", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  tester.AddInput<float>("fn", {2, 8}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("scale", {3}, {1.0f, 1.0f, 1.0f});
  tester.AddInput<float>("base", {8}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("norm_weight", {1}, {2.0f});
  tester.AddOutput<float>("residual_out", {1, 1, 2, 1}, {3.0f, 7.0f});
  tester.AddOutput<float>("post_mix_out", {1, 1, 2}, {1.0f, 1.0f});
  tester.AddOutput<float>("comb_mix_out", {1, 1, 2, 2}, std::vector<float>(4, 0.5f));
  tester.AddOutput<float>("layer_input", {1, 1, 1}, {2.0f});
  tester.SetOutputAbsErr("comb_mix_out", 1e-5f);
  tester.SetOutputAbsErr("layer_input", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#ifdef USE_CUDA

static std::vector<MLFloat16> ToHalf(const std::vector<float>& values) {
  std::vector<MLFloat16> result(values.size());
  ConvertFloatToMLFloat16(values.data(), result.data(), values.size());
  return result;
}

TEST(DeepSeekV4StandaloneOpsTest, CudaSinkhornNormalizeStartsWithColumns) {
  OpTester tester("SinkhornNormalize", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("iterations", 1);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddInput<float>("X", {1, 2, 2}, {1.0f, 3.0f, 2.0f, 4.0f});
  tester.AddOutput<float>("Y", {1, 2, 2},
                          {1.0f / 3.000001f, 3.0f / 7.000001f,
                           2.0f / 3.000001f, 4.0f / 7.000001f});
  tester.SetOutputAbsErr("Y", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaHyperConnectionMixFusesAdjacentBoundary) {
  OpTester tester("HyperConnectionMix", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("sinkhorn_iterations", 1);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddAttribute<float>("hc_epsilon", 1e-6f);
  tester.AddAttribute<float>("sinkhorn_epsilon", 1e-6f);
  tester.AddAttribute<float>("post_alpha", 2.0f);
  tester.AddInput<float>("x", {1, 1, 1}, {2.0f});
  tester.AddInput<float>("residual", {1, 1, 2, 1}, {1.0f, 3.0f});
  tester.AddInput<float>("post_mix", {1, 1, 2}, {1.0f, 2.0f});
  tester.AddInput<float>("comb_mix", {1, 1, 2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
  tester.AddInput<float>("fn", {2, 8}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("scale", {3}, {1.0f, 1.0f, 1.0f});
  tester.AddInput<float>("base", {8}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("norm_weight", {1}, {2.0f});
  tester.AddOutput<float>("residual_out", {1, 1, 2, 1}, {3.0f, 7.0f});
  tester.AddOutput<float>("post_mix_out", {1, 1, 2}, {1.0f, 1.0f});
  tester.AddOutput<float>("comb_mix_out", {1, 1, 2, 2}, std::vector<float>(4, 0.5f));
  tester.AddOutput<float>("layer_input", {1, 1, 1}, {2.0f});
  tester.SetOutputAbsErr("comb_mix_out", 1e-5f);
  tester.SetOutputAbsErr("layer_input", 1e-5f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaHyperConnectionComputesThreeModuleOutputs) {
  OpTester tester("HyperConnection", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddAttribute<int64_t>("sinkhorn_iterations", 1);
  tester.AddInput<MLFloat16>("hidden_streams", {1, 1, 2, 1}, ToHalf({1.0f, 3.0f}));
  tester.AddInput<float>("projection_weight", {8, 2}, std::vector<float>(16, 0.0f));
  tester.AddInput<float>("projection_bias", {8}, {std::log(3.0f), std::log(3.0f), 0.0f, 0.0f,
                                                   0.0f, 0.0f, 0.0f, 0.0f});
  tester.AddInput<float>("projection_scale", {3}, {2.0f, 2.0f, 2.0f});
  tester.AddOutput<MLFloat16>("post", {1, 1, 2}, ToHalf({1.0f, 1.0f}));
  tester.AddOutput<MLFloat16>("comb", {1, 1, 2, 2}, ToHalf(std::vector<float>(4, 0.5f)));
  tester.AddOutput<MLFloat16>("collapsed", {1, 1, 1}, ToHalf({3.000004f}));
  tester.SetOutputAbsErr("comb", 0.002f);
  tester.SetOutputAbsErr("collapsed", 0.002f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaHyperHeadCollapsesStreams) {
  OpTester tester("HyperHead", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("epsilon", 1e-6f);
  tester.AddInput<MLFloat16>("hidden_streams", {1, 1, 2, 1}, ToHalf({1.0f, 3.0f}));
  tester.AddInput<float>("projection_weight", {2, 2}, std::vector<float>(4, 0.0f));
  tester.AddInput<float>("projection_bias", {2}, {std::log(3.0f), std::log(3.0f)});
  tester.AddInput<float>("projection_scale", {1}, {2.0f});
  tester.AddOutput<MLFloat16>("output", {1, 1, 1}, ToHalf({3.000004f}));
  tester.SetOutputAbsErr("output", 0.002f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaHeavilyCompressedAttentionCompressesAndMasksCausally) {
  OpTester tester("HeavilyCompressedAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 2);
  tester.AddAttribute<int64_t>("rotary_dim", 2);
  tester.AddInput<MLFloat16>("hidden_states", {1, 2, 2}, ToHalf({1.0f, 0.0f, 0.0f, 1.0f}));
  tester.AddInput<int64_t>("position_ids", {1, 2}, {0, 1});
  tester.AddInput<MLFloat16>("cos_cache", {2, 1}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("sin_cache", {2, 1}, ToHalf({0.0f, 0.0f}));
  tester.AddInput<MLFloat16>("kv_weight", {2, 2}, ToHalf({1.0f, 0.0f, 0.0f, 1.0f}));
  tester.AddInput<MLFloat16>("gate_weight", {2, 2}, ToHalf(std::vector<float>(4, 0.0f)));
  tester.AddInput<MLFloat16>("position_bias", {2, 2}, ToHalf(std::vector<float>(4, 0.0f)));
  tester.AddInput<MLFloat16>("norm_weight", {2}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("past_pending_kv", {1, 0, 2}, {});
  tester.AddInput<MLFloat16>("past_pending_gate", {1, 0, 2}, {});
  tester.AddInput<MLFloat16>("past_entries", {1, 1, 0, 2}, {});
  const auto compressed = ToHalf({1.0f / std::sqrt(1.0f + 4e-6f), 1.0f / std::sqrt(1.0f + 4e-6f)});
  tester.AddOutput<MLFloat16>("compressed_kv", {1, 1, 1, 2}, compressed);
  tester.AddOutput<MLFloat16>("block_bias", {1, 1, 2, 1},
                              ToHalf({-std::numeric_limits<float>::infinity(), 0.0f}));
  tester.AddOutput<MLFloat16>("present_pending_kv", {1, 0, 2}, {});
  tester.AddOutput<MLFloat16>("present_pending_gate", {1, 0, 2}, {});
  tester.AddOutput<MLFloat16>("present_entries", {1, 1, 1, 2}, compressed);
  tester.SetOutputAbsErr("compressed_kv", 0.01f);
  tester.SetOutputAbsErr("present_entries", 0.01f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaCompressedSparseAttentionUsesOverlap) {
  OpTester tester("CompressedSparseAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 1);
  tester.AddAttribute<int64_t>("rotary_dim", 2);
  tester.AddInput<MLFloat16>("hidden_states", {1, 2, 2}, ToHalf({1.0f, 0.0f, 0.0f, 1.0f}));
  tester.AddInput<int64_t>("position_ids", {1, 2}, {0, 1});
  tester.AddInput<MLFloat16>("cos_cache", {2, 1}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("sin_cache", {2, 1}, ToHalf({0.0f, 0.0f}));
  tester.AddInput<MLFloat16>("kv_weight", {2, 4}, ToHalf({2.0f, 0.0f, 4.0f, 0.0f,
                                                            0.0f, 2.0f, 0.0f, 4.0f}));
  tester.AddInput<MLFloat16>("gate_weight", {2, 4}, ToHalf(std::vector<float>(8, 0.0f)));
  tester.AddInput<MLFloat16>("position_bias", {1, 4}, ToHalf(std::vector<float>(4, 0.0f)));
  tester.AddInput<MLFloat16>("norm_weight", {2}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("past_pending_kv", {1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_pending_gate", {1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_entries", {1, 1, 0, 2}, {});
  tester.AddInput<MLFloat16>("past_overlap_kv", {1, 1, 2}, ToHalf({6.0f, 0.0f}));
  tester.AddInput<MLFloat16>("past_overlap_gate", {1, 1, 2}, ToHalf({0.0f, 0.0f}));
  const float first = 5.0f / std::sqrt(12.5f + 1e-6f);
  const float second_scale = 1.0f / std::sqrt(2.5f + 1e-6f);
  const auto entries = ToHalf({first, 0.0f, second_scale, 2.0f * second_scale});
  tester.AddOutput<MLFloat16>("compressed_kv", {1, 1, 2, 2}, entries);
  tester.AddOutput<MLFloat16>("present_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_entries", {1, 1, 2, 2}, entries);
  tester.AddOutput<MLFloat16>("present_overlap_kv", {1, 1, 2}, ToHalf({0.0f, 2.0f}));
  tester.AddOutput<MLFloat16>("present_overlap_gate", {1, 1, 2}, ToHalf({0.0f, 0.0f}));
  tester.SetOutputAbsErr("compressed_kv", 0.02f);
  tester.SetOutputAbsErr("present_entries", 0.02f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaLightningIndexerSelectsAndPads) {
  OpTester tester("LightningIndexer", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("compress_rate", 1);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("head_size", 2);
  tester.AddAttribute<int64_t>("index_topk", 2);
  tester.AddAttribute<int64_t>("rotary_dim", 2);
  tester.AddInput<MLFloat16>("hidden_states", {1, 2, 2}, ToHalf({1.0f, 0.0f, 0.0f, 1.0f}));
  tester.AddInput<MLFloat16>("q_residual", {1, 2, 1}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<int64_t>("position_ids", {1, 2}, {0, 1});
  tester.AddInput<MLFloat16>("cos_cache", {2, 1}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("sin_cache", {2, 1}, ToHalf({0.0f, 0.0f}));
  tester.AddInput<MLFloat16>("kv_weight", {2, 4}, ToHalf({1.0f, 0.0f, 1.0f, 0.0f,
                                                            0.0f, 1.0f, 0.0f, 1.0f}));
  tester.AddInput<MLFloat16>("gate_weight", {2, 4}, ToHalf(std::vector<float>(8, 0.0f)));
  tester.AddInput<MLFloat16>("position_bias", {1, 4}, ToHalf(std::vector<float>(4, 0.0f)));
  tester.AddInput<MLFloat16>("norm_weight", {2}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("q_weight", {1, 2}, ToHalf({1.0f, 0.0f}));
  tester.AddInput<MLFloat16>("score_weight", {2, 1}, ToHalf({1.0f, 1.0f}));
  tester.AddInput<MLFloat16>("past_pending_kv", {1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_pending_gate", {1, 0, 4}, {});
  tester.AddInput<MLFloat16>("past_entries", {1, 0, 2}, {});
  tester.AddInput<MLFloat16>("past_overlap_kv", {1, 1, 2}, ToHalf({1.0f, 0.0f}));
  tester.AddInput<MLFloat16>("past_overlap_gate", {1, 1, 2}, ToHalf({0.0f, 0.0f}));
  tester.AddOutput<int64_t>("selected_indices", {1, 2, 2}, {0, -1, 0, 1});
  tester.AddOutput<MLFloat16>("present_pending_kv", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_pending_gate", {1, 0, 4}, {});
  tester.AddOutput<MLFloat16>("present_entries", {1, 2, 2},
                              ToHalf({1.0f / std::sqrt(0.5f + 1e-6f), 0.0f,
                                      0.5f / std::sqrt(0.25f + 1e-6f),
                                      0.5f / std::sqrt(0.25f + 1e-6f)}));
  tester.AddOutput<MLFloat16>("present_overlap_kv", {1, 1, 2}, ToHalf({0.0f, 1.0f}));
  tester.AddOutput<MLFloat16>("present_overlap_gate", {1, 1, 2}, ToHalf({0.0f, 0.0f}));
  tester.SetOutputAbsErr("present_entries", 0.02f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaCompressedAttentionSelectsWithSharedSinkSoftmax) {
  OpTester tester("CompressedAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddInput<MLFloat16>("query", {1, 1, 1, 2}, ToHalf({0.0f, 0.0f}));
  tester.AddInput<MLFloat16>("local_kv", {1, 1, 2, 2}, ToHalf({1.0f, 0.0f, 0.0f, 2.0f}));
  tester.AddInput<MLFloat16>("compressed_kv", {1, 1, 2, 2}, ToHalf({10.0f, 10.0f, 3.0f, 0.0f}));
  tester.AddInput<MLFloat16>("attention_bias", {1, 1, 1, 4}, ToHalf(std::vector<float>(4, 0.0f)));
  tester.AddInput<int64_t>("selected_indices", {1, 1, 2}, {1, -1});
  tester.AddInput<MLFloat16>("head_sink", {}, ToHalf({0.0f}));
  tester.AddOutput<MLFloat16>("output", {1, 1, 1, 2}, ToHalf({1.0f, 0.5f}));
  tester.SetOutputAbsErr("output", 0.01f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(DeepSeekV4StandaloneOpsTest, CudaHashRouterProjectsGathersAndNormalizesInt32) {
  OpTester tester("HashRouter", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("score_function", "sigmoid");
  tester.AddAttribute<float>("routed_scaling_factor", 2.0f);
  tester.AddInput<MLFloat16>("hidden_states", {1, 2}, ToHalf({1.0f, 2.0f}));
  tester.AddInput<int32_t>("input_ids", {1}, {1});
  tester.AddInput<MLFloat16>("gate_weight", {3, 2}, ToHalf({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}));
  tester.AddInput<int32_t>("token_to_expert", {2, 2}, {0, 1, 2, 0});
  tester.AddOutput<MLFloat16>("logits", {1, 3}, ToHalf({1.0f, 2.0f, 3.0f}));
  const float first = 1.0f / (1.0f + std::exp(-3.0f));
  const float second = 1.0f / (1.0f + std::exp(-1.0f));
  tester.AddOutput<MLFloat16>("routing_weights", {1, 2},
                              ToHalf({2.0f * first / (first + second), 2.0f * second / (first + second)}));
  tester.AddOutput<int32_t>("expert_indices", {1, 2}, {2, 0});
  tester.SetOutputAbsErr("routing_weights", 0.01f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#endif

}  // namespace test
}  // namespace onnxruntime