// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <vector>

#include <memory>

#include "gtest/gtest.h"
#include "core/framework/execution_provider.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

float Sigmoid(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// Returns false when the execution provider required by T is unavailable. This must be checked before
// an OpTester is constructed: BaseTester's destructor traps when a tester is destroyed without running.
template <typename T>
bool IsTypeSupported() {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return DefaultCudaExecutionProvider() != nullptr;
  } else {
    return true;
  }
}

// Runs the tester on CUDA only for BFloat16, otherwise on the default set of execution providers.
template <typename T>
void RunOnSupportedProviders(OpTester& test) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  } else {
    test.Run();
  }
}

// Reference for the gate pre-activation, sign(dot) * sqrt(max(abs(dot), 1e-6)).
// std::copysign is deliberately avoided: it maps a zero dot product to +sqrt(1e-6) instead of zero.
float GateArg(float dot) {
  if (dot == 0.0f) {
    return 0.0f;
  }
  const float magnitude = std::sqrt(std::max(std::abs(dot), 1.0e-6f));
  return dot < 0.0f ? -magnitude : magnitude;
}

template <typename T>
void RunShortConvTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr float epsilon = 1.0e-5f;
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> scale{1.0f, 2.0f};
  const std::vector<float> weight{0.25f, 0.5f, 0.75f, -0.5f};

  std::vector<float> normed(4);
  for (int64_t t = 0; t < 2; ++t) {
    float sum_sq = input[t * 2] * input[t * 2] + input[t * 2 + 1] * input[t * 2 + 1];
    float inv_rms = 1.0f / std::sqrt(sum_sq / 2.0f + epsilon);
    for (int64_t c = 0; c < 2; ++c) {
      normed[t * 2 + c] = input[t * 2 + c] * inv_rms * scale[c];
    }
  }
  std::vector<float> expected(4);
  for (int64_t t = 0; t < 2; ++t) {
    for (int64_t c = 0; c < 2; ++c) {
      float sum = 0.0f;
      if (t > 0) {
        sum += normed[(t - 1) * 2 + c] * weight[c * 2];
      }
      sum += normed[t * 2 + c] * weight[c * 2 + 1];
      expected[t * 2 + c] = sum * Sigmoid(sum);
    }
  }

  OpTester test("ShortConv", 1, kMSDomain);
  test.AddAttribute<int64_t>("dilation", 1);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<std::string>("activation", "silu");
  test.AddInput<T>("input", {1, 2, 1, 2}, ToTensorType<T>(input));
  test.AddInput<T>("weight", {2, 1, 2}, ToTensorType<T>(weight));
  test.AddInput<T>("norm_scale", {1, 2}, ToTensorType<T>(scale));
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("output", {1, 2, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);
  RunOnSupportedProviders<T>(test);
}

template <typename T>
void RunEngramGateTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr float epsilon = 1.0e-5f;
  const std::vector<float> embeddings{1.0f, 2.0f};
  const std::vector<float> hidden_states{3.0f, 4.0f};
  const std::vector<float> key_weight{0.5f, 1.0f, -0.25f, 0.75f};
  const std::vector<float> value_weight{1.0f, -1.0f, 0.5f, 0.25f};
  const std::vector<float> key_scale{1.0f, 1.0f};
  const std::vector<float> query_scale{1.0f, 1.0f};

  const float key0 = 1.0f * 0.5f + 2.0f * -0.25f;
  const float key1 = 1.0f * 1.0f + 2.0f * 0.75f;
  const float key_inv = 1.0f / std::sqrt((key0 * key0 + key1 * key1) / 2.0f + epsilon);
  const float query_inv = 1.0f / std::sqrt((3.0f * 3.0f + 4.0f * 4.0f) / 2.0f + epsilon);
  const float dot = (key0 * key_inv * 3.0f * query_inv + key1 * key_inv * 4.0f * query_inv) / std::sqrt(2.0f);
  const float gate = Sigmoid(GateArg(dot));
  const std::vector<float> expected{gate * (1.0f * 1.0f + 2.0f * 0.5f),
                                    gate * (1.0f * -1.0f + 2.0f * 0.25f)};

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<T>("embeddings", {1, 1, 2}, ToTensorType<T>(embeddings));
  test.AddInput<T>("hidden_states", {1, 1, 1, 2}, ToTensorType<T>(hidden_states));
  test.AddInput<T>("key_weight", {1, 2, 2}, ToTensorType<T>(key_weight));
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("value_weight", {2, 2}, ToTensorType<T>(value_weight));
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("key_norm_scale", {1, 2}, ToTensorType<T>(key_scale));
  test.AddInput<T>("query_norm_scale", {1, 2}, ToTensorType<T>(query_scale));
  test.AddOutput<T>("output", {1, 1, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);
  RunOnSupportedProviders<T>(test);
}

template <typename T>
void RunNGramHashMappingTest() {
  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", 3);
  test.AddAttribute<int64_t>("n_head_per_ngram", 2);
  test.AddAttribute<int64_t>("pad_id", 9);
  test.AddInput<T>("input_ids", {1, 4}, {3, 4, 5, 6});
  test.AddInput<T>("multipliers", {3}, {11, 13, 17});
  test.AddInput<T>("vocab_sizes", {4}, {101, 103, 107, 109});
  test.AddOutput<T>("hash_ids", {1, 4, 4},
                    {84, 84, 98, 96,
                     11, 11, 39, 37,
                     3, 3, 48, 48,
                     3, 3, 71, 71});
  test.Run();
}

// Verifies head_offsets is applied as a fixed additive offset after the modulo, per output head,
// on top of the same base hash computation as RunNGramHashMappingTest.
template <typename T>
void RunNGramHashMappingHeadOffsetsTest() {
  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", 3);
  test.AddAttribute<int64_t>("n_head_per_ngram", 2);
  test.AddAttribute<int64_t>("pad_id", 9);
  test.AddInput<T>("input_ids", {1, 4}, {3, 4, 5, 6});
  test.AddInput<T>("multipliers", {3}, {11, 13, 17});
  test.AddInput<T>("vocab_sizes", {4}, {101, 103, 107, 109});
  test.AddOptionalInputEdge<T>();  // past_tokens
  test.AddInput<T>("head_offsets", {4}, {1000, 2000, 3000, 4000});
  test.AddOutput<T>("hash_ids", {1, 4, 4},
                    {1084, 2084, 3098, 4096,
                     1011, 2011, 3039, 4037,
                     1003, 2003, 3048, 4048,
                     1003, 2003, 3071, 4071});
  test.Run();
}

// Verifies reset_on_eos: an EOS token inside (or implied before, since past_tokens is absent) the
// current chunk substitutes eos_token_id for any n-gram shift that would otherwise reach across
// the EOS boundary into unrelated history.
template <typename T>
void RunNGramHashMappingEosResetTest() {
  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", 3);
  test.AddAttribute<int64_t>("n_head_per_ngram", 1);
  test.AddAttribute<int64_t>("pad_id", 0);
  test.AddAttribute<int64_t>("reset_on_eos", 1);
  test.AddInput<T>("input_ids", {1, 4}, {3, 9, 5, 6});
  test.AddInput<T>("multipliers", {3}, {11, 13, 17});
  test.AddInput<T>("vocab_sizes", {2}, {101, 103});
  test.AddOptionalInputEdge<T>();  // past_tokens
  test.AddOptionalInputEdge<T>();  // head_offsets
  test.AddInput<T>("eos_token_id", {}, {9});
  test.AddOutput<T>("hash_ids", {1, 4, 2},
                    {84, 102,
                     68, 15,
                     66, 13,
                     3, 51});
  test.Run();
}

// Verifies segment_ids resets causal history at packed-sequence boundaries within input_ids,
// independent of reset_on_eos/eos_token_id.
template <typename T>
void RunNGramHashMappingSegmentIdsTest() {
  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", 3);
  test.AddAttribute<int64_t>("n_head_per_ngram", 1);
  test.AddAttribute<int64_t>("pad_id", 0);
  test.AddInput<T>("input_ids", {1, 4}, {3, 4, 5, 6});
  test.AddInput<T>("multipliers", {3}, {11, 13, 17});
  test.AddInput<T>("vocab_sizes", {2}, {101, 103});
  test.AddOptionalInputEdge<T>();  // past_tokens
  test.AddOptionalInputEdge<T>();  // head_offsets
  test.AddOptionalInputEdge<T>();  // eos_token_id
  test.AddInput<int32_t>("segment_ids", {1, 4}, {0, 0, 1, 1});
  test.AddOutput<T>("hash_ids", {1, 4, 2},
                    {33, 33,
                     11, 11,
                     3, 48,
                     66, 66});
  test.Run();
}

// Verifies past_tokens/present_tokens round-trip parity: splitting a sequence into two decode
// chunks and threading present_tokens from the first chunk into past_tokens of the second must
// produce the same hash_ids as running the whole sequence in a single prefill call.
template <typename T>
void RunNGramHashMappingPastPresentParityTest() {
  constexpr int64_t history_length = 2;  // max_ngram_size - 1

  {
    OpTester full("NGramHashMapping", 1, kMSDomain);
    full.AddAttribute<int64_t>("max_ngram_size", 3);
    full.AddAttribute<int64_t>("n_head_per_ngram", 1);
    full.AddAttribute<int64_t>("pad_id", 0);
    full.AddInput<T>("input_ids", {1, 5}, {2, 3, 4, 5, 6});
    full.AddInput<T>("multipliers", {3}, {11, 13, 17});
    full.AddInput<T>("vocab_sizes", {2}, {101, 103});
    full.AddOutput<T>("hash_ids", {1, 5, 2},
                      {22, 22,
                       59, 59,
                       11, 41,
                       3, 48,
                       3, 71});
    full.Run();
  }

  {
    OpTester chunk1("NGramHashMapping", 1, kMSDomain);
    chunk1.AddAttribute<int64_t>("max_ngram_size", 3);
    chunk1.AddAttribute<int64_t>("n_head_per_ngram", 1);
    chunk1.AddAttribute<int64_t>("pad_id", 0);
    chunk1.AddInput<T>("input_ids", {1, 2}, {2, 3});
    chunk1.AddInput<T>("multipliers", {3}, {11, 13, 17});
    chunk1.AddInput<T>("vocab_sizes", {2}, {101, 103});
    chunk1.AddOutput<T>("hash_ids", {1, 2, 2}, {22, 22, 59, 59});
    chunk1.AddOutput<T>("present_tokens", {1, history_length}, {2, 3});
    chunk1.Run();
  }

  {
    OpTester chunk2("NGramHashMapping", 1, kMSDomain);
    chunk2.AddAttribute<int64_t>("max_ngram_size", 3);
    chunk2.AddAttribute<int64_t>("n_head_per_ngram", 1);
    chunk2.AddAttribute<int64_t>("pad_id", 0);
    chunk2.AddInput<T>("input_ids", {1, 3}, {4, 5, 6});
    chunk2.AddInput<T>("multipliers", {3}, {11, 13, 17});
    chunk2.AddInput<T>("vocab_sizes", {2}, {101, 103});
    chunk2.AddInput<T>("past_tokens", {1, history_length}, {2, 3});
    // Matches hash_ids[2:5] from the full-sequence run above.
    chunk2.AddOutput<T>("hash_ids", {1, 3, 2},
                        {11, 41,
                         3, 48,
                         3, 71});
    chunk2.AddOutput<T>("present_tokens", {1, history_length}, {5, 6});
    chunk2.Run();
  }
}

}  // namespace

TEST(EngramOpsTest, NGramHashMappingInt64) {
  RunNGramHashMappingTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so it must be covered explicitly.
TEST(EngramOpsTest, NGramHashMappingInt32) {
  RunNGramHashMappingTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingHeadOffsetsInt64) {
  RunNGramHashMappingHeadOffsetsTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingHeadOffsetsInt32) {
  RunNGramHashMappingHeadOffsetsTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingEosResetInt64) {
  RunNGramHashMappingEosResetTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingEosResetInt32) {
  RunNGramHashMappingEosResetTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingSegmentIdsInt64) {
  RunNGramHashMappingSegmentIdsTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingSegmentIdsInt32) {
  RunNGramHashMappingSegmentIdsTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingPastPresentParityInt64) {
  RunNGramHashMappingPastPresentParityTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingPastPresentParityInt32) {
  RunNGramHashMappingPastPresentParityTest<int32_t>();
}

TEST(EngramOpsTest, ShortConvFloat) {
  RunShortConvTest<float>(1e-4f);
}

TEST(EngramOpsTest, ShortConvFloat16) {
  RunShortConvTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, EngramGateFloat) {
  RunEngramGateTest<float>(1e-4f);
}

TEST(EngramOpsTest, EngramGateFloat16) {
  RunEngramGateTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, ShortConvBFloat16) {
  RunShortConvTest<BFloat16>(2e-2f);
}

TEST(EngramOpsTest, EngramGateBFloat16) {
  RunEngramGateTest<BFloat16>(2e-2f);
}

// A zero dot product must produce a gate of exactly 0.5 on every EP. Orthogonal key/query rows make
// the dot product vanish, which would silently become sigmoid(sqrt(1e-6)) if copysign were used.
TEST(EngramOpsTest, EngramGateZeroDotProduct) {
  constexpr float epsilon = 1.0e-5f;
  // key = embeddings * key_weight = (1, 0); query = hidden_states = (0, 1), so the dot product is 0.
  const std::vector<float> embeddings{1.0f, 0.0f};
  const std::vector<float> hidden_states{0.0f, 1.0f};
  const std::vector<float> key_weight{1.0f, 0.0f, 0.0f, 1.0f};
  const std::vector<float> value_weight{1.0f, -1.0f, 0.5f, 0.25f};
  const std::vector<float> unit_scale{1.0f, 1.0f};
  const std::vector<float> expected{0.5f * 1.0f, 0.5f * -1.0f};

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<float>("embeddings", {1, 1, 2}, embeddings);
  test.AddInput<float>("hidden_states", {1, 1, 1, 2}, hidden_states);
  test.AddInput<float>("key_weight", {1, 2, 2}, key_weight);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("value_weight", {2, 2}, value_weight);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("key_norm_scale", {1, 2}, unit_scale);
  test.AddInput<float>("query_norm_scale", {1, 2}, unit_scale);
  test.AddOutput<float>("output", {1, 1, 1, 2}, expected, false, 1e-5f, 1e-5f);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
