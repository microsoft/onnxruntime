// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

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

}  // namespace

TEST(EngramOpsTest, NgramHashMappingInt64) {
  OpTester test("NgramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", 3);
  test.AddAttribute<int64_t>("n_head_per_ngram", 2);
  test.AddAttribute<int64_t>("pad_id", 9);
  test.AddInput<int64_t>("input_ids", {1, 4}, {3, 4, 5, 6});
  test.AddInput<int64_t>("multipliers", {3}, {11, 13, 17});
  test.AddInput<int64_t>("vocab_sizes", {4}, {101, 103, 107, 109});
  test.AddOutput<int64_t>("hash_ids", {1, 4, 4},
                          {84, 84, 98, 96,
                           11, 11, 39, 37,
                           3, 3, 48, 48,
                           3, 3, 71, 71});
  test.Run();
}

TEST(EngramOpsTest, ShortConvFloat) {
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
  test.AddInput<float>("input", {1, 2, 1, 2}, input);
  test.AddInput<float>("weight", {2, 1, 2}, weight);
  test.AddInput<float>("norm_scale", {1, 2}, scale);
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("output", {1, 2, 1, 2}, expected);
  test.Run();
}

TEST(EngramOpsTest, EngramGateFloat) {
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
  const float gate = Sigmoid(std::copysign(std::sqrt(std::max(std::abs(dot), 1.0e-6f)), dot));
  const std::vector<float> expected{gate * (1.0f * 1.0f + 2.0f * 0.5f),
                                    gate * (1.0f * -1.0f + 2.0f * 0.25f)};

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<float>("embeddings", {1, 1, 2}, embeddings);
  test.AddInput<float>("hidden_states", {1, 1, 1, 2}, hidden_states);
  test.AddInput<float>("key_weight", {1, 2, 2}, key_weight);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("value_weight", {2, 2}, value_weight);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("key_norm_scale", {1, 2}, key_scale);
  test.AddInput<float>("query_norm_scale", {1, 2}, query_scale);
  test.AddOutput<float>("output", {1, 1, 1, 2}, expected);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
