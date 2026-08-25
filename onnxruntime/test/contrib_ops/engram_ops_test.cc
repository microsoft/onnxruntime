// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
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

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else {
    return data;
  }
}

template <typename T>
void RunShortConvTest(float tolerance) {
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
  test.Run();
}

template <typename T>
void RunEngramGateTest(float tolerance) {
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
  test.AddInput<T>("embeddings", {1, 1, 2}, ToTensorType<T>(embeddings));
  test.AddInput<T>("hidden_states", {1, 1, 1, 2}, ToTensorType<T>(hidden_states));
  test.AddInput<T>("key_weight", {1, 2, 2}, ToTensorType<T>(key_weight));
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("value_weight", {2, 2}, ToTensorType<T>(value_weight));
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("key_norm_scale", {1, 2}, ToTensorType<T>(key_scale));
  test.AddInput<T>("query_norm_scale", {1, 2}, ToTensorType<T>(query_scale));
  test.AddOutput<T>("output", {1, 1, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);
  test.Run();
}

}  // namespace

TEST(EngramOpsTest, NGramHashMappingInt64) {
  OpTester test("NGramHashMapping", 1, kMSDomain);
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

}  // namespace test
}  // namespace onnxruntime
