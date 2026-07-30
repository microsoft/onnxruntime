// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the two fused linear-attention gate ops: LinearAttentionGate and GatedRMSNorm.
// Both replace float32 elementwise chains that an exporter emits around the LinearAttention op,
// so the references here are the same float32 formulas ORT's Softplus/Sigmoid/RMSNorm kernels use.

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

float SigmoidRef(float x) {
  return x > 0.0f ? 1.0f / (1.0f + std::exp(-x)) : 1.0f - 1.0f / (1.0f + std::exp(x));
}

float SoftplusRef(float x) {
  return x > 0.0f ? x + std::log(std::exp(-x) + 1.0f) : std::log(std::exp(x) + 1.0f);
}

std::vector<float> RandomFloats(size_t count, float lo, float hi, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> out(count);
  for (auto& v : out) {
    v = dist(gen);
  }
  return out;
}

template <typename T>
void RunLinearAttentionGateTest(int batch_size, int seq_length, int num_heads, bool with_beta,
                                float tolerance) {
  const size_t count = static_cast<size_t>(batch_size) * seq_length * num_heads;
  const auto a = RandomFloats(count, -6.0f, 6.0f, 11);
  const auto b = RandomFloats(count, -6.0f, 6.0f, 12);
  const auto dt_bias = RandomFloats(static_cast<size_t>(num_heads), -2.0f, 2.0f, 13);
  const auto decay_scale = RandomFloats(static_cast<size_t>(num_heads), -4.0f, -0.1f, 14);

  std::vector<float> expected_decay(count);
  std::vector<float> expected_beta(count);
  for (size_t i = 0; i < count; ++i) {
    const int h = static_cast<int>(i % num_heads);
    expected_decay[i] = decay_scale[h] * SoftplusRef(a[i] + dt_bias[h]);
    expected_beta[i] = SigmoidRef(b[i]);
  }

  const std::vector<int64_t> dims = {batch_size, seq_length, num_heads};
  const std::vector<int64_t> param_dims = {num_heads};

  OpTester tester("LinearAttentionGate", 1, onnxruntime::kMSDomain);
  if constexpr (std::is_same_v<T, MLFloat16>) {
    tester.AddInput<MLFloat16>("a", dims, ToFloat16(a));
    tester.AddInput<float>("dt_bias", param_dims, dt_bias);
    tester.AddInput<float>("decay_scale", param_dims, decay_scale);
    if (with_beta) {
      tester.AddInput<MLFloat16>("b", dims, ToFloat16(b));
    } else {
      tester.AddOptionalInputEdge<MLFloat16>();
    }
    tester.AddOutput<MLFloat16>("decay", dims, ToFloat16(expected_decay), false, tolerance, tolerance);
    if (with_beta) {
      tester.AddOutput<MLFloat16>("beta", dims, ToFloat16(expected_beta), false, tolerance, tolerance);
    }
  } else {
    tester.AddInput<float>("a", dims, a);
    tester.AddInput<float>("dt_bias", param_dims, dt_bias);
    tester.AddInput<float>("decay_scale", param_dims, decay_scale);
    if (with_beta) {
      tester.AddInput<float>("b", dims, b);
    } else {
      tester.AddOptionalInputEdge<float>();
    }
    tester.AddOutput<float>("decay", dims, expected_decay, false, tolerance, tolerance);
    if (with_beta) {
      tester.AddOutput<float>("beta", dims, expected_beta, false, tolerance, tolerance);
    }
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

template <typename T>
void RunGatedRMSNormTest(int batch_size, int seq_length, int num_heads, int head_dim,
                         float epsilon, float tolerance) {
  const int hidden = num_heads * head_dim;
  const size_t count = static_cast<size_t>(batch_size) * seq_length * hidden;
  const auto x = RandomFloats(count, -3.0f, 3.0f, 21);
  const auto gate = RandomFloats(count, -5.0f, 5.0f, 22);
  const auto scale = RandomFloats(static_cast<size_t>(head_dim), 0.2f, 1.8f, 23);

  std::vector<float> expected(count);
  const size_t num_rows = count / head_dim;
  for (size_t r = 0; r < num_rows; ++r) {
    const size_t base = r * head_dim;
    float sum_sq = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      sum_sq += x[base + i] * x[base + i];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(head_dim) + epsilon);
    for (int i = 0; i < head_dim; ++i) {
      const float z = gate[base + i];
      expected[base + i] = x[base + i] * inv_rms * scale[i] * (z * SigmoidRef(z));
    }
  }

  const std::vector<int64_t> dims = {batch_size, seq_length, hidden};
  const std::vector<int64_t> scale_dims = {head_dim};

  OpTester tester("GatedRMSNorm", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("epsilon", epsilon);
  if constexpr (std::is_same_v<T, MLFloat16>) {
    tester.AddInput<MLFloat16>("X", dims, ToFloat16(x));
    tester.AddInput<MLFloat16>("scale", scale_dims, ToFloat16(scale));
    tester.AddInput<MLFloat16>("gate", dims, ToFloat16(gate));
    tester.AddOutput<MLFloat16>("Y", dims, ToFloat16(expected), false, tolerance, tolerance);
  } else {
    tester.AddInput<float>("X", dims, x);
    tester.AddInput<float>("scale", scale_dims, scale);
    tester.AddInput<float>("gate", dims, gate);
    tester.AddOutput<float>("Y", dims, expected, false, tolerance, tolerance);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(ContribOpLinearAttentionGateTest, Float_Decode) {
  RunLinearAttentionGateTest<float>(1, 1, 32, /*with_beta=*/true, 1e-4f);
}

TEST(ContribOpLinearAttentionGateTest, Float_SpeculativeDecodeTile) {
  RunLinearAttentionGateTest<float>(1, 4, 32, /*with_beta=*/true, 1e-4f);
}

TEST(ContribOpLinearAttentionGateTest, Float_DecayOnly) {
  RunLinearAttentionGateTest<float>(2, 3, 16, /*with_beta=*/false, 1e-4f);
}

TEST(ContribOpLinearAttentionGateTest, Float16_SpeculativeDecodeTile) {
  RunLinearAttentionGateTest<MLFloat16>(1, 4, 32, /*with_beta=*/true, 2e-3f);
}

TEST(ContribOpLinearAttentionGateTest, Float16_Prefill) {
  RunLinearAttentionGateTest<MLFloat16>(2, 37, 32, /*with_beta=*/true, 2e-3f);
}

// num_heads not a multiple of the launch tile, to exercise the bounds check.
TEST(ContribOpLinearAttentionGateTest, Float16_RaggedTail) {
  RunLinearAttentionGateTest<MLFloat16>(1, 5, 7, /*with_beta=*/true, 2e-3f);
}

TEST(ContribOpGatedRMSNormTest, Float_PerHead) {
  RunGatedRMSNormTest<float>(1, 4, 32, 128, 1e-6f, 1e-4f);
}

TEST(ContribOpGatedRMSNormTest, Float_SingleGroup) {
  RunGatedRMSNormTest<float>(2, 3, 1, 512, 1e-5f, 1e-4f);
}

TEST(ContribOpGatedRMSNormTest, Float16_PerHead) {
  RunGatedRMSNormTest<MLFloat16>(1, 4, 32, 128, 1e-6f, 2e-3f);
}

TEST(ContribOpGatedRMSNormTest, Float16_Prefill) {
  RunGatedRMSNormTest<MLFloat16>(2, 17, 32, 128, 1e-6f, 2e-3f);
}

// norm_size below, at, and above the thread-count dispatch boundaries.
TEST(ContribOpGatedRMSNormTest, Float16_SmallNormSize) {
  RunGatedRMSNormTest<MLFloat16>(1, 4, 8, 48, 1e-6f, 2e-3f);
}

TEST(ContribOpGatedRMSNormTest, Float16_LargeNormSize) {
  RunGatedRMSNormTest<MLFloat16>(1, 2, 2, 1536, 1e-6f, 4e-3f);
}

}  // namespace test
}  // namespace onnxruntime
