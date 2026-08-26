// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the two fused linear-attention gate ops: LinearAttentionGate and GatedRMSNorm.
// Both replace float32 elementwise chains that an exporter emits around the LinearAttention op,
// so the references here are the same float32 formulas ORT's Softplus/Sigmoid/RMSNorm kernels use.
// The tests run against every EP (CPU, CUDA, WebGPU) that has these ops registered.

#include <cmath>
#include <random>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

std::vector<std::unique_ptr<IExecutionProvider>> AvailableGatedOpExecutionProviders() {
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCpuExecutionProvider());
  if (auto cuda_ep = DefaultCudaExecutionProvider()) {
    eps.push_back(std::move(cuda_ep));
  }
  if (auto webgpu_ep = DefaultWebGpuExecutionProvider()) {
    eps.push_back(std::move(webgpu_ep));
  }
  return eps;
}

// BFloat16 is currently CUDA-only; other types run on every available EP.
template <typename T>
std::vector<std::unique_ptr<IExecutionProvider>> ExecutionProvidersForType() {
  if constexpr (std::is_same_v<T, BFloat16>) {
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    if (auto cuda_ep = DefaultCudaExecutionProvider()) {
      eps.push_back(std::move(cuda_ep));
    }
    return eps;
  } else {
    return AvailableGatedOpExecutionProviders();
  }
}

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
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

template <typename T>
void RunLinearAttentionGateTest(int batch_size, int seq_length, int num_heads, bool with_beta,
                                float tolerance) {
  auto execution_providers = ExecutionProvidersForType<T>();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }

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

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("LinearAttentionGate", 1, onnxruntime::kMSDomain);
    tester.AddInput<T>("a", dims, ToTensorType<T>(a));
    tester.AddInput<float>("dt_bias", param_dims, dt_bias);
    tester.AddInput<float>("decay_scale", param_dims, decay_scale);
    if (with_beta) {
      tester.AddInput<T>("b", dims, ToTensorType<T>(b));
    } else {
      tester.AddOptionalInputEdge<T>();
    }
    tester.AddOutput<T>("decay", dims, ToTensorType<T>(expected_decay), false, tolerance, tolerance);
    if (with_beta) {
      tester.AddOutput<T>("beta", dims, ToTensorType<T>(expected_beta), false, tolerance, tolerance);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
}

template <typename T>
void RunGatedRMSNormTest(int batch_size, int seq_length, int num_heads, int head_dim,
                         float epsilon, float tolerance, const std::string& activation = "silu") {
  auto execution_providers = ExecutionProvidersForType<T>();
  if (execution_providers.empty()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }

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
      const float activated = activation == "sigmoid" ? SigmoidRef(z) : (z * SigmoidRef(z));
      expected[base + i] = x[base + i] * inv_rms * scale[i] * activated;
    }
  }

  const std::vector<int64_t> dims = {batch_size, seq_length, hidden};
  const std::vector<int64_t> scale_dims = {head_dim};

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("GatedRMSNorm", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<float>("epsilon", epsilon);
    tester.AddAttribute<std::string>("activation", activation);
    tester.AddInput<T>("X", dims, ToTensorType<T>(x));
    tester.AddInput<T>("scale", scale_dims, ToTensorType<T>(scale));
    tester.AddInput<T>("gate", dims, ToTensorType<T>(gate));
    tester.AddOutput<T>("Y", dims, ToTensorType<T>(expected), false, tolerance, tolerance);

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  }
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

TEST(ContribOpLinearAttentionGateTest, BFloat16_SpeculativeDecodeTile) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunLinearAttentionGateTest<BFloat16>(1, 4, 32, /*with_beta=*/true, 2e-2f);
}

// Requesting beta without b must be rejected by shape inference, not at execution time.
TEST(ContribOpLinearAttentionGateTest, BetaWithoutB_FailsShapeInference) {
  auto execution_providers = AvailableGatedOpExecutionProviders();

  constexpr int kNumHeads = 8;
  const std::vector<int64_t> dims = {1, 2, kNumHeads};
  const std::vector<int64_t> param_dims = {kNumHeads};
  const std::vector<float> values(static_cast<size_t>(2 * kNumHeads), 0.5f);
  const std::vector<float> params(kNumHeads, 0.5f);

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("LinearAttentionGate", 1, onnxruntime::kMSDomain);
    tester.AddInput<float>("a", dims, values);
    tester.AddInput<float>("dt_bias", param_dims, params);
    tester.AddInput<float>("decay_scale", param_dims, params);
    tester.AddOptionalInputEdge<float>();
    tester.AddOutput<float>("decay", dims, values);
    tester.AddOutput<float>("beta", dims, values);

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure,
               "The b input is required when the beta output is requested",
               {}, nullptr, &providers);
  }
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

TEST(ContribOpGatedRMSNormTest, BFloat16_PerHead) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunGatedRMSNormTest<BFloat16>(1, 4, 32, 128, 1e-6f, 2e-2f);
}

TEST(ContribOpGatedRMSNormTest, Float_SigmoidActivation) {
  RunGatedRMSNormTest<float>(1, 4, 32, 128, 1e-6f, 1e-4f, "sigmoid");
}

TEST(ContribOpGatedRMSNormTest, Float16_SigmoidActivation) {
  RunGatedRMSNormTest<MLFloat16>(2, 17, 32, 128, 1e-6f, 2e-3f, "sigmoid");
}

TEST(ContribOpGatedRMSNormTest, BFloat16_SigmoidActivation) {
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "bfloat16 requires compute capability 8.0 or later";
  }
  RunGatedRMSNormTest<BFloat16>(1, 4, 32, 128, 1e-6f, 2e-2f, "sigmoid");
}

// Invalid activation strings must be rejected at kernel construction, not silently accepted.
TEST(ContribOpGatedRMSNormTest, InvalidActivation_Fails) {
  auto execution_providers = AvailableGatedOpExecutionProviders();

  const std::vector<int64_t> dims = {1, 2, 8};
  const std::vector<int64_t> scale_dims = {8};
  const std::vector<float> values(16, 0.5f);
  const std::vector<float> scale(8, 1.0f);

  for (auto& ep : execution_providers) {
    SCOPED_TRACE("EP: " + ep->Type());
    OpTester tester("GatedRMSNorm", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<std::string>("activation", "relu");
    tester.AddInput<float>("X", dims, values);
    tester.AddInput<float>("scale", scale_dims, scale);
    tester.AddInput<float>("gate", dims, values);
    tester.AddOutput<float>("Y", dims, values);

    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &providers);
  }
}

}  // namespace test
}  // namespace onnxruntime
