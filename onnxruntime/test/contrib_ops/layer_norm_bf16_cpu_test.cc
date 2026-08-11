// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// BFloat16 CPU operator-level tests for LayerNormalization, SimplifiedLayerNormalization,
// and SkipSimplifiedLayerNormalization.
//
// PURPOSE: Prove that the CPU EP can execute these ops with bf16 typed tensors,
// routing to the widen-to-f32-accumulate-narrow kernel (NOT native bf16 math).
//
// ANTI-FALLBACK DESIGN:
//   Each test exclusively provides the CPU EP via ConfigEp() AND sets
//   graph_optimization_level = ORT_DISABLE_ALL to prevent the graph transformer
//   from inserting Cast nodes.  If the bf16 CPU kernel is not registered, the
//   session build fails with "no kernel found" — the test cannot pass via a
//   silent Cast-to-float fallback.
//
//   On upstream main (commit 16b486a2) these tests MUST fail:
//     - Core LayerNormalization (opset 17): no BFloat16 in cpu_execution_provider.cc:1080-1082
//     - Contrib SimplifiedLayerNormalization: no BFloat16 in cpu_contrib_kernels.cc:159-161
//     - Contrib SkipSimplifiedLayerNormalization: no BFloat16 in cpu_contrib_kernels.cc:165-167
//
// TOLERANCE RATIONALE (based on Chew's empirical measurements, 45/45 kernel tests):
//   BFloat16 has a 7-bit stored mantissa.  One bf16 ULP at unit scale ≈ 2^-7 ≈ 0.0078.
//   The bf16 representation-error floor (0.5 ULP) is ~3.9e-3.
//   The widen→f32-accumulate→narrow kernel adds ≤1 bf16 ULP above that floor,
//   even at N=65536.  Total expected error: ≤2 bf16 ULP ≈ 1.6e-2 at unit scale.
//   We set tolerance to 2 bf16 ULP (0.016) — tight enough to catch real regressions,
//   loose enough to accommodate the accumulation error Chew measured.
//   Do NOT reuse the f32 test's 1e-4; bf16 precision is fundamentally lower.

#include <cmath>
#include <vector>

#include "core/graph/constants.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// bf16 absolute tolerance: 2 bf16 ULP at unit scale — see header comment for rationale.
constexpr float kBF16AbsTolerance = 0.016f;

// Compute LayerNorm reference in f32: output[i] = gamma * (x[i] - mean) / sqrt(var + eps) + bias
// This mirrors what the widen→f32→narrow kernel does.
std::vector<float> LayerNormRef(const std::vector<float>& x, const std::vector<float>& gamma,
                                const std::vector<float>& bias, int64_t norm_size, float epsilon) {
  const int64_t num_rows = static_cast<int64_t>(x.size()) / norm_size;
  std::vector<float> output(x.size());
  for (int64_t r = 0; r < num_rows; ++r) {
    float mean = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      mean += x[static_cast<size_t>(r * norm_size + c)];
    }
    mean /= static_cast<float>(norm_size);
    float var = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      float d = x[static_cast<size_t>(r * norm_size + c)] - mean;
      var += d * d;
    }
    var /= static_cast<float>(norm_size);
    float inv_std = 1.0f / std::sqrt(var + epsilon);
    for (int64_t c = 0; c < norm_size; ++c) {
      auto idx = static_cast<size_t>(r * norm_size + c);
      auto cidx = static_cast<size_t>(c);
      float normed = (x[idx] - mean) * inv_std;
      output[idx] = normed * gamma[cidx] + (bias.empty() ? 0.0f : bias[cidx]);
    }
  }
  return output;
}

// Compute RMSNorm (SimplifiedLayerNorm) reference in f32:
//   output[i] = gamma * x[i] / sqrt(mean(x^2) + eps)
std::vector<float> RMSNormRef(const std::vector<float>& x, const std::vector<float>& gamma,
                              int64_t norm_size, float epsilon) {
  const int64_t num_rows = static_cast<int64_t>(x.size()) / norm_size;
  std::vector<float> output(x.size());
  for (int64_t r = 0; r < num_rows; ++r) {
    float sq_mean = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      float v = x[static_cast<size_t>(r * norm_size + c)];
      sq_mean += v * v;
    }
    sq_mean /= static_cast<float>(norm_size);
    float inv_rms = 1.0f / std::sqrt(sq_mean + epsilon);
    for (int64_t c = 0; c < norm_size; ++c) {
      auto idx = static_cast<size_t>(r * norm_size + c);
      output[idx] = x[idx] * inv_rms * gamma[static_cast<size_t>(c)];
    }
  }
  return output;
}

// Truncate f32 values to bf16 and back to simulate bf16 representation error on inputs.
std::vector<float> RoundTripBF16(const std::vector<float>& data) {
  std::vector<float> result(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    result[i] = BFloat16(data[i]).ToFloat();
  }
  return result;
}

// Helper: configure an OpTester for CPU-EP-only.
// This is the anti-fallback mechanism: if the CPU EP doesn't have a bf16 kernel,
// the node cannot be placed on any EP and session build fails.  With a single EP
// there is no alternative provider to fall back to, and no Cast insertion can
// satisfy an unsupported type.
void RunBF16CpuOnly(OpTester& test, float abs_tol, const char* output_name = "output") {
  test.SetOutputAbsErr(output_name, abs_tol);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) {
    GTEST_SKIP() << "CPU EP not available in this build.";
  }
  test.ConfigEp(std::move(cpu))
      .RunWithConfig();
}

}  // anonymous namespace

// =============================================================================
// LayerNormalization (core ONNX opset 17) — BFloat16 on CPU
// =============================================================================

TEST(LayerNormBFloat16CpuTest, LayerNorm17_SmallNormSize) {
  // NormSize=3, 2 rows — basic functional test with bias.
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 3;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f};
  std::vector<float> bias_f32 = {0.0f, 0.0f, 0.0f};

  // Round-trip through bf16 so reference uses the same precision as inputs
  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto expected = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_NoBias) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 4;
  std::vector<int64_t> x_dims{3, norm_size};
  std::vector<float> x_f32 = {-1.0f, 2.0f, -3.0f, 4.0f,
                              5.0f, -6.0f, 7.0f, -8.0f,
                              0.5f, 1.5f, -2.5f, 3.5f};
  std::vector<float> gamma_f32 = {0.5f, -1.0f, 1.5f, -0.5f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto expected = LayerNormRef(x_rt, gamma_rt, /*bias=*/{}, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  // No bias input
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_NonMultipleOfVectorWidth) {
  // NormSize=7 — not a multiple of any SIMD vector width (SSE=4, AVX=8, AVX-512=16).
  // Tests scalar/remainder handling.
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 7;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.2f, -0.5f, 3.1f, -2.8f, 0.7f, -1.1f, 4.0f,
                              -3.0f, 2.5f, -0.3f, 1.8f, -4.2f, 0.1f, -0.9f};
  std::vector<float> gamma_f32 = {1.0f, -0.5f, 2.0f, -1.0f, 0.3f, -2.0f, 1.5f};
  std::vector<float> bias_f32 = {0.1f, -0.2f, 0.3f, -0.1f, 0.0f, 0.5f, -0.3f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto expected = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_LargerNormSize) {
  // NormSize=128 — representative of a small transformer hidden dimension.
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 128;
  constexpr int64_t num_rows = 4;
  std::vector<int64_t> x_dims{num_rows, norm_size};

  RandomValueGenerator random{42};
  std::vector<float> x_f32 = random.Uniform<float>(x_dims, -5.0f, 5.0f);
  std::vector<int64_t> gamma_dims{norm_size};
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);
  std::vector<float> bias_f32 = random.Uniform<float>(gamma_dims, -1.0f, 1.0f);

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto expected = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

// =============================================================================
// SimplifiedLayerNormalization (contrib, kOnnxDomain opset 1) — BFloat16 on CPU
// This is RMSNorm — no mean subtraction, no bias.
// =============================================================================

TEST(LayerNormBFloat16CpuTest, SimplifiedLayerNorm_SmallNormSize) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 3;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto expected = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, SimplifiedLayerNorm_NonMultipleOfVectorWidth) {
  // NormSize=5 — odd, not a power of 2.
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 5;
  std::vector<int64_t> x_dims{3, norm_size};
  std::vector<float> x_f32 = {1.5f, -2.0f, 3.0f, -0.5f, 1.0f,
                              -4.0f, 2.5f, -1.0f, 3.5f, -2.5f,
                              0.1f, 0.2f, -0.3f, 0.4f, -0.5f};
  std::vector<float> gamma_f32 = {0.5f, -1.0f, 2.0f, -0.3f, 1.5f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto expected = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, SimplifiedLayerNorm_LargerNormSize) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 256;
  constexpr int64_t num_rows = 4;
  std::vector<int64_t> x_dims{num_rows, norm_size};

  RandomValueGenerator random{123};
  std::vector<float> x_f32 = random.Uniform<float>(x_dims, -5.0f, 5.0f);
  std::vector<int64_t> gamma_dims{norm_size};
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto expected = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

// =============================================================================
// SkipSimplifiedLayerNormalization (contrib, kMSDomain opset 1) — BFloat16 on CPU
// output = RMSNorm(input + skip) * gamma
// =============================================================================

TEST(LayerNormBFloat16CpuTest, SkipSimplifiedLayerNorm_Basic) {
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 4;
  constexpr int64_t batch_size = 1;
  constexpr int64_t seq_len = 2;
  std::vector<int64_t> input_dims{batch_size, seq_len, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  std::vector<float> input_f32 = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> skip_f32 = {0.5f, -0.5f, 0.5f, -0.5f,
                                 -1.0f, 1.0f, -1.0f, 1.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f, 1.0f};

  // Reference: RMSNorm(input + skip) * gamma
  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto expected = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

TEST(LayerNormBFloat16CpuTest, SkipSimplifiedLayerNorm_NonMultipleNormSize) {
  // hidden_size=5 — remainder path
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 5;
  std::vector<int64_t> input_dims{2, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  std::vector<float> input_f32 = {1.0f, -2.0f, 3.0f, -1.5f, 0.5f,
                                  -3.0f, 2.0f, -0.5f, 4.0f, -1.0f};
  std::vector<float> skip_f32 = {-0.5f, 1.0f, -1.5f, 2.0f, -0.3f,
                                 1.5f, -1.0f, 0.8f, -2.0f, 3.0f};
  std::vector<float> gamma_f32 = {0.5f, -1.0f, 2.0f, -0.5f, 1.5f};

  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto expected = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

TEST(LayerNormBFloat16CpuTest, SkipSimplifiedLayerNorm_LargerHiddenSize) {
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t num_tokens = 8;
  std::vector<int64_t> input_dims{num_tokens, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  RandomValueGenerator random{77};
  std::vector<float> input_f32 = random.Uniform<float>(input_dims, -5.0f, 5.0f);
  std::vector<float> skip_f32 = random.Uniform<float>(input_dims, -5.0f, 5.0f);
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);

  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto expected = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(expected));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

}  // namespace test
}  // namespace onnxruntime
