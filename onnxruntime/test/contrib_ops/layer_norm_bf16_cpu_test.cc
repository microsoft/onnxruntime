// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// BFloat16 CPU operator-level tests for all BF16 LayerNorm registrations in this PR:
//   1. Core LayerNormalization (opset 17)
//   2. Contrib LayerNormalization (opset 1–16, kOnnxDomain)
//   3. Contrib SimplifiedLayerNormalization (opset 1, kOnnxDomain) — RMSNorm
//   4. Contrib SkipLayerNormalization (opset 1, kMSDomain)
//   5. Contrib SkipSimplifiedLayerNormalization (opset 1, kMSDomain)
//
// ANTI-FALLBACK DESIGN:
//   Each test exclusively provides the CPU EP via ConfigEp().  If the CPU EP
//   doesn't have a bf16 kernel, session build fails with "no kernel found" —
//   the test cannot pass via a silent Cast-to-float fallback.
//
// TOLERANCE POLICY:
//   BFloat16 (7-bit stored mantissa): 1 bf16 ULP at unit scale ≈ 2^-7 ≈ 0.0078.
//   For bf16-typed outputs (Y), we use a relative tolerance of 2 ULP in bf16
//   (≈ 0.016 at unit scale).  This is applied as an absolute tolerance because
//   the OpTester framework requires absolute tolerance, and the reference values
//   are computed from bf16-round-tripped inputs at unit scale.
//
//   For float-typed stat outputs (Mean, InvStdDev), these MUST hold to f32
//   precision since U=float in the kernel registration.  We use 1e-5 absolute
//   tolerance — tight enough that a kernel that round-trips stats through bf16
//   (losing ~0.4% at unit scale) will fail, but loose enough to accommodate
//   f32 accumulation error.  This is the B5 regression test.

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

// bf16 output tolerance: 2 bf16 ULP at unit scale.
// BFloat16 has a 7-bit stored mantissa; 1 ULP at unit scale ≈ 2^-7 ≈ 0.0078.
// The widen→f32-accumulate→narrow kernel adds ≤1 ULP above the representation
// floor of 0.5 ULP, so 2 ULP total (≈ 0.016) covers both representation and
// accumulation error.
constexpr float kBF16AbsTolerance = 0.016f;

// f32 stat output tolerance.  Mean and InvStdDev are typed as float (U=float)
// and must hold to f32 precision.  1e-5 catches a bf16 round-trip bug (~0.4%
// error at unit scale) while accommodating normal f32 accumulation noise.
constexpr float kF32StatTolerance = 1e-5f;

// Compute LayerNorm reference in f32.
// Returns {output, per-row mean, per-row inv_std_dev}.
struct LayerNormRefResult {
  std::vector<float> output;
  std::vector<float> mean;
  std::vector<float> inv_std_dev;
};

LayerNormRefResult LayerNormRef(const std::vector<float>& x, const std::vector<float>& gamma,
                                const std::vector<float>& bias, int64_t norm_size, float epsilon) {
  const int64_t num_rows = static_cast<int64_t>(x.size()) / norm_size;
  LayerNormRefResult result;
  result.output.resize(x.size());
  result.mean.resize(static_cast<size_t>(num_rows));
  result.inv_std_dev.resize(static_cast<size_t>(num_rows));

  for (int64_t r = 0; r < num_rows; ++r) {
    float row_mean = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      row_mean += x[static_cast<size_t>(r * norm_size + c)];
    }
    row_mean /= static_cast<float>(norm_size);
    float var = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      float d = x[static_cast<size_t>(r * norm_size + c)] - row_mean;
      var += d * d;
    }
    var /= static_cast<float>(norm_size);
    float inv_std = 1.0f / std::sqrt(var + epsilon);
    result.mean[static_cast<size_t>(r)] = row_mean;
    result.inv_std_dev[static_cast<size_t>(r)] = inv_std;
    for (int64_t c = 0; c < norm_size; ++c) {
      auto idx = static_cast<size_t>(r * norm_size + c);
      auto cidx = static_cast<size_t>(c);
      float normed = (x[idx] - row_mean) * inv_std;
      result.output[idx] = normed * gamma[cidx] + (bias.empty() ? 0.0f : bias[cidx]);
    }
  }
  return result;
}

// Compute RMSNorm (SimplifiedLayerNorm) reference in f32.
// Returns {output, per-row inv_rms}.
struct RMSNormRefResult {
  std::vector<float> output;
  std::vector<float> inv_rms;
};

RMSNormRefResult RMSNormRef(const std::vector<float>& x, const std::vector<float>& gamma,
                            int64_t norm_size, float epsilon) {
  const int64_t num_rows = static_cast<int64_t>(x.size()) / norm_size;
  RMSNormRefResult result;
  result.output.resize(x.size());
  result.inv_rms.resize(static_cast<size_t>(num_rows));

  for (int64_t r = 0; r < num_rows; ++r) {
    float sq_mean = 0.0f;
    for (int64_t c = 0; c < norm_size; ++c) {
      float v = x[static_cast<size_t>(r * norm_size + c)];
      sq_mean += v * v;
    }
    sq_mean /= static_cast<float>(norm_size);
    float inv = 1.0f / std::sqrt(sq_mean + epsilon);
    result.inv_rms[static_cast<size_t>(r)] = inv;
    for (int64_t c = 0; c < norm_size; ++c) {
      auto idx = static_cast<size_t>(r * norm_size + c);
      result.output[idx] = x[idx] * inv * gamma[static_cast<size_t>(c)];
    }
  }
  return result;
}

// Round-trip f32 values through bf16 to match the kernel's input precision.
std::vector<float> RoundTripBF16(const std::vector<float>& data) {
  std::vector<float> result(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    result[i] = BFloat16(data[i]).ToFloat();
  }
  return result;
}

// Run an OpTester with CPU-EP only.  If the CPU EP doesn't have the kernel,
// session build fails — no silent fallback to float.
void RunBF16CpuOnly(OpTester& test, float abs_tol, const char* output_name = "output") {
  test.SetOutputAbsErr(output_name, abs_tol);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) {
    GTEST_SKIP() << "CPU EP not available in this build.";
  }
  test.ConfigEp(std::move(cpu))
      .RunWithConfig();
}

// Run with per-output tolerances: bf16 tolerance for Y, f32 tolerance for stats.
void RunBF16CpuOnlyMultiOutput(OpTester& test,
                               const std::vector<std::pair<const char*, float>>& tols) {
  for (auto& [name, tol] : tols) {
    test.SetOutputAbsErr(name, tol);
  }
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
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 3;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f};
  std::vector<float> bias_f32 = {0.0f, 0.0f, 0.0f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

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
  auto ref = LayerNormRef(x_rt, gamma_rt, /*bias=*/{}, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_NonMultipleOfVectorWidth) {
  // NormSize=7 — not a multiple of any SIMD vector width.
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
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_LargerNormSize) {
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
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

// =============================================================================
// LayerNormalization (core ONNX opset 17) — Mean + InvStdDev float outputs
// These stats are typed U=float.  The tolerance here is f32-grade (1e-5) so
// a kernel that round-trips stats through bf16 (~0.4% error) WILL FAIL.
// This is the regression test for the B5 stat-narrowing bug.
// =============================================================================

TEST(LayerNormBFloat16CpuTest, LayerNorm17_MeanInvStdDev_FloatPrecision) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 8;
  constexpr int64_t num_rows = 3;
  std::vector<int64_t> x_dims{num_rows, norm_size};
  std::vector<int64_t> stat_dims{num_rows, 1};

  RandomValueGenerator random{314};
  std::vector<float> x_f32 = random.Uniform<float>(x_dims, -5.0f, 5.0f);
  std::vector<int64_t> gamma_dims{norm_size};
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);
  std::vector<float> bias_f32 = random.Uniform<float>(gamma_dims, -1.0f, 1.0f);

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));
  test.AddOutput<float>("Mean", stat_dims, ref.mean);
  test.AddOutput<float>("InvStdDev", stat_dims, ref.inv_std_dev);

  RunBF16CpuOnlyMultiOutput(test, {{"Y", kBF16AbsTolerance},
                                   {"Mean", kF32StatTolerance},
                                   {"InvStdDev", kF32StatTolerance}});
}

TEST(LayerNormBFloat16CpuTest, LayerNorm17_MeanInvStdDev_LargerNorm) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 128;
  constexpr int64_t num_rows = 4;
  std::vector<int64_t> x_dims{num_rows, norm_size};
  std::vector<int64_t> stat_dims{num_rows, 1};

  RandomValueGenerator random{271};
  std::vector<float> x_f32 = random.Uniform<float>(x_dims, -5.0f, 5.0f);
  std::vector<int64_t> gamma_dims{norm_size};
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);
  std::vector<float> bias_f32 = random.Uniform<float>(gamma_dims, -1.0f, 1.0f);

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 17);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));
  test.AddOutput<float>("Mean", stat_dims, ref.mean);
  test.AddOutput<float>("InvStdDev", stat_dims, ref.inv_std_dev);

  RunBF16CpuOnlyMultiOutput(test, {{"Y", kBF16AbsTolerance},
                                   {"Mean", kF32StatTolerance},
                                   {"InvStdDev", kF32StatTolerance}});
}

// =============================================================================
// Contrib LayerNormalization (kOnnxDomain opset 1–16) — BFloat16 T, float U
// The contrib registration uses VERSIONED_TYPED_KERNEL(1, 16) and constrains
// U=float.  This tests the versioned contrib path distinct from opset 17.
// =============================================================================

TEST(LayerNormBFloat16CpuTest, ContribLayerNorm_Opset1_SmallNormSize) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 4;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.0f, -2.0f, 3.0f, -4.0f,
                              5.0f, 6.0f, -7.0f, 8.0f};
  std::vector<float> gamma_f32 = {1.0f, 0.5f, -1.0f, 2.0f};
  std::vector<float> bias_f32 = {0.1f, -0.2f, 0.3f, -0.1f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  // Contrib LayerNormalization opset 1 (versioned 1–16), kOnnxDomain
  OpTester test("LayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));
  // Contrib schema outputs Mean and InvStdDev as float (U=float)
  std::vector<int64_t> stat_dims{2, 1};
  test.AddOutput<float>("Mean", stat_dims, ref.mean);
  test.AddOutput<float>("InvStdDev", stat_dims, ref.inv_std_dev);

  RunBF16CpuOnlyMultiOutput(test, {{"Y", kBF16AbsTolerance},
                                   {"Mean", kF32StatTolerance},
                                   {"InvStdDev", kF32StatTolerance}});
}

TEST(LayerNormBFloat16CpuTest, ContribLayerNorm_Opset1_LargerNormSize) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 64;
  constexpr int64_t num_rows = 4;
  std::vector<int64_t> x_dims{num_rows, norm_size};
  std::vector<int64_t> stat_dims{num_rows, 1};

  RandomValueGenerator random{161};
  std::vector<float> x_f32 = random.Uniform<float>(x_dims, -5.0f, 5.0f);
  std::vector<int64_t> gamma_dims{norm_size};
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);
  std::vector<float> bias_f32 = random.Uniform<float>(gamma_dims, -1.0f, 1.0f);

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto bias_rt = RoundTripBF16(bias_f32);
  auto ref = LayerNormRef(x_rt, gamma_rt, bias_rt, norm_size, epsilon);

  OpTester test("LayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("B", {norm_size}, ToBFloat16(bias_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));
  test.AddOutput<float>("Mean", stat_dims, ref.mean);
  test.AddOutput<float>("InvStdDev", stat_dims, ref.inv_std_dev);

  RunBF16CpuOnlyMultiOutput(test, {{"Y", kBF16AbsTolerance},
                                   {"Mean", kF32StatTolerance},
                                   {"InvStdDev", kF32StatTolerance}});
}

// =============================================================================
// SimplifiedLayerNormalization (contrib, kOnnxDomain opset 1) — BFloat16 on CPU
// RMSNorm: no mean subtraction, no bias.
// =============================================================================

TEST(LayerNormBFloat16CpuTest, SimplifiedLayerNorm_SmallNormSize) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 3;
  std::vector<int64_t> x_dims{2, norm_size};
  std::vector<float> x_f32 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto ref = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

TEST(LayerNormBFloat16CpuTest, SimplifiedLayerNorm_NonMultipleOfVectorWidth) {
  constexpr float epsilon = 1e-05f;
  constexpr int64_t norm_size = 5;
  std::vector<int64_t> x_dims{3, norm_size};
  std::vector<float> x_f32 = {1.5f, -2.0f, 3.0f, -0.5f, 1.0f,
                              -4.0f, 2.5f, -1.0f, 3.5f, -2.5f,
                              0.1f, 0.2f, -0.3f, 0.4f, -0.5f};
  std::vector<float> gamma_f32 = {0.5f, -1.0f, 2.0f, -0.3f, 1.5f};

  auto x_rt = RoundTripBF16(x_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto ref = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

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
  auto ref = RMSNormRef(x_rt, gamma_rt, norm_size, epsilon);

  OpTester test("SimplifiedLayerNormalization", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<BFloat16>("X", x_dims, ToBFloat16(x_f32));
  test.AddInput<BFloat16>("Scale", {norm_size}, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("Y", x_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance, "Y");
}

// =============================================================================
// SkipLayerNormalization (contrib, kMSDomain opset 1) — BFloat16 on CPU
// output = LayerNorm(input + skip, gamma, beta)
// =============================================================================

TEST(LayerNormBFloat16CpuTest, SkipLayerNorm_Basic) {
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 4;
  std::vector<int64_t> input_dims{1, 2, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  std::vector<float> input_f32 = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> skip_f32 = {0.5f, -0.5f, 0.5f, -0.5f,
                                 -1.0f, 1.0f, -1.0f, 1.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> beta_f32 = {0.0f, 0.0f, 0.0f, 0.0f};

  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto beta_rt = RoundTripBF16(beta_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto ref = LayerNormRef(added, gamma_rt, beta_rt, hidden_size, epsilon);

  OpTester test("SkipLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("beta", gamma_dims, ToBFloat16(beta_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

TEST(LayerNormBFloat16CpuTest, SkipLayerNorm_NoBeta) {
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
  auto ref = LayerNormRef(added, gamma_rt, /*bias=*/{}, hidden_size, epsilon);

  OpTester test("SkipLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOptionalInputEdge<BFloat16>();  // no beta
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

TEST(LayerNormBFloat16CpuTest, SkipLayerNorm_LargerHiddenSize) {
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 128;
  constexpr int64_t num_tokens = 8;
  std::vector<int64_t> input_dims{num_tokens, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  RandomValueGenerator random{99};
  std::vector<float> input_f32 = random.Uniform<float>(input_dims, -5.0f, 5.0f);
  std::vector<float> skip_f32 = random.Uniform<float>(input_dims, -5.0f, 5.0f);
  std::vector<float> gamma_f32 = random.Uniform<float>(gamma_dims, -2.0f, 2.0f);
  std::vector<float> beta_f32 = random.Uniform<float>(gamma_dims, -1.0f, 1.0f);

  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  auto beta_rt = RoundTripBF16(beta_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto ref = LayerNormRef(added, gamma_rt, beta_rt, hidden_size, epsilon);

  OpTester test("SkipLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddInput<BFloat16>("beta", gamma_dims, ToBFloat16(beta_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

// =============================================================================
// SkipSimplifiedLayerNormalization (contrib, kMSDomain opset 1) — BFloat16 on CPU
// output = RMSNorm(input + skip) * gamma
// =============================================================================

TEST(LayerNormBFloat16CpuTest, SkipSimplifiedLayerNorm_Basic) {
  constexpr float epsilon = 1e-12f;
  constexpr int64_t hidden_size = 4;
  std::vector<int64_t> input_dims{1, 2, hidden_size};
  std::vector<int64_t> gamma_dims{hidden_size};

  std::vector<float> input_f32 = {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> skip_f32 = {0.5f, -0.5f, 0.5f, -0.5f,
                                 -1.0f, 1.0f, -1.0f, 1.0f};
  std::vector<float> gamma_f32 = {1.0f, 1.0f, 1.0f, 1.0f};

  auto input_rt = RoundTripBF16(input_f32);
  auto skip_rt = RoundTripBF16(skip_f32);
  auto gamma_rt = RoundTripBF16(gamma_f32);
  std::vector<float> added(input_rt.size());
  for (size_t i = 0; i < added.size(); ++i) {
    added[i] = input_rt[i] + skip_rt[i];
  }
  auto ref = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

TEST(LayerNormBFloat16CpuTest, SkipSimplifiedLayerNorm_NonMultipleNormSize) {
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
  auto ref = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

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
  auto ref = RMSNormRef(added, gamma_rt, hidden_size, epsilon);

  OpTester test("SkipSimplifiedLayerNormalization", 1, onnxruntime::kMSDomain);
  test.AddAttribute("epsilon", epsilon);
  test.AddInput<BFloat16>("input", input_dims, ToBFloat16(input_f32));
  test.AddInput<BFloat16>("skip", input_dims, ToBFloat16(skip_f32));
  test.AddInput<BFloat16>("gamma", gamma_dims, ToBFloat16(gamma_f32));
  test.AddOutput<BFloat16>("output", input_dims, ToBFloat16(ref.output));

  RunBF16CpuOnly(test, kBF16AbsTolerance);
}

}  // namespace test
}  // namespace onnxruntime
