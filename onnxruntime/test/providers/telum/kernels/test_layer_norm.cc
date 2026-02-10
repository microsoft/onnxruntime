// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/float16.h"
#include "test/providers/provider_test_utils.h"

#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

class TelumLayerNormTest : public TelumTestBase {};

TEST_F(TelumLayerNormTest, LayerNorm_AxisNotLast_ExpectFailure) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(0));
  test.AddAttribute("epsilon", 1e-5f);

  const std::vector<float> X = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.0f, 1.0f,
  };
  const std::vector<float> Scale = {1.0f, 1.0f, 1.0f};
  const std::vector<float> Bias = {0.0f, 0.0f, 0.0f};

  test.AddInput<float>("X", {2, 3}, X);
  test.AddInput<float>("Scale", {3}, Scale);
  test.AddInput<float>("Bias", {3}, Bias);

  // Outputs are unused for failure expectation, but must be present for model construction.
  test.AddOutput<float>("Y", {2, 3}, X);
  test.AddOptionalOutputEdge<float>();  // Mean
  test.AddOptionalOutputEdge<float>();  // InvStdDev

  RunOnTelumExpectFailure(test);
}

TEST_F(TelumLayerNormTest, LayerNorm2D_Float_WithBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("epsilon", 1e-5f);

  // X shape [2, 3] => N=2, C=3
  const std::vector<float> X = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.0f, 1.0f,
  };
  const std::vector<float> Scale = {1.0f, 0.5f, 2.0f};
  const std::vector<float> Bias = {0.1f, -0.2f, 0.3f};

  const auto ref = ComputeLayerNormLastDimReference(X, Scale, Bias, /*N=*/2, /*C=*/3, /*epsilon=*/1e-5f);

  test.AddInput<float>("X", {2, 3}, X);
  test.AddInput<float>("Scale", {3}, Scale);
  test.AddInput<float>("Bias", {3}, Bias);

  test.AddOutput<float>("Y", {2, 3}, ref.Y);
  test.AddOutput<float>("Mean", {2, 1}, ref.Mean);
  test.AddOutput<float>("InvStdDev", {2, 1}, ref.InvStd);
  test.SetOutputTolerance(1e-5f);

  RunOnTelum(test);
}

TEST_F(TelumLayerNormTest, LayerNorm2D_Float_OmitMeanInvStd) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("epsilon", 1e-5f);

  // X shape [2, 3] => N=2, C=3
  const std::vector<float> X = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.0f, 1.0f,
  };
  const std::vector<float> Scale = {1.0f, 0.5f, 2.0f};
  const std::vector<float> Bias = {0.1f, -0.2f, 0.3f};

  const auto ref = ComputeLayerNormLastDimReference(X, Scale, Bias, /*N=*/2, /*C=*/3, /*epsilon=*/1e-5f);

  test.AddInput<float>("X", {2, 3}, X);
  test.AddInput<float>("Scale", {3}, Scale);
  test.AddInput<float>("Bias", {3}, Bias);

  test.AddOutput<float>("Y", {2, 3}, ref.Y);
  test.AddOptionalOutputEdge<float>();  // Mean omitted
  test.AddOptionalOutputEdge<float>();  // InvStdDev omitted
  test.SetOutputTolerance(1e-5f);

  RunOnTelum(test);
}

TEST_F(TelumLayerNormTest, LayerNorm2D_Float_OmitMean_KeepInvStd) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("epsilon", 1e-5f);

  const std::vector<float> X = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.0f, 1.0f,
  };
  const std::vector<float> Scale = {1.0f, 1.0f, 1.0f};
  const std::vector<float> Bias;  // no bias

  const auto ref = ComputeLayerNormLastDimReference(X, Scale, Bias, /*N=*/2, /*C=*/3, /*epsilon=*/1e-5f);

  test.AddInput<float>("X", {2, 3}, X);
  test.AddInput<float>("Scale", {3}, Scale);
  test.AddOptionalInputEdge<float>();  // Bias omitted

  test.AddOutput<float>("Y", {2, 3}, ref.Y);
  test.AddOptionalOutputEdge<float>();  // Mean omitted
  test.AddOutput<float>("InvStdDev", {2, 1}, ref.InvStd);
  test.SetOutputTolerance(1e-5f);

  RunOnTelum(test);
}

TEST_F(TelumLayerNormTest, LayerNorm3D_Float16_NoBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("epsilon", 1e-5f);

  // X shape [1, 2, 4] => N=2, C=4
  const std::vector<float> Xf = GenerateRandomFloats(1 * 2 * 4, -1.0f, 1.0f, 777);
  const std::vector<float> Scalef = {1.0f, 1.5f, 0.25f, 2.0f};
  const std::vector<float> Biasf;  // no bias

  const auto ref = ComputeLayerNormLastDimReference(Xf, Scalef, Biasf, /*N=*/2, /*C=*/4, /*epsilon=*/1e-5f);

  std::vector<MLFloat16> X;
  std::vector<MLFloat16> Scale;
  std::vector<MLFloat16> Y_expected;
  X.reserve(Xf.size());
  Scale.reserve(Scalef.size());
  Y_expected.reserve(ref.Y.size());
  for (float v : Xf) X.emplace_back(MLFloat16(v));
  for (float v : Scalef) Scale.emplace_back(MLFloat16(v));
  for (float v : ref.Y) Y_expected.emplace_back(MLFloat16(v));

  test.AddInput<MLFloat16>("X", {1, 2, 4}, X);
  test.AddInput<MLFloat16>("Scale", {4}, Scale);
  test.AddOptionalInputEdge<MLFloat16>();  // Bias omitted

  test.AddOutput<MLFloat16>("Y", {1, 2, 4}, Y_expected);
  test.AddOutput<float>("Mean", {1, 2, 1}, ref.Mean);
  test.AddOutput<float>("InvStdDev", {1, 2, 1}, ref.InvStd);
  test.SetOutputTolerance(5e-3f, 5e-3f);

  RunOnTelum(test);
}

TEST_F(TelumLayerNormTest, LayerNorm2D_BFloat16_WithBias) {
  OpTester test("LayerNormalization", 17);
  test.AddAttribute("axis", static_cast<int64_t>(-1));
  test.AddAttribute("epsilon", 1e-5f);

  // X shape [2, 4] => N=2, C=4
  const std::vector<float> Xf = GenerateRandomFloats(2 * 4, -2.0f, 2.0f, 999);
  const std::vector<float> Scalef = {0.5f, 1.0f, 2.0f, 1.5f};
  const std::vector<float> Biasf = {0.0f, 0.1f, -0.2f, 0.3f};

  const auto ref = ComputeLayerNormLastDimReference(Xf, Scalef, Biasf, /*N=*/2, /*C=*/4, /*epsilon=*/1e-5f);

  std::vector<BFloat16> X;
  std::vector<BFloat16> Scale;
  std::vector<BFloat16> Bias;
  std::vector<BFloat16> Y_expected;
  X.reserve(Xf.size());
  Scale.reserve(Scalef.size());
  Bias.reserve(Biasf.size());
  Y_expected.reserve(ref.Y.size());
  for (float v : Xf) X.emplace_back(BFloat16(v));
  for (float v : Scalef) Scale.emplace_back(BFloat16(v));
  for (float v : Biasf) Bias.emplace_back(BFloat16(v));
  for (float v : ref.Y) Y_expected.emplace_back(BFloat16(v));

  test.AddInput<BFloat16>("X", {2, 4}, X);
  test.AddInput<BFloat16>("Scale", {4}, Scale);
  test.AddInput<BFloat16>("Bias", {4}, Bias);

  test.AddOutput<BFloat16>("Y", {2, 4}, Y_expected);
  test.AddOutput<float>("Mean", {2, 1}, ref.Mean);
  test.AddOutput<float>("InvStdDev", {2, 1}, ref.InvStd);
  test.SetOutputTolerance(5e-3f, 5e-3f);

  RunOnTelum(test);
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime
