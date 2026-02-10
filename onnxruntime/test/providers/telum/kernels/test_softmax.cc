// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/float16.h"
#include "test/providers/provider_test_utils.h"

#include "../test_utils.h"

namespace onnxruntime {
namespace test {
namespace telum {

class TelumSoftmaxTest : public TelumTestBase {};

TEST_F(TelumSoftmaxTest, Softmax2D_Float) {
  OpTester test("Softmax", 13);

  // Shape [2, 4]
  const std::vector<float> X = {
      1.0f, 2.0f, 3.0f, 4.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
  };

  const auto expected = ComputeSoftmaxLastDimReference(X, /*outer=*/2, /*inner=*/4);

  test.AddInput<float>("X", {2, 4}, X);
  test.AddOutput<float>("Y", {2, 4}, expected);
  test.SetOutputTolerance(1e-5f);

  RunOnTelum(test);
}

TEST_F(TelumSoftmaxTest, Softmax4D_AttentionLike_Float) {
  OpTester test("Softmax", 13);

  // Shape [B=1, H=2, Q=3, K=4] => outer = 1*2*3 = 6 vectors, each length 4.
  const std::vector<float> X = GenerateRandomFloats(1 * 2 * 3 * 4, -2.0f, 2.0f, 123);
  const auto expected = ComputeSoftmaxLastDimReference(X, /*outer=*/6, /*inner=*/4);

  test.AddInput<float>("X", {1, 2, 3, 4}, X);
  test.AddOutput<float>("Y", {1, 2, 3, 4}, expected);
  test.SetOutputTolerance(1e-5f);

  RunOnTelum(test);
}

TEST_F(TelumSoftmaxTest, Softmax2D_Float16) {
  OpTester test("Softmax", 13);

  const std::vector<float> Xf = {
      0.25f, -0.5f, 0.75f, 1.25f,
      2.0f, 0.0f, -1.0f, -2.0f,
  };
  const auto expected_f = ComputeSoftmaxLastDimReference(Xf, /*outer=*/2, /*inner=*/4);

  std::vector<MLFloat16> X;
  std::vector<MLFloat16> expected;
  X.reserve(Xf.size());
  expected.reserve(expected_f.size());
  for (float v : Xf) X.emplace_back(MLFloat16(v));
  for (float v : expected_f) expected.emplace_back(MLFloat16(v));

  test.AddInput<MLFloat16>("X", {2, 4}, X);
  test.AddOutput<MLFloat16>("Y", {2, 4}, expected);
  test.SetOutputTolerance(5e-3f, 5e-3f);

  RunOnTelum(test);
}

TEST_F(TelumSoftmaxTest, Softmax2D_BFloat16) {
  OpTester test("Softmax", 13);

  const std::vector<float> Xf = {
      0.5f, 1.0f, 2.0f, 4.0f,
      -0.25f, 0.25f, -0.5f, 0.5f,
  };
  const auto expected_f = ComputeSoftmaxLastDimReference(Xf, /*outer=*/2, /*inner=*/4);

  std::vector<BFloat16> X;
  std::vector<BFloat16> expected;
  X.reserve(Xf.size());
  expected.reserve(expected_f.size());
  for (float v : Xf) X.emplace_back(BFloat16(v));
  for (float v : expected_f) expected.emplace_back(BFloat16(v));

  test.AddInput<BFloat16>("X", {2, 4}, X);
  test.AddOutput<BFloat16>("Y", {2, 4}, expected);
  test.SetOutputTolerance(5e-3f, 5e-3f);

  RunOnTelum(test);
}

}  // namespace telum
}  // namespace test
}  // namespace onnxruntime

