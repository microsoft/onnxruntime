// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include <math.h>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

void TestGradientOpWithTwoInputs(const char* szOp,
                                 std::vector<float>& dY,
                                 std::vector<float>& X,
                                 std::function<float(float, float)> expected_func,
                                 const std::unordered_map<std::string, float> attrs = {},
                                 int opset_version = 7, const char* domain = kOnnxDomain) {
  OpTester test(szOp, opset_version, domain);
  ORT_ENFORCE(dY.size() == X.size());
  for (auto attr : attrs)
    test.AddAttribute(attr.first, attr.second);

  std::vector<int64_t> dims{(int64_t)X.size()};

  std::vector<float> expected_vals;
  for (size_t i = 0; i < X.size(); i++) {
    expected_vals.push_back(expected_func(dY[i], X[i]));
  }

  test.AddInput<float>("dY", dims, dY);
  test.AddInput<float>("X", dims, X);
  test.AddOutput<float>("dX", dims, expected_vals);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {});
}

TEST(GeluGradTest, Basic) {
  std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};

  std::vector<float> dY(7, 1.0f);
  TestGradientOpWithTwoInputs(
      "GeluGrad",
      dY,
      x_vals,
      [](float dy, float x) {
        return dy * (0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))) +
                     x * std::exp(-0.5f * x * x) * static_cast<float>(M_2_SQRTPI) * static_cast<float>(M_SQRT1_2) * 0.5f);
      },
      {}, 1, kMSDomain);
}

TEST(FastGeluGradTest, Basic) {
  std::vector<float> x_vals = {-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f};
  std::vector<float> dY(7, 1.0f);

  const float kAlpha = static_cast<float>(M_2_SQRTPI * M_SQRT1_2);
  const float kGamma = 0.044715f;
  const float kBeta = kAlpha * kGamma * 3.0f;

  TestGradientOpWithTwoInputs(
      "FastGeluGrad",
      dY,
      x_vals,
      [&](float dy, float x) {
        float x_cube = x * x * x;
        float tanh_value = std::tanh(kAlpha * (x + kGamma * x_cube));
        float sech_sqr_value = 1 - tanh_value * tanh_value;
        float result = dy * 0.5f * (tanh_value + (sech_sqr_value * (kAlpha * x + kBeta * x_cube)) + 1.0f);
        return result;
      },
      {}, 1, kMSDomain);
}

}  // namespace test
}  // namespace onnxruntime
