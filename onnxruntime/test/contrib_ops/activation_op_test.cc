// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

void TestActivationContribOp(const char* szOp, std::vector<float>& input_vals,
                             std::function<float(float)> expected_func,
                             const std::unordered_map<std::string, float> attribs = {},
                             bool is_tensorrt_supported = true,
                             int opset_version = 7,
                             const char* domain = kOnnxDomain) {
  OpTester test(szOp, opset_version, domain);

  for (auto attr : attribs)
    test.AddAttribute(attr.first, attr.second);

  std::vector<int64_t> dims{(int64_t)input_vals.size()};

  std::vector<float> expected_vals;
  for (const auto& iv : input_vals)
    expected_vals.push_back(expected_func(iv));

  test.AddInput<float>("X", dims, input_vals);
  test.AddOutput<float>("Y", dims, expected_vals);

  // Disable TensorRT on unsupported tests
  std::unordered_set<std::string> excluded_providers;
  if (!is_tensorrt_supported) {
    excluded_providers.insert(kTensorrtExecutionProvider);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

std::vector<float> input_values = {
    -1.0f, 0, 1.0f,                                              // normal input values for activation
    100.0f, -100.0f, 1000.0f, -1000.0f,                          // input values that leads to exp() overflow
    FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                        // min, denorm, -denorm
    FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};  // max, -max, inf

TEST(ActivationContribOpTest, ThresholdedRelu_version_1_to_9) {
  float alpha = 0.1f;
  TestActivationContribOp(
      "ThresholdedRelu",
      input_values,
      [alpha](float x) { return (x >= alpha) ? x : 0; },
      {{"alpha", alpha}}, true, 1);
}

TEST(ActivationContribOpTest, ScaledTanh) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationContribOp("ScaledTanh",
                          input_values,
                          [](float x) { return alpha * tanh(beta * x); },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST(ActivationContribOpTest, ParametricSoftplus) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationContribOp("ParametricSoftplus",
                          input_values,
                          [](float x) {
                            float bx = beta * x;
                            if (bx > 0)
                              return alpha * (bx + logf(expf(-bx) + 1));
                            else
                              return alpha * logf(expf(bx) + 1);
                          },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST(ActivationContribOpTest, Gelu) {
  TestActivationContribOp(
      "Gelu",
      input_values,
      [](float x) { return x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))); },
      {}, false, 1, kMSDomain);
}

}  // namespace test
}  // namespace onnxruntime
