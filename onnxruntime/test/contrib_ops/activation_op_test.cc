// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <random>

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

void TestActivationContribOp(const char* szOp, const std::vector<std::vector<float>>& input_vals_vec,
                             std::function<float(float)> expected_func,
                             const std::unordered_map<std::string, float> attribs = {},
                             bool is_tensorrt_supported = true, int opset_version = 7,
                             const char* domain = kOnnxDomain) {
  for (const std::vector<float>& input_vals : input_vals_vec) {
    OpTester test(szOp, opset_version, domain);

    for (auto attr : attribs) test.AddAttribute(attr.first, attr.second);
    std::vector<int64_t> dims{(int64_t)input_vals.size()};

    std::vector<float> expected_vals;
    for (const auto& iv : input_vals) expected_vals.push_back(expected_func(iv));

    test.AddInput<float>("X", dims, input_vals);
    test.AddOutput<float>("Y", dims, expected_vals);

    // Disable TensorRT on unsupported tests
    std::unordered_set<std::string> excluded_providers;
    if (!is_tensorrt_supported) {
      excluded_providers.insert(kTensorrtExecutionProvider);
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }
}
}  // namespace test

class ActivationContribOpTest : public ::testing::Test {
 protected:
  std::vector<std::vector<float>> input_values;  // max, -max, inf

  void SetUp() override {
    float low = -1.0f, high = 1.0f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<std::size_t> batch_size_list = {1, 2, 4, 9, 100000};
    for (auto batch_size : batch_size_list) {
      std::vector<float> vec(batch_size);
      for (size_t i = 0; i != batch_size; ++i) {
        vec[i] = dist(gen);
      }
      input_values.emplace_back(vec);
    }
  }
};

TEST_F(ActivationContribOpTest, ThresholdedRelu_version_1_to_9) {
  float alpha = 0.1f;
  TestActivationContribOp(
      "ThresholdedRelu", input_values, [alpha](float x) { return (x >= alpha) ? x : 0; }, {{"alpha", alpha}}, true, 1);
}

TEST_F(ActivationContribOpTest, ScaledTanh) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationContribOp("ScaledTanh", input_values, [](float x) { return alpha * tanh(beta * x); },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationContribOpTest, ParametricSoftplus) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationContribOp("ParametricSoftplus", input_values,
                          [](float x) {
                            float bx = beta * x;
                            if (bx > 0)
                              return alpha * (bx + logf(expf(-bx) + 1));
                            else
                              return alpha * logf(expf(bx) + 1);
                          },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationContribOpTest, Gelu) {
  TestActivationContribOp(
      "Gelu", input_values, [](float x) { return x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))); }, {},
      false, 1, kMSDomain);
}

}  // namespace onnxruntime
