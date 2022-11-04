// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/cpu/activation/activation_op_test.h"
#include <random>

using namespace onnxruntime::test;

namespace onnxruntime {

namespace test{

TEST_F(ActivationOpTest, ThresholdedRelu_version_1_to_9) {
  float alpha = 0.1f;
  TestActivationOp<float>(
      "ThresholdedRelu", input_values, [alpha](float x) { return (x >= alpha) ? x : 0; }, {{"alpha", alpha}}, true, 1);
}

TEST_F(ActivationOpTest, ScaledTanh) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationOp<float>("ScaledTanh", input_values, [](float x) { return alpha * tanh(beta * x); },
                          {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationOpTest, ParametricSoftplus) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestActivationOp<float>("ParametricSoftplus", input_values,
                          [](float x) {
                            float bx = beta * x;
                            if (bx > 0)
                              return alpha * (bx + logf(expf(-bx) + 1));
                            else
                              return alpha * logf(expf(bx) + 1);
                          },
                          {{"alpha", alpha}, {"beta", beta}}, false); // Disable TensorRT due to result mismatch
}

TEST_F(ActivationOpTest, Gelu) {
  TestActivationOp<float>(
      "Gelu", input_values, [](float x) { return x * 0.5f * (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2))); }, {},
      false, 1, kMSDomain);
}

TEST_F(ActivationOpTest, QuickGelu) {
  // QuickGelu is not a single activation, some corner values in input_values will not work.
  std::vector<std::vector<float>> quick_gelu_input_values{{-1.0f, 0, 1.0f, 100.0f, -100.0f, 1000.0f, -1000.0f}};

  // Positive alpha.
  {
    float alpha = 1.702f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, false, 1, kMSDomain);
  }

  // Silu = x*sigmoid(x), i.e., alpha = 1.0f.
  {
    float alpha = 1.0f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, false, 1, kMSDomain);
  }

  // Negative alpha.
  {
    float alpha = -1.702f;
    TestActivationOp<float>(
        "QuickGelu", quick_gelu_input_values,
        [alpha](float x) {
          auto tmp = x * alpha;
          auto y = 1.f / (1.f + std::exp(-std::abs(tmp)));  // safe sigmoid
          y = tmp >= 0 ? y : 1 - y;
          return x * y;
        },
        {{"alpha", alpha}}, false, 1, kMSDomain);
  }
}

}  // namespace test
}  // namespace onnxruntime
