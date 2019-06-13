// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "test/common/activation_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

std::vector<float> input_vals = {
    -1.0f, 0, 1.0f,                                              // normal input values for activation
    100.0f, -100.0f, 1000.0f, -1000.0f,                          // input values that leads to exp() overflow
    FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,                        // min, denorm, -denorm
    FLT_MAX, -FLT_MAX, std::numeric_limits<float>::infinity()};  // max, -max, inf

std::vector<float> no_inf_input_vals = {
    -1.0f, 0, 1.0f,                        // normal input values for activation
    FLT_MIN, FLT_MIN / 10, -FLT_MIN / 10,  // min, denorm, -denorm
    FLT_MAX, -FLT_MAX};                    // max, -max


TEST(ActivationOpTest, ThresholdedRelu_version_1_to_9) {
  float alpha = 0.1f;
  TestUnaryElementwiseOp("ThresholdedRelu",
                         input_vals,
                         [alpha](float x) { return (x >= alpha) ? x : 0; },
                         {{"alpha", alpha}}, true, 1);
}

TEST(ActivationOpTest, ScaledTanh) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestUnaryElementwiseOp("ScaledTanh",
                         input_vals,
                         [](float x) { return alpha * tanh(beta * x); },
                         {{"alpha", alpha}, {"beta", beta}});
}

TEST(ActivationOpTest, ParametricSoftplus) {
  static constexpr float alpha = 2.0f;
  static constexpr float beta = 1.5f;

  TestUnaryElementwiseOp("ParametricSoftplus",
                         input_vals,
                         [](float x) {
                           float bx = beta * x;
                           if (bx > 0)
                             return alpha * (bx + logf(expf(-bx) + 1));
                           else
                             return alpha * logf(expf(bx) + 1);
                         },
                         {{"alpha", alpha}, {"beta", beta}});
}

}  // namespace test
}  // namespace onnxruntime
