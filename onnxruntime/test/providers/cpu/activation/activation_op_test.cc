// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activation_op_test.h"
#include "core/providers/cpu/activation/activations.h"

namespace onnxruntime {
namespace test {


TEST_F(ActivationOpTest, Sigmoid) {
  TestActivationOp("Sigmoid",
                         input_values,
                         [](float x) {
                           auto y = 1.f / (1.f + std::exp(-std::abs(x)));  // safe sigmoid
                           y = x > 0 ? y : 1 - y;
                           return y;
                         });
}

TEST_F(ActivationOpTest, HardSigmoid) {
  float alpha = 0.2f;
  float beta = 0.5f;
  TestActivationOp("HardSigmoid",
                         input_values,
                         [alpha, beta](float x) {
                           return std::max(std::min((alpha * x + beta), 1.0f), 0.0f);
                         },
                         {{"alpha", alpha}, {"beta", beta}});
}

TEST_F(ActivationOpTest, Tanh) {
  TestActivationOp("Tanh",
                         input_values,
                         [](float x) { return std::tanh(x); });
}

TEST_F(ActivationOpTest, Relu) {
  TestActivationOp("Relu",
                         input_values,
                         [](float x) { return std::max(x, 0.0f); });
}

TEST_F(ActivationOpTest, Elu) {
  float alpha = 0.1f;
  TestActivationOp("Elu",
                         input_values,
                         [alpha](float x) { return (x >= 0) ? x : alpha * (exp(x) - 1); },
                         {{"alpha", alpha}});
}

TEST_F(ActivationOpTest, LeakyRelu) {
  float alpha = 0.1f;
  TestActivationOp("LeakyRelu",
                         input_values,
                         [alpha](float x) { return (x >= 0) ? x : alpha * x; },
                         {{"alpha", alpha}});
}

TEST_F(ActivationOpTest, ThresholdedRelu) {
  float alpha = 0.1f;
  TestActivationOp(
      "ThresholdedRelu",
      input_values,
      [alpha](float x) { return (x >= alpha) ? x : 0; },
      {{"alpha", alpha}}, true, 10);
}

TEST_F(ActivationOpTest, Selu) {
  static constexpr float alpha = 1.6732f;
  static constexpr float gamma = 1.0507f;

  TestActivationOp("Selu",
                         input_values,
                         [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                         {{"alpha", alpha}, {"gamma", gamma}});
}

TEST_F(ActivationOpTest, Selu_Attributes) {
  static constexpr float alpha = 1.8f;
  static constexpr float gamma = 0.5f;

  TestActivationOp("Selu",
                         input_values,
                         [](float x) { return x <= 0 ? gamma * (alpha * exp(x) - alpha) : gamma * x; },
                         {{"alpha", alpha}, {"gamma", gamma}});
}

TEST_F(ActivationOpTest, PRelu) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, -4.0f, 0.0f, -9.0f};
  std::vector<float> slopes{1.0f, -2.0f, 3.0f, -4.0f};
  std::vector<float> outputs;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i]));

  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, inputs);
  test.AddInput<float>("slope", dims, slopes);
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, PRelu_SingleSlope) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  auto inputs = {1.0f, -4.0f, 0.0f, -9.0f};
  auto slope = 1.5f;
  std::vector<float> outputs;
  for (auto& input : inputs)
    outputs.push_back(formula(input, slope));

  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, inputs);
  test.AddInput<float>("slope", {}, {slope});
  test.AddOutput<float>("Y", dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, PRelu_MultiChannel) {
  OpTester test("PRelu");

  auto formula = [](float x, float slope) { return x < 0 ? slope * x : x; };

  std::vector<float> inputs{1.0f, 2.0f, -4.0f, 3.0f, 0.0f, 5.0f, -9.0f, 8.0f};
  std::vector<float> slopes{1.0f, -2.0f};
  std::vector<float> outputs;
  const int64_t num_images = 2;
  const int64_t num_channels = 2;
  const int64_t num_pixels = 2;
  for (unsigned i = 0; i < inputs.size(); i++)
    outputs.push_back(formula(inputs[i], slopes[i / num_pixels % num_channels]));

  std::vector<int64_t> x_dims{num_images, num_channels, num_pixels};
  std::vector<int64_t> slope_dims{num_channels, 1};
  test.AddInput<float>("X", x_dims, inputs);
  test.AddInput<float>("slope", slope_dims, slopes);
  test.AddOutput<float>("Y", x_dims, outputs);
  test.Run();
}

TEST_F(ActivationOpTest, Softplus) {
  TestActivationOp("Softplus",
                         input_values,
                         [](float x) {
                           if (x > 0)
                             return x + logf(expf(-x) + 1);
                           else
                             return logf(expf(x) + 1);
                         });
}

TEST_F(ActivationOpNoInfTest, Softsign) {
  TestActivationOp(
      "Softsign",
      input_values,
      [](float x) { return x / (1 + std::abs(x)); }, {}, false);  // Disable TensorRT because result mismatches
}

}  // namespace test
}  // namespace onnxruntime
