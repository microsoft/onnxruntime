// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(GradientOps, SinGrad) {
  OpTester test("SinGrad", 9);

  test.AddInput<float>("dY", {3}, {0, 1, 2});
  test.AddInput<float>("X", {3}, {-1, 0, 1});

  test.AddOutput<float>("dX", {3}, {std::cos(-1.0f) * 0, std::cos(0.0f) * 1, std::cos(1.0f) * 2});
  test.Run();
}

TEST(GradientOps, SigmoidGrad) {
  OpTester test("SigmoidGrad", 9);

  std::function<float(float, float)> sigmoid_grad = [](float dY, float Y) {
    return dY * Y * (1 - Y);
  };

  test.AddInput<float>("dY", {3}, {0, 1, 2});
  test.AddInput<float>("Y", {3}, {-1, 0, 1});

  test.AddOutput<float>("dX", {3}, {sigmoid_grad(0.0f, -1.0f), sigmoid_grad(1.0f, 0.0f), sigmoid_grad(2.0f, 1.0f)});
  test.Run();
}

TEST(GradientOps, SoftmaxGrad) {
  OpTester test("SoftmaxGrad", 9);

  // TODO: Enable test case....
  std::function<float(float, float)> softmax_grad = [](float /*dY*/, float /*Y*/) {
    return 0.0f;
  };

  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("dY", {3}, {0, 0, 0});
  test.AddInput<float>("Y", {3}, {0, 0, 0});

  test.AddOutput<float>("dX", {3}, {softmax_grad(0.0f, 0.0f), softmax_grad(0.0f, 0.0f), softmax_grad(0.0f, 0.0f)});
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime
