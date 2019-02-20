// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ElementWiseOpGrad, SinGrad) {
  OpTester test("SinGrad", 9);

  test.AddInput<float>("dY", {3}, {0, 1, 2});
  test.AddInput<float>("X", {3}, {-1, 0, 1});

  test.AddOutput<float>("dX", {3}, {std::cosf(-1.0) * 0, std::cosf(0) * 1, std::cosf(1) * 2});
  test.Run();
}  // namespace test
}  // namespace test
}  // namespace onnxruntime
