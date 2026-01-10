// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/data_types.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(RoundTest, SimpleTestFloat) {
  OpTester test("Round", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f});
  test.AddOutput<float>("y", {5}, {1.0f, 2.0f, 2.0f, 2.0f, -4.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(RoundTest, SimpleTestDouble) {
  OpTester test("Round", 11, onnxruntime::kOnnxDomain);
  test.AddInput<double>("x", {5}, {0.9, 2.5, 2.3, 1.5, -4.5});
  test.AddOutput<double>("y", {5}, {1.0, 2.0, 2.0, 2.0, -4.0});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(RoundTest, SimpleTestFloat16) {
  OpTester test("Round", 11, onnxruntime::kOnnxDomain);
  test.AddInput<MLFloat16>("x", {5}, {MLFloat16(0.9f), MLFloat16(2.5f), MLFloat16(2.3f), MLFloat16(1.5f), MLFloat16(-4.5f)});
  test.AddOutput<MLFloat16>("y", {5}, {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(-4.0f)});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime