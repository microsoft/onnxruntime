// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(Dropout, Opset7) {
  OpTester test("Dropout", 7, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run();
}

TEST(Dropout, Opset10) {
  OpTester test("Dropout", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.Run();
}

TEST(Dropout, WithOptionalOutput) {
  OpTester test("Dropout", 10, kOnnxDomain);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 3.0f, 5.0f});
  test.AddOutput<bool>("mask", dims, {false, false, false, false});
  // The NGraph execution provider doesn't seem to support 'Dropout' with optional mask output
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
