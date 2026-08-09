// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(UniqueOpTest, Unique_Spec_Example) {
  OpTester test("Unique", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("x", {6}, {2.0f, 1.0f, 1.0f, 3.0f, 4.0f, 3.0f});
  test.AddOutput<float>("uniques", {4}, {2.0f, 1.0f, 3.0f, 4.0f});
  test.AddOutput<int64_t>("idx", {6}, {0, 1, 1, 2, 3, 2});
  test.AddOutput<int64_t>("counts", {4}, {1, 2, 2, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UniqueOpTest, Unique_Complicated_Example) {
  OpTester test("Unique", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("x", {12}, {2.0f, 1.0f, 1.0f, 3.0f, 4.0f, 3.0f, 2.0f, 1.0f, 1.0f, 3.0f, 4.0f, 3.0f});
  test.AddOutput<float>("uniques", {4}, {2.0f, 1.0f, 3.0f, 4.0f});
  test.AddOutput<int64_t>("idx", {12}, {0, 1, 1, 2, 3, 2, 0, 1, 1, 2, 3, 2});
  test.AddOutput<int64_t>("counts", {4}, {2, 4, 4, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UniqueOpTest, Unique_Example_SingleElement) {
  OpTester test("Unique", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("x", {1}, {2.0f});
  test.AddOutput<float>("uniques", {1}, {2.0f});
  test.AddOutput<int64_t>("idx", {1}, {0});
  test.AddOutput<int64_t>("counts", {1}, {1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UniqueOpTest, Unique_AllUniqueElements) {
  OpTester test("Unique", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("x", {2}, {2.0f, 3.0f});
  test.AddOutput<float>("uniques", {2}, {2.0f, 3.0f});
  test.AddOutput<int64_t>("idx", {2}, {0, 1});
  test.AddOutput<int64_t>("counts", {2}, {1, 1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
