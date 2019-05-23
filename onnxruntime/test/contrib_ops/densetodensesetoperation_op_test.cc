// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(DenseToDenseSetOperation, DenseSet_Int32_2_2_4) {
  OpTester test("DenseToDenseSetOperation", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("x", {2, 2, 4}, {1, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6, 9, 0});
  test.AddInput<int32_t>("y", {2, 2, 4}, {1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 6, 7, 8});
  test.AddOutput<int32_t>("z", {2, 2, 2}, {1, 0, 0, 0, 4, 0, 5, 6});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DenseToDenseSetOperation, DenseSet_Int64_2_2_4) {
  OpTester test("DenseToDenseSetOperation", 1, onnxruntime::kMSDomain);
  test.AddInput<int64_t>("x", {2, 2, 4}, {1, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6, 9, 0});
  test.AddInput<int64_t>("y", {2, 2, 4}, {1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 6, 7, 8});
  test.AddOutput<int64_t>("z", {2, 2, 2}, {1, 0, 0, 0, 4, 0, 5, 6});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DenseToDenseSetOperation, DenseSet_Int64_2_2_4_default_value) {
  OpTester test("DenseToDenseSetOperation", 1, onnxruntime::kMSDomain);
  test.AddAttribute("default_value", (int64_t)-1);
  test.AddInput<int64_t>("x", {2, 2, 4}, {1, 2, -1, -1, 3, -1, -1, -1, 4, -1, -1, -1, 5, 6, 9, -1});
  test.AddInput<int64_t>("y", {2, 2, 4}, {1, -1, -1, -1, -1, -1, -1, -1, 4, -1, -1, -1, 5, 6, 7, 8});
  test.AddOutput<int64_t>("z", {2, 2, 2}, {1, -1, -1, -1, 4, -1, 5, 6});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DenseToDenseSetOperation, DenseSet_Int64_2_2_4_empty_result) {
  OpTester test("DenseToDenseSetOperation", 1, onnxruntime::kMSDomain);
  test.AddInput<int64_t>("x", {2, 2, 4}, {1, 2, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 6, 9, 0});
  test.AddInput<int64_t>("y", {2, 2, 4}, {11, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 15, 16, 17, 18});
  test.AddOutput<int64_t>("z", {2, 2, 0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(DenseToDenseSetOperation, DenseSet_Int64_2_2_4_empty_inputs) {
  OpTester test("DenseToDenseSetOperation", 1, onnxruntime::kMSDomain);
  test.AddInput<int64_t>("x", {2, 2, 0}, {});
  test.AddInput<int64_t>("y", {2, 2, 4}, {11, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 15, 16, 17, 18});
  test.AddOutput<int64_t>("z", {2, 2, 0}, {});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}


}  // namespace test
}  // namespace onnxruntime
