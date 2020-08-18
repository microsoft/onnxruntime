// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on the tests because of SegFault errors in the parser

TEST(TensorOpTest, Unsqueeze_1) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Unsqueeze_1_int32) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddInput<int32_t>("input", {2, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  test.AddOutput<int32_t>("output", {2, 1, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Unsqueeze_2) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{0, 4});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 2, 3, 4, 1}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_3) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_Duplicate) {
  OpTester test("Unsqueeze", -1);  // use latest opset for shape inference errors

  test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0, 2});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "[ShapeInferenceError] 'axes' attribute must not contain any duplicates",
           {kTensorrtExecutionProvider});  //TensorRT failed
}

TEST(TensorOpTest, Unsqueeze_OutOfRange) {
  OpTester test("Unsqueeze", -1);  // use latest opset for shape inference errors

  test.AddAttribute("axes", std::vector<int64_t>{4});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "[ShapeInferenceError] values in 'axes' are beyond the bounds of the computed output shape");
}

TEST(TensorOpTest, UnsqueezeNegAxis_3) {
  OpTester test("Unsqueeze", 12);  // use latest opset for shape inference errors

  test.AddAttribute("axes", std::vector<int64_t>{-4, 1, -6});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  // nGraph and TensorRT does not support negative axis.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
