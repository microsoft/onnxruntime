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
  {
    OpTester test("Unsqueeze", 12); // opset 1-12 has axes attribute

    test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0, 2});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] 'axes' attribute must not contain any duplicates",
             {kTensorrtExecutionProvider});  //TensorRT failed
  }
  {
    OpTester test("Unsqueeze", -1); // use latest opset with axis input

    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {4}, std::vector<int64_t>{2, 1, 0, 2}, true); //set as initializer to enable shape inference
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] 'axes' attribute must not contain any duplicates",
             {kTensorrtExecutionProvider});  //TensorRT failed
  }
}

TEST(TensorOpTest, Unsqueeze_OutOfRange) {
  {
    OpTester test("Unsqueeze", 12); // opset 1-12 has axes attribute
    test.AddAttribute("axes", std::vector<int64_t>{4});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] values in 'axes' are beyond the bounds of the computed output shape");
  }
  {
    OpTester test("Unsqueeze", -1);  // use latest opset with axis input
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{4}, true); //set as initializer to enable shape inference
    test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] values in 'axes' are beyond the bounds of the computed output shape",
             {kTensorrtExecutionProvider}); //TensorRT expects 'axes' attribute
  }
}

TEST(TensorOpTest, UnsqueezeNegAxis_3) {
  {
    OpTester test("Unsqueeze", 12);  // opset 1-12 has axes attribute
    test.AddAttribute("axes", std::vector<int64_t>{-4, 1, -6});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
  {
    OpTester test("Unsqueeze", 13);  // use latest opset with axis input
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{-4, 1, -6});
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    // TODO: TensorRT, OpenVINO dont support "axes" input in opset 13, re-enable after
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", { kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  }
}

TEST(TensorOpTest, Unsqueeze_1_int32_axes_input) {
  OpTester test("Unsqueeze", 13);

  test.AddInput<int32_t>("input", {2, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1});
  test.AddOutput<int32_t>("output", {2, 1, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(TensorOpTest, Unsqueeze_3_axes_input) {
  OpTester test("Unsqueeze", 13);

  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{2, 1, 0});
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
