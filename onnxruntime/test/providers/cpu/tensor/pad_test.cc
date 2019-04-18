// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because only constant mode and value 0 is supported for "Pad" node

TEST(TensorOpTest, Pad_Spec_Example) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{0, 2, 0, 0});
  test.AddAttribute("value", 0.0f);
  test.AddInput<float>("data", {3, 2}, {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f});
  test.AddOutput<float>("output", {3, 4}, {0.0f, 0.0f, 1.0f, 1.2f, 0.0f, 0.0f, 2.3f, 3.4f, 0.0f, 0.0f, 4.5f, 5.7f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Constant_1D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {5}, {1234.0f, 1.0f, 2.0f, 1234.0f, 1234.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Constant_1D_Zero) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {2}, {1.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Constant_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 1, 2});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2, 2},
                       {11.0f, 21.0f,
                        12.0f, 22.0f});
  test.AddOutput<float>("output", {4, 6},
                        {1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 11.0f, 21.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 12.0f, 22.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Constant_2D_negative) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 1, -1});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {4, 4},
                        {1234.0f, 1234.0f, 1234.0f, 1234.0f,
                         1234.0f, 1234.0f, 11.0f, 21.0f,
                         1234.0f, 1234.0f, 12.0f, 22.0f,
                         1234.0f, 1234.0f, 1234.0f, 1234.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_3D_complex) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 0, 0, -1, 0, 0});
  test.AddAttribute("value", 0.0f);
  test.AddInput<float>("data", {2, 2, 2},
                       {111.0f, 112.0f,
                        121.0f, 122.0f,

                        211.0f, 212.0f,
                        221.0f, 222.0f});
  test.AddOutput<float>("output", {2, 2, 2},
                        {0.0f, 0.0f,
                         0.0f, 0.0f,

                         111.0f, 112.0f,
                         121.0f, 122.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Edge_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});
  test.AddAttribute("mode", "edge");
  test.AddInput<float>("data", {2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {6, 7},
                        {11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Edge_3D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2, 2, 1, 2, 2});
  test.AddAttribute("mode", "edge");
  test.AddInput<float>("data", {1, 2, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f});
  test.AddOutput<float>("output", {3, 6, 7},
                        {11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,

                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,

                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                         12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Pad_Reflect_2D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{2, 2, 2, 2});
  test.AddAttribute("mode", "reflect");
  test.AddInput<float>("data", {3, 3},
                       {11.0f, 21.0f, 31.0f,
                        12.0f, 22.0f, 32.0f,
                        13.0f, 23.0f, 33.0f});
  test.AddOutput<float>("output", {7, 7},
                        {33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                         32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                         31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
