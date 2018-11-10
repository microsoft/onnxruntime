// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(TensorOpTest, Pad_Constant_1D) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{1, 2});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {5}, {1234.0f, 1.0f, 2.0f, 1234.0f, 1234.0f});
  test.Run();
}

TEST(TensorOpTest, Pad_Constant_1D_Zero) {
  OpTester test("Pad");

  test.AddAttribute("pads", std::vector<int64_t>{0, 0});
  test.AddAttribute("value", 1234.0f);
  test.AddInput<float>("data", {2}, {1.0f, 2.0f});
  test.AddOutput<float>("output", {2}, {1.0f, 2.0f});
  test.Run();
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
  test.Run();
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
  test.Run();
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

  test.Run();
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
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
