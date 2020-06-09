// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
// scalar zero & scale with uint8
TEST(DequantizeLinearOpTest, DequantizeLinear_0) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  test.Run();
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, DequantizeLinear_1) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  test.Run();
}

// 2d inputs
TEST(DequantizeLinearOpTest, DequantizeLinear_2D) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddInput<float>("scale", {}, {1.0f});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 10, 20, 30});
  test.Run();
}

// dequantize with scalar data
TEST(DequantizeLinearOpTest, DequantizeLinear_Scalar) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10});
  test.AddOutput<float>("y", {}, {220.0f});
  test.Run();
}

// dequantize without zero point
TEST(DequantizeLinearOpTest, DequantizeLinear_Without_Zero_Point) {
  OpTester test("DequantizeLinear", 10);
  test.AddInput<int8_t>("x", {}, {100});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddOutput<float>("y", {}, {200.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, QuantizeLinear_uint8) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 1000, -254, -1000});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", dims, {128, 129, 130, 255, 1, 0});
  test.Run();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, QuantizeLinear_int8) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{6};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, -2, -5});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", dims, {0, 51, 76, 127, -51, -127});
  test.Run();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, QuantizeLinear_int8_NegativeZeroPoint) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {-23});
  test.AddOutput<int8_t>("y", dims, {-23, 28, 53, 104, 127, -74, -128, -128});
  test.Run();
}

// quantize with scalar zero point and scale
TEST(QuantizeLinearOpTest, QuantizeLinear_int8_PositiveZeroPoint) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{8};
  test.AddInput<float>("x", dims, {0, 2, 3, 5, 6, -2, -5, -6});
  test.AddInput<float>("y_scale", {}, {.039215686f});
  test.AddInput<int8_t>("y_zero_point", {}, {23});
  test.AddOutput<int8_t>("y", dims, {23, 74, 99, 127, 127, -28, -104, -128});
  test.Run();
}

// quantize with 2D data
TEST(QuantizeLinearOpTest, QuantizeLinear_2D) {
  OpTester test("QuantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<float>("X", dims,
                       {0, 2, 3, 1000,
                        0, 2, 3, 1000,
                        0, 2, 3, 1000});
  test.AddInput<float>("scale", {}, {4});
  test.AddInput<uint8_t>("zero_point", {}, {0});
  test.AddOutput<uint8_t>("Y", dims,
                          {0, 0, 1, 250,
                           0, 0, 1, 250,
                           0, 0, 1, 250});
  test.Run();
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, QuantizeLinear_Scalar) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {128});
  test.AddOutput<uint8_t>("y", {}, {130});
  test.Run();
}

// quantize with scalar data
TEST(QuantizeLinearOpTest, DISABLED_QuantizeLinear_Without_Zero_Point) {
  OpTester test("QuantizeLinear", 10);
  test.AddInput<float>("x", {}, {3});
  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddOutput<uint8_t>("y", {}, {2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
