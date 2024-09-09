// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
TEST(ConvIntegerTest, WithoutPadding_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {2, 2,
                          2, 2});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {1});
  std::vector<int64_t> y_dims{1, 1, 2, 2};
  test.AddOutput<int32_t>("y", y_dims,
                          {12, 16,
                           24, 28});
  test.Run();
}

TEST(ConvIntegerTest, WithPadding_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {2, 2,
                          2, 2});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {1});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  std::vector<int64_t> y_dims{1, 1, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 3, 5, 3,
                           5, 12, 16, 9,
                           11, 24, 28, 15,
                           7, 15, 17, 9});
  test.Run();
}

TEST(ConvIntegerTest, WithoutPadding_2D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 2, 2};
  test.AddInput<int8_t>("w", w_dims,
                        {-9, -9,
                         -9, -9});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {-10});
  std::vector<int64_t> y_dims{1, 1, 2, 2};
  test.AddOutput<int32_t>("y", y_dims,
                          {12, 16,
                           24, 28});
  test.Run();
}

TEST(ConvIntegerTest, WithGroup_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{3, 1, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {11, 12,
                          12, 11,
                          13, 14,
                          14, 13,
                          15, 16,
                          16, 15});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {10});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  test.AddAttribute("group", static_cast<int64_t>(3));
  std::vector<int64_t> y_dims{1, 3, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 4, 7, 6,
                           6, 18, 24, 15,
                           15, 36, 42, 24,
                           14, 23, 26, 9,
                           30, 73, 80, 48,
                           79, 168, 182, 96,
                           100, 210, 224, 117,
                           64, 116, 123, 54,
                           95, 214, 225, 126,
                           224, 462, 484, 249,
                           257, 528, 550, 282,
                           150, 281, 292, 135});
  test.Run();
}

TEST(ConvIntegerTest, WithGroup_2D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{3, 1, 2, 2};
  test.AddInput<int8_t>("w", w_dims,
                        {-2, -1,
                         -2, -1,
                         0, 1,
                         1, 0,
                         2, 3,
                         3, 2});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {-3});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  test.AddAttribute("group", static_cast<int64_t>(3));
  std::vector<int64_t> y_dims{1, 3, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 4, 7, 6,
                           6, 18, 24, 15,
                           15, 36, 42, 24,
                           14, 23, 26, 9,
                           30, 73, 80, 48,
                           79, 168, 182, 96,
                           100, 210, 224, 117,
                           64, 116, 123, 54,
                           95, 214, 225, 126,
                           224, 462, 484, 249,
                           257, 528, 550, 282,
                           150, 281, 292, 135});
  test.Run();
}

TEST(ConvIntegerTest, WithPadding_3D_u8u8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{1, 1, 2, 2, 2};
  test.AddInput<uint8_t>("w", w_dims,
                         {11, 11,
                          11, 11,
                          11, 11,
                          11, 11});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {10});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1, 1, 1});
  std::vector<int64_t> y_dims{1, 1, 4, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 3, 5, 3,
                           5, 12, 16, 9,
                           11, 24, 28, 15,
                           7, 15, 17, 9,
                           11, 24, 28, 15,
                           28, 60, 68, 36,
                           40, 84, 92, 48,
                           23, 48, 52, 27,
                           29, 60, 64, 33,
                           64, 132, 140, 72,
                           76, 156, 164, 84,
                           41, 84, 88, 45,
                           19, 39, 41, 21,
                           41, 84, 88, 45,
                           47, 96, 100, 51,
                           25, 51, 53, 27});
  test.Run();
}

TEST(ConvIntegerTest, WithPadding_3D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{1, 1, 2, 2, 2};
  test.AddInput<int8_t>("w", w_dims,
                        {-9, -9,
                         -9, -9,
                         -9, -9,
                         -9, -9});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {-10});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1, 1, 1});
  std::vector<int64_t> y_dims{1, 1, 4, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {1, 3, 5, 3,
                           5, 12, 16, 9,
                           11, 24, 28, 15,
                           7, 15, 17, 9,
                           11, 24, 28, 15,
                           28, 60, 68, 36,
                           40, 84, 92, 48,
                           23, 48, 52, 27,
                           29, 60, 64, 33,
                           64, 132, 140, 72,
                           76, 156, 164, 84,
                           41, 84, 88, 45,
                           19, 39, 41, 21,
                           41, 84, 88, 45,
                           47, 96, 100, 51,
                           25, 51, 53, 27});
  test.Run();
}

TEST(ConvIntegerTest, Pointwise_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10});
  std::vector<int64_t> w_dims{1, 1, 1, 1};
  test.AddInput<uint8_t>("w", w_dims, {5});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {1});
  std::vector<int64_t> y_dims{1, 1, 3, 3};
  test.AddOutput<int32_t>("y", y_dims,
                          {4, 8, 12,
                           16, 20, 24,
                           28, 32, 36});
  test.Run();
}

TEST(ConvIntegerTest, Pointwise_3D_u8u8) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{1, 1, 1, 1, 1};
  test.AddInput<uint8_t>("w", w_dims, {5});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<uint8_t>("w_zero_point", {}, {1});
  std::vector<int64_t> y_dims{1, 1, 3, 3, 3};
  test.AddOutput<int32_t>("y", y_dims,
                          {4, 8, 12,
                           16, 20, 24,
                           28, 32, 36,
                           40, 44, 48,
                           52, 56, 60,
                           64, 68, 72,
                           76, 80, 84,
                           88, 92, 96,
                           100, 104, 108});
  test.Run();
}
TEST(ConvIntegerTest, Pointwise_3D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 3, 3, 3};
  test.AddInput<uint8_t>("x", x_dims,
                         {2, 3, 4,
                          5, 6, 7,
                          8, 9, 10,
                          11, 12, 13,
                          14, 15, 16,
                          17, 18, 19,
                          20, 21, 22,
                          23, 24, 25,
                          26, 27, 28});
  std::vector<int64_t> w_dims{1, 1, 1, 1, 1};
  test.AddInput<int8_t>("w", w_dims, {-16});
  test.AddInput<uint8_t>("x_zero_point", {}, {1});
  test.AddInput<int8_t>("w_zero_point", {}, {-20});
  std::vector<int64_t> y_dims{1, 1, 3, 3, 3};
  test.AddOutput<int32_t>("y", y_dims,
                          {4, 8, 12,
                           16, 20, 24,
                           28, 32, 36,
                           40, 44, 48,
                           52, 56, 60,
                           64, 68, 72,
                           76, 80, 84,
                           88, 92, 96,
                           100, 104, 108});
  test.Run();
}

TEST(ConvIntegerTest, WithStride2_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 7, 7};
  test.AddInput<uint8_t>("x", x_dims,
                         {10, 11, 12, 13, 14, 15, 16,
                          20, 21, 22, 23, 24, 25, 26,
                          30, 31, 32, 33, 34, 35, 36,
                          40, 41, 42, 43, 44, 45, 46,
                          50, 51, 52, 53, 54, 55, 56,
                          60, 61, 62, 63, 64, 65, 66,
                          70, 71, 72, 73, 74, 75, 76});
  std::vector<int64_t> w_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("w", w_dims,
                         {11, 12, 11,
                          12, 13, 12,
                          11, 12, 11});
  test.AddInput<uint8_t>("x_zero_point", {}, {10});
  test.AddInput<uint8_t>("w_zero_point", {}, {10});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  test.AddAttribute<std::vector<int64_t>>("strides", {2, 2});
  std::vector<int64_t> y_dims{1, 1, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {33, 62, 84, 75,
                           224, 330, 360, 282,
                           444, 630, 660, 502,
                           453, 642, 664, 495});
  // Exercise the (stride_w = 2) path inside Math::Im2col.
  test.Run();
}

TEST(ConvIntegerTest, WithStride2_2D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 7, 7};
  test.AddInput<uint8_t>("x", x_dims,
                         {10, 11, 12, 13, 14, 15, 16,
                          20, 21, 22, 23, 24, 25, 26,
                          30, 31, 32, 33, 34, 35, 36,
                          40, 41, 42, 43, 44, 45, 46,
                          50, 51, 52, 53, 54, 55, 56,
                          60, 61, 62, 63, 64, 65, 66,
                          70, 71, 72, 73, 74, 75, 76});
  std::vector<int64_t> w_dims{1, 1, 3, 3};
  test.AddInput<int8_t>("w", w_dims,
                        {-2, -1, -2,
                         -1, 0, -1,
                         -2, -1, -2});
  test.AddInput<uint8_t>("x_zero_point", {}, {10});
  test.AddInput<int8_t>("w_zero_point", {}, {-3});
  test.AddAttribute<std::vector<int64_t>>("pads", {1, 1, 1, 1});
  test.AddAttribute<std::vector<int64_t>>("strides", {2, 2});
  std::vector<int64_t> y_dims{1, 1, 4, 4};
  test.AddOutput<int32_t>("y", y_dims,
                          {33, 62, 84, 75,
                           224, 330, 360, 282,
                           444, 630, 660, 502,
                           453, 642, 664, 495});
  // Exercise the (stride_w = 2) path inside Math::Im2col.
  test.Run();
}

TEST(ConvIntegerTest, WithStride3_2D_u8u8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 7, 7};
  test.AddInput<uint8_t>("x", x_dims,
                         {10, 11, 12, 13, 14, 15, 16,
                          20, 21, 22, 23, 24, 25, 26,
                          30, 31, 32, 33, 34, 35, 36,
                          40, 41, 42, 43, 44, 45, 46,
                          50, 51, 52, 53, 54, 55, 56,
                          60, 61, 62, 63, 64, 65, 66,
                          70, 71, 72, 73, 74, 75, 76});
  std::vector<int64_t> w_dims{1, 1, 3, 3};
  test.AddInput<uint8_t>("w", w_dims,
                         {11, 12, 11,
                          12, 13, 12,
                          11, 12, 11});
  test.AddInput<uint8_t>("x_zero_point", {}, {10});
  test.AddInput<uint8_t>("w_zero_point", {}, {10});
  test.AddAttribute<std::vector<int64_t>>("pads", {2, 2, 1, 1});
  test.AddAttribute<std::vector<int64_t>>("strides", {3, 3});
  std::vector<int64_t> y_dims{1, 1, 3, 3};
  test.AddOutput<int32_t>("y", y_dims,
                          {0, 8, 20,
                           80, 330, 375,
                           200, 780, 825});
  // Exercise the (stride_w > 2) path inside Math::Im2col.
  test.Run();
}

TEST(ConvIntegerTest, WithStride3_2D_u8s8) {
  OpTester test("ConvInteger", 10);
  std::vector<int64_t> x_dims{1, 1, 7, 7};
  test.AddInput<uint8_t>("x", x_dims,
                         {10, 11, 12, 13, 14, 15, 16,
                          20, 21, 22, 23, 24, 25, 26,
                          30, 31, 32, 33, 34, 35, 36,
                          40, 41, 42, 43, 44, 45, 46,
                          50, 51, 52, 53, 54, 55, 56,
                          60, 61, 62, 63, 64, 65, 66,
                          70, 71, 72, 73, 74, 75, 76});
  std::vector<int64_t> w_dims{1, 1, 3, 3};
  test.AddInput<int8_t>("w", w_dims,
                        {-2, -1, -2,
                         -1, 0, -1,
                         -2, -1, -2});
  test.AddInput<uint8_t>("x_zero_point", {}, {10});
  test.AddInput<int8_t>("w_zero_point", {}, {-3});
  test.AddAttribute<std::vector<int64_t>>("pads", {2, 2, 1, 1});
  test.AddAttribute<std::vector<int64_t>>("strides", {3, 3});
  std::vector<int64_t> y_dims{1, 1, 3, 3};
  test.AddOutput<int32_t>("y", y_dims,
                          {0, 8, 20,
                           80, 330, 375,
                           200, 780, 825});
  // Exercise the (stride_w > 2) path inside Math::Im2col.
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
