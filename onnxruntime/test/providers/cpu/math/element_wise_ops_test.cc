// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"
#include <algorithm>
#include <cmath>

namespace onnxruntime {
namespace test {

TEST(MathOpTest, Add_int32) {
  OpTester test("Add");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int32_t>("C", {3}, {5, 7, 9});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: elementwise inputs must not be Int32
}

TEST(MathOpTest, Add_int64) {
  OpTester test("Add");
  test.AddInput<int64_t>("A", {3}, {1, 2, 3});
  test.AddInput<int64_t>("B", {3}, {4, 5, 6});
  test.AddOutput<int64_t>("C", {3}, {5, 7, 9});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: INT64 is not supported
}

TEST(MathOpTest, Add) {
  OpTester test("Add");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {0.0f, 6.4f, 431.3f,
                         0.0f, 5.0f, -36.0f,
                         -10.8f, 18.6f, 0.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_Axis) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f});
  test.AddInput<float>("B", {3, 1},
                       {3.0f,
                        2.0f,
                        1.0f});
  test.AddOutput<float>("C", dims,
                        {4.0f, 5.0f, 6.0f,
                         6.0f, 7.0f, 8.0f,
                         8.0f, 9.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_0x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Add_Broadcast_0x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Add_Broadcast_1x0) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Add_Broadcast_1x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {1}, {10.0f});
  test.AddInput<float>("B", {1}, {2.0f});
  test.AddOutput<float>("C", {1}, {12.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_3x2_3x1) {
  OpTester test("Add");

  std::vector<int64_t> dims{3, 2};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f});
  test.AddInput<float>("B", {3, 1},
                       {1.0f,
                        2.0f,
                        3.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, 3.0f,
                         5.0f, 6.0f,
                         8.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, Add_Broadcast_2x1x4_1x3x1) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 4},
                       {101.0f, 102.0f, 103.0f, 104.0f,
                        201.0f, 202.0f, 203.0f, 204.0f});
  test.AddInput<float>("B", {1, 3, 1},
                       {010.0f, 020.0f, 030.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //Input batch size is inconsistent
}

TEST(MathOpTest, Add_Broadcast_2x1x1_3x4) {
  OpTester test("Add");

  test.AddInput<float>("A", {2, 1, 1},
                       {100.0f, 200.0f});
  test.AddInput<float>("B", {3, 4},
                       {011.0f, 012.0f, 013.0f, 014.0f,
                        021.0f, 022.0f, 023.0f, 024.0f,
                        031.0f, 032.0f, 033.0f, 034.0f});
  test.AddOutput<float>("C", {2, 3, 4},
                        {111.0f, 112.0f, 113.0f, 114.0f,
                         121.0f, 122.0f, 123.0f, 124.0f,
                         131.0f, 132.0f, 133.0f, 134.0f,

                         211.0f, 212.0f, 213.0f, 214.0f,
                         221.0f, 222.0f, 223.0f, 224.0f,
                         231.0f, 232.0f, 233.0f, 234.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //Input batch size is inconsistent
}

TEST(MathOpTest, Sub_int32) {
  OpTester test("Sub");
  test.AddInput<int32_t>("A", {3}, {1, 4, 3});
  test.AddInput<int32_t>("B", {3}, {4, 2, 4});
  test.AddOutput<int32_t>("C", {3}, {-3, 2, -1});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Sub_int64) {
  OpTester test("Sub");
  test.AddInput<int64_t>("A", {3}, {1, 5, 6});
  test.AddInput<int64_t>("B", {3}, {4, 5, 3});
  test.AddOutput<int64_t>("C", {3}, {-3, 0, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: INT64 is not supported
}

TEST(MathOpTest, Sub) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {2.0f, -2.4f, -433.3f,
                         0.0f, -2.0f, -164.0f,
                         0.0f, 0.0f, -20'000.0f});
  test.Run();
}

TEST(MathOpTest, Sub_Broadcast_Scalar) {
  OpTester test("Sub");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f,
                        -5.4f, 9.3f, -10'000.0f});
  test.AddInput<float>("B", {}, {5.0f});
  test.AddOutput<float>("C", dims,
                        {-4.0f, -3.0f, -6.0f,
                         -5.0f, -3.5f, -105.0f,
                         -10.4f, 4.3f, -10'005.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Mul_int32) {
  OpTester test("Mul");
  test.AddInput<int32_t>("A", {3}, {1, 2, 3});
  test.AddInput<int32_t>("B", {3}, {4, -3, 6});
  test.AddOutput<int32_t>("C", {3}, {4, -6, 18});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Mul_int64) {
  OpTester test("Mul");
  test.AddInput<int64_t>("A", {3}, {3, 6, -3});
  test.AddInput<int64_t>("B", {3}, {4, -3, -2});
  test.AddOutput<int64_t>("C", {3}, {12, -18, 6});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: INT64 is not supported
}

TEST(MathOpTest, Mul) {
  OpTester test("Mul");
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("A", dims,
                       {1.0f, 2.0f, -1.0f,
                        0.0f, 1.5f, -100.0f, -5.4f,
                        9.30f, -10'000.0f});
  test.AddInput<float>("B", dims,
                       {-1.0f, 4.4f, 432.3f,
                        0.0f, 3.5f, 64.0f, -5.4f,
                        9.30f, 10'000.0f});
  test.AddOutput<float>("C", dims,
                        {-1.0f, 8.8f, -432.3f,
                         0.0f, 5.25f, -6'400.0f,
                         29.16f, 86.49f, -100'000'000.0f});
  test.Run();
}

TEST(MathOpTest, Div_int32) {
  OpTester test("Div");
  test.AddInput<int32_t>("A", {3}, {4, 8, 8});
  test.AddInput<int32_t>("B", {3}, {1, 3, 2});
  test.AddOutput<int32_t>("C", {3}, {4, 2, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser:elementwise inputs must not be Int32
}

TEST(MathOpTest, Div_int64) {
  OpTester test("Div");
  test.AddInput<int64_t>("A", {3}, {4, 8, 8});
  test.AddInput<int64_t>("B", {3}, {2, 3, 4});
  test.AddOutput<int64_t>("C", {3}, {2, 2, 2});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: INT64 is not supported
}

TEST(MathOpTest, Div) {
  OpTester test("Div");
  std::vector<int64_t> dims{2, 3};
  test.AddInput<float>("A", dims,
                       {1'000.0f, 1.0f, 6.0f,
                        0.0f, -10.0f, -1.0f});
  test.AddInput<float>("B", dims,
                       {1'000.0f, 2.0f, 3.0f,
                        1.0f, -1.0f, 4.0f});
  test.AddOutput<float>("C", dims,
                        {1.0f, 0.5f, 2.0f,
                         0.0f, 10.0f, -0.25f});
  test.Run();
}

TEST(MathOpTest, Abs) {
  OpTester test("Abs");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims, {1.0f, -2.0f, -0.0f, -10.0f});
  test.AddOutput<float>("Y", dims, {1.0f, 2.0f, 0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Abs_int8) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int8_t>("Y", dims, {1, 2, 1, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Abs_int32) {
  OpTester test("Abs");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, 2, -1, -5});
  test.AddOutput<int32_t>("Y", dims, {1, 2, 1, 5});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Neg) {
  OpTester test("Neg");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, -2.0f,
                        0.0f, -10.0f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 2.0f,
                         -0.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Neg_int8) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int8_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Neg_int32) {
  OpTester test("Neg");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("X", dims, {1, -2, 0, -10});
  test.AddOutput<int32_t>("Y", dims, {-1, 2, 0, 10});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT parser: Int32 not allowed as input to this layer
}

TEST(MathOpTest, Floor) {
  OpTester test("Floor");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-2.0f, 0.0f,
                         -1.0f, 10.0f});
  test.Run();
}

TEST(MathOpTest, Ceil) {
  OpTester test("Ceil");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {-1.5f, 0.2f,
                        -0.5f, 10.3f});
  test.AddOutput<float>("Y", dims,
                        {-1.0f, 1.0f,
                         0.0f, 11.0f});
  test.Run();
}

TEST(MathOpTest, Reciprocal) {
  OpTester test("Reciprocal");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        -1.0f, -2.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 0.5f,
                         -1.0f, -0.5f});
  test.Run();
}

TEST(MathOpTest, Sqrt) {
  OpTester test("Sqrt");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 4.0f,
                        0.0f, 9.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, 2.0f,
                         0.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Pow) {
  OpTester test("Pow");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {2.0f, 2.0f,
                        std::sqrt(2.0f), 1.0f});
  test.AddInput<float>("Y", dims,
                       {0.0f, 8.0f,
                        2.0f, 9.0f});
  test.AddOutput<float>("Z", dims,
                        {1.0f, 256.0f,
                         2.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Pow_Broadcast_Scalar0) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", {}, {2.0f});
  test.AddInput<float>("Y", dims, {1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("Z", dims, {2.0f, 4.0f, 8.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Pow_Broadcast_Scalar1) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f});
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: dynamic shape is not supported
}

TEST(MathOpTest, Exp) {
  OpTester test("Exp");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {0.0f, 1.0f,
                        2.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {1.0f, std::exp(1.0f),
                         std::exp(2.0f), std::exp(10.0f)});
  test.SetOutputRelErr("Y", 1e-7f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Log) {
  OpTester test("Log");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("X", dims,
                       {1.0f, 2.0f,
                        5.0f, 10.0f});
  test.AddOutput<float>("Y", dims,
                        {0.0f, std::log(2.0f),
                         std::log(5.0f), std::log(10.0f)});
  test.Run();
}

TEST(MathOpTest, Sum_6) {
  OpTester test("Sum", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.25f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {3.0f, 0.0f, 6.0f,
                         -6.0f, 6.6f, 28.0f,
                         -1.0f, 0.06f, 0.25f});
  test.Run();
}

TEST(MathOpTest, Sum_8_Test1) {
  OpTester test("Sum", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 1}, {10.0f, 20.0f, 30.0f});
  test.AddInput<float>("data_2", {3, 1, 1}, {100.0f, 200.0f, 300.0f});
  test.AddOutput<float>("sum", {3, 3, 3},
                        {111.0f, 112.0f, 113.0f,
                         121.0f, 122.0f, 123.0f,
                         131.0f, 132.0f, 133.0f,

                         211.0f, 212.0f, 213.0f,
                         221.0f, 222.0f, 223.0f,
                         231.0f, 232.0f, 233.0f,

                         311.0f, 312.0f, 313.0f,
                         321.0f, 322.0f, 323.0f,
                         331.0f, 332.0f, 333.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT parser failed on this test
}

TEST(MathOpTest, Sum_8_Test2) {
  OpTester test("Sum", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {
                           1.0f,
                           0.0f,
                           1.0f,
                           -1.0f,
                           1.1f,
                           -100.0f,
                           -5.4f,
                           0.01f,
                           -74.0f,
                       });
  std::vector<int64_t> dims_1{3};
  test.AddInput<float>("data_1", dims_1,
                       {1.0f, 0.0f, 2.0f});
  std::vector<int64_t> dims_2{3, 1};
  test.AddInput<float>("data_2", dims_2,
                       {-3.0f, 3.3f, 64.0f});
  test.AddOutput<float>("sum", dims,
                        {-1.0f, -3.0f, 0.0f,
                         3.3f, 4.4f, -94.7f,
                         59.6f, 64.01f, -8.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "Sum is not correct", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Min_6) {
  OpTester test("Min", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("sum", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10'000.0f});
  test.Run();
}

TEST(MathOpTest, Min_8) {
  OpTester test("Min", 8);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("min", dims,
                        {1.0f, 0.0f, 1.0f,
                         -3.0f, 1.1f, -100.0f,
                         -5.4f, 0.01f, -10'000.0f});
  test.Run();
}

TEST(MathOpTest, Max_6) {
  OpTester test("Max", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.4f, 0.01f, -10'000.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 64.0f,
                        -1.0f, 0.02f, 0.1f});
  test.AddInput<float>("data_2", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 64.0f,
                        5.4f, 0.03f, 10'000.0f});
  test.AddOutput<float>("max", dims,
                        {1.0f, 0.0f, 3.0f,
                         -1.0f, 3.3f, 64.0f,
                         5.4f, 0.03f, 10'000.0f});
  test.Run();
}

TEST(MathOpTest, Max_8) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_2", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {-1.0f, -2.0f, 300.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         300.0f, 300.0f, 300.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //Input batch size is inconsistent
}

TEST(MathOpTest, Max_8_2inputbroadcast) {
  OpTester test("Max", 8);
  test.AddInput<float>("data_0", {1, 3},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_1", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("max", {3, 3},
                        {10.0f, 20.0f, 30.0f,
                         40.0f, 50.0f, 60.0f,
                         70.0f, 80.0f, 90.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //Input batch size is inconsistent
}

TEST(MathOpTest, Not) {
  OpTester test("Not");
  std::vector<int64_t> dims{2};
  test.AddInput<bool>("X", dims, {false, true});
  test.AddOutput<bool>("Y", dims, {true, false});
  test.Run();
}

TEST(MathOpTest, And) {
  OpTester test("And");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, false, false, true});
  test.Run();
}

TEST(MathOpTest, Or) {
  OpTester test("Or");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, true});
  test.Run();
}

TEST(MathOpTest, Xor) {
  OpTester test("Xor");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Xor_bcast3v2d) {
  OpTester test("Xor");

  test.AddInput<bool>("A", {2, 3, 4},
                      {false, true, false, true,
                       false, true, false, true,
                       false, true, false, true,

                       false, true, false, true,
                       false, true, false, true,
                       false, true, false, true});
  test.AddInput<bool>("B", {3, 4},
                      {false, false, true, true,
                       false, false, true, true,
                       false, false, true, true});
  test.AddOutput<bool>("C", {2, 3, 4},
                       {false, true, true, false,
                        false, true, true, false,
                        false, true, true, false,

                        false, true, true, false,
                        false, true, true, false,
                        false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less) {
  OpTester test("Less");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less_Scalar0) {
  OpTester test("Less");
  test.AddInput<float>("A", {1}, {1.0f});
  test.AddInput<float>("B", {4}, {1.0f, 1.5f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, true, false});
  test.Run();
}

TEST(MathOpTest, Less_Scalar1) {
  OpTester test("Less");
  test.AddInput<float>("A", {4}, {1.0f, 0.5f, 2.0f, -1.0f});
  test.AddInput<float>("B", {1}, {1.0f});
  test.AddOutput<bool>("C", {4}, {false, true, false, true});
  test.Run();
}

TEST(MathOpTest, Greater) {
  OpTester test("Greater");
  std::vector<int64_t> dims{4};
  test.AddInput<float>("A", dims, {1.0f, 0.0f, -1.0f, -1.0f});
  test.AddInput<float>("B", dims, {1.0f, 1.0f, 2.0f, -1.0f});
  test.AddOutput<bool>("C", dims, {false, false, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_bool) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<bool>("A", dims, {false, true, false, true});
  test.AddInput<bool>("B", dims, {false, false, true, true});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar0) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {1}, {false});
  test.AddInput<bool>("B", {4}, {false, false, true, true});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_bool_scalar1) {
  OpTester test("Equal");
  test.AddInput<bool>("A", {4}, {false, false, true, true});
  test.AddInput<bool>("B", {1}, {false});
  test.AddOutput<bool>("C", {4}, {true, true, false, false});
  test.Run();
}

TEST(MathOpTest, Equal_int32) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int32_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Equal_int64) {
  OpTester test("Equal");
  std::vector<int64_t> dims{4};
  test.AddInput<int64_t>("A", dims, {1, 0, -1, -1});
  test.AddInput<int64_t>("B", dims, {1, 1, 2, -1});
  test.AddOutput<bool>("C", dims, {true, false, false, true});
  test.Run();
}

TEST(MathOpTest, Mean_6) {
  OpTester test("Mean", 6);
  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("data_0", dims,
                       {1.0f, 0.0f, 1.0f,
                        -1.0f, 1.1f, -100.0f,
                        -5.0f, 0.01f, -10.0f});
  test.AddInput<float>("data_1", dims,
                       {1.0f, 0.0f, 2.0f,
                        -2.0f, 2.2f, 65.0f,
                        -1.0f, 0.02f, -1.0f});
  test.AddInput<float>("data_3", dims,
                       {1.0f, 0.0f, 3.0f,
                        -3.0f, 3.3f, 65.0f,
                        -3.0f, 0.03f, -1.0f});
  test.AddOutput<float>("mean", dims,
                        {1.0f, 0.0f, 2.0f,
                         -2.0f, 2.2f, 10.0f,
                         -3.0f, 0.02f, -4.0f});
  test.Run();
}

TEST(MathOpTest, Mean_8) {
  OpTester test("Mean", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<float>("data_1", {3, 1},
                       {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("data_3", {3, 3},
                       {10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f});
  test.AddOutput<float>("mean", {3, 3},
                        {12.0f / 3.0f, 22.0f / 3.0f, 32.0f / 3.0f,
                         43.0f / 3.0f, 53.0f / 3.0f, 63.0f / 3.0f,
                         74.0f / 3.0f, 84.0f / 3.0f, 94.0f / 3.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //Input batch size is inconsistent
}

#ifndef DISABLE_CONTRIB_OPS
TEST(MathOpTest, AffineDefaultAttributes) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Affine) {
  OpTester test("Affine");
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("alpha", 2.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {1.0f, 3.0f, 5.0f, 7.0f});
  test.Run();
}
#endif

template <float (&op)(float value)>
void TrigTest(OpTester& test, std::initializer_list<float> input) {
  std::vector<int64_t> dims{static_cast<int64_t>(input.size())};

  std::vector<float> output;
  for (auto v : input)
    output.push_back(op(v));

  test.AddInput<float>("X", dims, input);
  test.AddOutput<float>("Y", dims, output);
  test.Run();
}

TEST(MathOpTest, Sin) {
  OpTester test("Sin");
  TrigTest<std::sin>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Cos) {
  OpTester test("Cos");
  TrigTest<std::cos>(test, {1.1f, -1.1f, 2.2f, -2.2f});
}

TEST(MathOpTest, Tan) {
  OpTester test("Tan");
  TrigTest<std::tan>(test, {-100.0f, -50.0f, 0.0f, 50.0f, 100.0f});
}

TEST(MathOpTest, Asin) {
  OpTester test("Asin");
  TrigTest<std::asin>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acos) {
  OpTester test("Acos");
  TrigTest<std::acos>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Atan) {
  OpTester test("Atan");
  TrigTest<std::atan>(test, {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f});
}

TEST(MathOpTest, Sinh) {
  OpTester test("Sinh", 9);
  TrigTest<std::sinh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Cosh) {
  OpTester test("Cosh", 9);
  TrigTest<std::cosh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Asinh) {
  OpTester test("Asinh", 9);
  TrigTest<std::asinh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Acosh) {
  OpTester test("Acosh", 9);
  TrigTest<std::acosh>(test, {1.0f, 1.1f, 3.0f, 10.0f, 100.0f});
}

TEST(MathOpTest, Atanh) {
  OpTester test("Atanh", 9);
  TrigTest<std::atanh>(test, {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f});
}

TEST(MathOpTest, Expand_8_3x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3, 1}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {1}, {MLFloat16(math::floatToHalf(1.0f))});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f))});
  test.Run();
}

TEST(MathOpTest, Expand_8_3x1_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f)),
                             MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.Run();
}

TEST(MathOpTest, Expand_8_1x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3, 1}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(1.0f)),
                             MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(2.0f)),
                             MLFloat16(math::floatToHalf(3.0f)), MLFloat16(math::floatToHalf(3.0f)), MLFloat16(math::floatToHalf(3.0f))});
  test.Run();
}

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
TEST(MathOpTest, Scale) {
  OpTester test("Scale");
  std::vector<int64_t> dims{2, 2};
  test.AddAttribute("scale", 2.0f);
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(MathOpTest, Scale_Default) {
  OpTester test("Scale");
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.AddOutput<float>("B", dims, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}
}  // namespace contrib
#endif

TEST(MathOpTest, Erf) {
  OpTester test("Erf", 9);
  std::vector<int64_t> dims{2, 2};
  test.AddInput<float>("A", dims, {0.5f, 1.0f, 0.7f, 2.0f});
  test.AddOutput<float>("B", dims, {0.5204999f, 0.8427008f, 0.6778012f, 0.9953223f});
  test.Run();
}

const int ModOp_ver = 10;

TEST(ModOpTest, Fmod_float_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);
  test.AddInput<float>("X", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
  test.AddInput<float>("Y", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
  test.AddOutput<float>("Z", {6}, {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f});

  test.Run();
}

std::vector<MLFloat16> MakeMLFloat16(const std::initializer_list<float>& input) {
  std::vector<MLFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output),
                 [](float fl) {
                   return MLFloat16(math::floatToHalf(fl));
                 });
  return output;
}

TEST(ModOpTest, Fmod_float16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<MLFloat16>("X", {6}, MakeMLFloat16({-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f}));
  test.AddInput<MLFloat16>("Y", {6}, MakeMLFloat16({2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f}));
  // The output above is {-0.1f, 0.4f, 5.f, 0.1f, -0.4f, 3.f} for float
  test.AddOutput<MLFloat16>("Z", {6}, MakeMLFloat16({-0.1015625f, 0.3984375f, 5.f, 0.1015625f, -0.3984375f, 3.f}));

  test.Run();
}

TEST(ModOpTest, Int8_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int8_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int8_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int8_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int8_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt8_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint8_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint8_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint8_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int16_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int16_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int16_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int16_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int16_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt16_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint16_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint16_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint16_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int32_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int32_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int32_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt32_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint32_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint32_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint32_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, -2, 5, 0, 2, 3});

  test.Run();
}

TEST(ModOpTest, Int64_mixed_sign_fmod) {
  OpTester test("Mod", ModOp_ver);
  test.AddAttribute<int64_t>("fmod", 1);

  test.AddInput<int64_t>("X", {6}, {-4, 7, 5, 4, -7, 8});
  test.AddInput<int64_t>("Y", {6}, {2, -3, 8, -2, 3, 5});
  test.AddOutput<int64_t>("Z", {6}, {0, 1, 5, 0, -1, 3});

  test.Run();
}

TEST(ModOpTest, UInt64_mod) {
  OpTester test("Mod", ModOp_ver);
  test.AddInput<uint64_t>("X", {6}, {4, 7, 5, 4, 7, 8});
  test.AddInput<uint64_t>("Y", {6}, {2, 3, 8, 2, 3, 5});
  test.AddOutput<uint64_t>("Z", {6}, {0, 1, 5, 0, 1, 3});

  test.Run();
}

TEST(ModOpTest, Int32_mod_bcast) {
  OpTester test("Mod", ModOp_ver);

  std::vector<int32_t> input_sequence;
  input_sequence.resize(30);
  std::generate(input_sequence.begin(), input_sequence.end(),
                [n = 0]() mutable { return n++; });

  // input [0..29]
  test.AddInput<int32_t>("X", {3, 2, 5}, input_sequence);
  test.AddInput<int32_t>("Y", {1}, {7});

  test.AddOutput<int32_t>("Z", {3, 2, 5},
                          {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
