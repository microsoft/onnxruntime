// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(CumSumTest, _1DTest) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {5}, {1., 3., 6., 10., 15.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestFloat16) {
  if (DefaultCudaExecutionProvider().get() != nullptr) {
    OpTester test("CumSum", 14, onnxruntime::kOnnxDomain);
    test.AddInput<MLFloat16>("x", {3}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(2.0f)), MLFloat16(math::floatToHalf(3.0f))});
    test.AddInput<int32_t>("axis", {1}, {0});
    test.AddOutput<MLFloat16>("y", {3}, {MLFloat16(math::floatToHalf(1.0f)), MLFloat16(math::floatToHalf(3.0f)), MLFloat16(math::floatToHalf(6.0f))});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kCpuExecutionProvider});
  }
}
TEST(CumSumTest, _1DTestInvalidAxis) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {-3});
  test.AddOutput<float>("y", {5}, {1., 3., 6., 10., 15.});
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestNegAxis) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {-1});
  test.AddOutput<float>("y", {5}, {1., 3., 6., 10., 15.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestExclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {5}, {0., 1., 3., 6., 10.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _2DTestAxis0) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3}, {1., 2., 3., 4., 5., 6.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3}, {1., 2., 3., 5., 7., 9.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _2DTestAxis1) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3}, {1., 2., 3., 4., 5., 6.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3}, {1., 3., 6., 4., 9., 15.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _2DTestExclusiveAxis0) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3}, {1., 2., 3., 4., 5., 6.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3}, {0., 0., 0., 1., 2., 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _2DTestExclusiveAxis1) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3}, {1., 2., 3., 4., 5., 6.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3}, {0., 1., 3., 0., 4., 9.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis0) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 34., 36.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis1) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3, 4}, {1., 2., 3., 4., 6., 8., 10., 12., 15., 18., 21., 24., 13., 14., 15., 16., 30., 32., 34., 36., 51., 54., 57., 60.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis2) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {2});
  test.AddOutput<float>("y", {2, 3, 4}, {1., 3., 6., 10., 5., 11., 18., 26., 9., 19., 30., 42., 13., 27., 42., 58., 17., 35., 54., 74., 21., 43., 66., 90.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis0Exclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3, 4}, {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis1Exclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3, 4}, {0., 0., 0., 0., 1., 2., 3., 4., 6., 8., 10., 12., 0., 0., 0., 0., 13., 14., 15., 16., 30., 32., 34., 36.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis2Exclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {2});
  test.AddOutput<float>("y", {2, 3, 4}, {0., 1., 3., 6., 0., 5., 11., 18., 0., 9., 19., 30., 0., 13., 27., 42., 0., 17., 35., 54., 0., 21., 43., 66.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestReverse) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {5}, {15., 14., 12., 9., 5.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestReverseExclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {5}, {14., 12., 9., 5., 0.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis0Reverse) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3, 4}, {14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 34., 36., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis1Reverse) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3, 4}, {15., 18., 21., 24., 14., 16., 18., 20., 9., 10., 11., 12., 51., 54., 57., 60., 38., 40., 42., 44., 21., 22., 23., 24.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis2Reverse) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {2});
  test.AddOutput<float>("y", {2, 3, 4}, {10., 9., 7., 4., 26., 21., 15., 8., 42., 33., 23., 12., 58., 45., 31., 16., 74., 57., 39., 20., 90., 69., 47., 24.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis0ReverseExclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<float>("y", {2, 3, 4}, {13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis1ReverseExclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {1});
  test.AddOutput<float>("y", {2, 3, 4}, {14., 16., 18., 20., 9., 10., 11., 12., 0., 0., 0., 0., 38., 40., 42., 44., 21., 22., 23., 24., 0., 0., 0., 0.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _3DTestAxis2ReverseExclusive) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4}, {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.});
  test.AddInput<int32_t>("axis", {1}, {2});
  test.AddOutput<float>("y", {2, 3, 4}, {9., 7., 4., 0., 21., 15., 8., 0., 33., 23., 12., 0., 45., 31., 16., 0., 57., 39., 20., 0., 69., 47., 24., 0.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestInt32) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<int32_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<int32_t>("y", {5}, {1, 3, 6, 10, 15});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestInt64) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<int64_t>("y", {5}, {1, 3, 6, 10, 15});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestdouble) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<double>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {1}, {0});
  test.AddOutput<double>("y", {5}, {1., 3., 6., 10., 15.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
TEST(CumSumTest, _1DTestdouble_WithInt64Axis) {
  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<double>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int64_t>("axis", {1}, {0});
  test.AddOutput<double>("y", {5}, {1., 3., 6., 10., 15.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}
}  // namespace test
}  // namespace onnxruntime
