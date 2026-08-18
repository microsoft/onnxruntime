// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

// 1D tests - basic functionality
TEST(CumProdTest, _1DTest) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {5}, {1.f, 2.f, 6.f, 24.f, 120.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestInvalidAxis) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {-3});
  test.AddOutput<float>("y", {5}, {1.f, 2.f, 6.f, 24.f, 120.f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestNegAxis) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {-1});
  test.AddOutput<float>("y", {5}, {1.f, 2.f, 6.f, 24.f, 120.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Exclusive mode: identity element is 1, shift right
// input:  [1, 2, 3, 4, 5]
// output: [1, 1, 2, 6, 24]
TEST(CumProdTest, _1DTestExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {5}, {1.f, 1.f, 2.f, 6.f, 24.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Exclusive with axis dim=1: all elements should be identity (1)
TEST(CumProdTest, _1DTestExclusiveAxisHasSingleValue) {
  {
    // forward
    OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
    test.AddAttribute<int64_t>("exclusive", 1);
    test.AddInput<float>("x", {1, 2}, {3.f, 4.f});
    test.AddInput<int32_t>("axis", {}, {0});
    test.AddOutput<float>("y", {1, 2}, {1.f, 1.f});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
  {
    // reverse
    OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
    test.AddAttribute<int64_t>("exclusive", 1);
    test.AddAttribute<int64_t>("reverse", 1);
    test.AddInput<float>("x", {1, 2}, {3.f, 4.f});
    test.AddInput<int32_t>("axis", {}, {0});
    test.AddOutput<float>("y", {1, 2}, {1.f, 1.f});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
}

// 2D tests
// input: [[1, 2, 3], [4, 5, 6]], axis=0
// output: [[1, 2, 3], [4, 10, 18]]
TEST(CumProdTest, _2DTestAxis0) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {2, 3}, {1.f, 2.f, 3.f, 4.f, 10.f, 18.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// input: [[1, 2, 3], [4, 5, 6]], axis=1
// output: [[1, 2, 6], [4, 20, 120]]
TEST(CumProdTest, _2DTestAxis1) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<float>("y", {2, 3}, {1.f, 2.f, 6.f, 4.f, 20.f, 120.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Exclusive 2D axis=0: identity row, then element-wise product with input
// input: [[1, 2, 3], [4, 5, 6]], axis=0, exclusive
// output: [[1, 1, 1], [1, 2, 3]]
TEST(CumProdTest, _2DTestExclusiveAxis0) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {2, 3}, {1.f, 1.f, 1.f, 1.f, 2.f, 3.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Exclusive 2D axis=1
// input: [[1, 2, 3], [4, 5, 6]], axis=1, exclusive
// output: [[1, 1, 2], [1, 4, 20]]
TEST(CumProdTest, _2DTestExclusiveAxis1) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  test.AddInput<int32_t>("axis", {}, {1});
  test.AddOutput<float>("y", {2, 3}, {1.f, 1.f, 2.f, 1.f, 4.f, 20.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 3D tests with shape {2, 3, 4}
// Using values 1..24
TEST(CumProdTest, _3DTestAxis0) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {0});
  // axis=0: product along first dimension
  // out[0,:,:] = x[0,:,:], out[1,:,:] = x[0,:,:] * x[1,:,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                         13.f, 28.f, 45.f, 64.f, 85.f, 108.f, 133.f, 160.f, 189.f, 220.f, 253.f, 288.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis1) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {1});
  // axis=1: product along second dimension
  // out[:,0,:] = x[:,0,:], out[:,1,:] = x[:,0,:]*x[:,1,:], out[:,2,:] = x[:,0,:]*x[:,1,:]*x[:,2,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 2.f, 3.f, 4.f, 5.f, 12.f, 21.f, 32.f, 45.f, 120.f, 231.f, 384.f,
                         13.f, 14.f, 15.f, 16.f, 221.f, 252.f, 285.f, 320.f, 4641.f, 5544.f, 6555.f, 7680.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis2) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {2});
  // axis=2: product along last dimension
  // out[:,:,0] = x[:,:,0], out[:,:,1] = x[:,:,0]*x[:,:,1], etc.
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 2.f, 6.f, 24.f, 5.f, 30.f, 210.f, 1680.f, 9.f, 90.f, 990.f, 11880.f,
                         13.f, 182.f, 2730.f, 43680.f, 17.f, 306.f, 5814.f, 116280.f, 21.f, 462.f, 10626.f, 255024.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 3D exclusive tests
TEST(CumProdTest, _3DTestAxis0Exclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {0});
  // exclusive axis=0: first slice is all 1s, second = x[0,:,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
                         1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis1Exclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {1});
  // exclusive axis=1: out[:,0,:] = 1, out[:,1,:] = x[:,0,:], out[:,2,:] = x[:,0,:]*x[:,1,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 3.f, 4.f, 5.f, 12.f, 21.f, 32.f,
                         1.f, 1.f, 1.f, 1.f, 13.f, 14.f, 15.f, 16.f, 221.f, 252.f, 285.f, 320.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis2Exclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {2});
  // exclusive axis=2: out[:,:,0] = 1, out[:,:,1] = x[:,:,0], out[:,:,2] = x[:,:,0]*x[:,:,1], etc.
  test.AddOutput<float>("y", {2, 3, 4},
                        {1.f, 1.f, 2.f, 6.f, 1.f, 5.f, 30.f, 210.f, 1.f, 9.f, 90.f, 990.f,
                         1.f, 13.f, 182.f, 2730.f, 1.f, 17.f, 306.f, 5814.f, 1.f, 21.f, 462.f, 10626.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Reverse tests
// input: [1, 2, 3, 4, 5], reverse
// output: [120, 120, 60, 20, 5]
TEST(CumProdTest, _1DTestReverse) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {5}, {120.f, 120.f, 60.f, 20.f, 5.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Reverse exclusive
// input: [1, 2, 3, 4, 5], reverse, exclusive
// output: [120, 60, 20, 5, 1]
TEST(CumProdTest, _1DTestReverseExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {5}, {1.f, 2.f, 3.f, 4.f, 5.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {5}, {120.f, 60.f, 20.f, 5.f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 3D reverse tests
TEST(CumProdTest, _3DTestAxis0Reverse) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {0});
  // reverse axis=0: out[1,:,:]=x[1,:,:], out[0,:,:]=x[0,:,:]*x[1,:,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {13.f, 28.f, 45.f, 64.f, 85.f, 108.f, 133.f, 160.f, 189.f, 220.f, 253.f, 288.f,
                         13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis1Reverse) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {1});
  // reverse axis=1: out[:,2,:]=x[:,2,:], out[:,1,:]=x[:,1,:]*x[:,2,:], out[:,0,:]=x[:,0,:]*x[:,1,:]*x[:,2,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {45.f, 120.f, 231.f, 384.f, 45.f, 60.f, 77.f, 96.f, 9.f, 10.f, 11.f, 12.f,
                         4641.f, 5544.f, 6555.f, 7680.f, 357.f, 396.f, 437.f, 480.f, 21.f, 22.f, 23.f, 24.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis2Reverse) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {2});
  // reverse axis=2: out[:,:,3]=x[:,:,3], out[:,:,2]=x[:,:,2]*x[:,:,3], etc.
  test.AddOutput<float>("y", {2, 3, 4},
                        {24.f, 24.f, 12.f, 4.f, 1680.f, 336.f, 56.f, 8.f, 11880.f, 1320.f, 132.f, 12.f,
                         43680.f, 3360.f, 240.f, 16.f, 116280.f, 6840.f, 380.f, 20.f, 255024.f, 12144.f, 552.f, 24.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// 3D reverse exclusive tests
TEST(CumProdTest, _3DTestAxis0ReverseExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {0});
  // reverse exclusive axis=0: out[1,:,:]=1, out[0,:,:]=x[1,:,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f,
                         1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis1ReverseExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {1});
  // reverse exclusive axis=1: out[:,2,:]=1, out[:,1,:]=x[:,2,:], out[:,0,:]=x[:,1,:]*x[:,2,:]
  test.AddOutput<float>("y", {2, 3, 4},
                        {45.f, 60.f, 77.f, 96.f, 9.f, 10.f, 11.f, 12.f, 1.f, 1.f, 1.f, 1.f,
                         357.f, 396.f, 437.f, 480.f, 21.f, 22.f, 23.f, 24.f, 1.f, 1.f, 1.f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _3DTestAxis2ReverseExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {2, 3, 4},
                       {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f,
                        13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f});
  test.AddInput<int32_t>("axis", {}, {2});
  // reverse exclusive axis=2: out[:,:,3]=1, out[:,:,2]=x[:,:,3], out[:,:,1]=x[:,:,2]*x[:,:,3], etc.
  test.AddOutput<float>("y", {2, 3, 4},
                        {24.f, 12.f, 4.f, 1.f, 336.f, 56.f, 8.f, 1.f, 1320.f, 132.f, 12.f, 1.f,
                         3360.f, 240.f, 16.f, 1.f, 6840.f, 380.f, 20.f, 1.f, 12144.f, 552.f, 24.f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Type-specific tests
TEST(CumProdTest, _1DTestInt32) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<int32_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<int32_t>("y", {5}, {1, 2, 6, 24, 120});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestInt64) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<int64_t>("y", {5}, {1, 2, 6, 24, 120});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestDouble) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<double>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<double>("y", {5}, {1., 2., 6., 24., 120.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestDouble_WithInt64Axis) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<double>("x", {5}, {1., 2., 3., 4., 5.});
  test.AddInput<int64_t>("axis", {}, {0});
  test.AddOutput<double>("y", {5}, {1., 2., 6., 24., 120.});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestUint32) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<uint32_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<uint32_t>("y", {5}, {1, 2, 6, 24, 120});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(CumProdTest, _1DTestUint64) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<uint64_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<uint64_t>("y", {5}, {1, 2, 6, 24, 120});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Matches ONNX spec example exactly
// input: [1, 2, 3], axis=0 -> [1, 2, 6]
TEST(CumProdTest, _OnnxSpecExample) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddInput<float>("x", {3}, {1.f, 2.f, 3.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {3}, {1.f, 2.f, 6.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// ONNX spec example: exclusive=1 -> [1, 1, 2]
TEST(CumProdTest, _OnnxSpecExampleExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddInput<float>("x", {3}, {1.f, 2.f, 3.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {3}, {1.f, 1.f, 2.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// ONNX spec example: reverse=1 -> [6, 6, 3]
TEST(CumProdTest, _OnnxSpecExampleReverse) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {3}, {1.f, 2.f, 3.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {3}, {6.f, 6.f, 3.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// ONNX spec example: exclusive=1, reverse=1 -> [6, 3, 1]
TEST(CumProdTest, _OnnxSpecExampleReverseExclusive) {
  OpTester test("CumProd", 26, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("exclusive", 1);
  test.AddAttribute<int64_t>("reverse", 1);
  test.AddInput<float>("x", {3}, {1.f, 2.f, 3.f});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<float>("y", {3}, {6.f, 3.f, 1.f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
