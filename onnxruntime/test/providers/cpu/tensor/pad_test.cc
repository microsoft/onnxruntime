// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// There is support for int32, int64, float, and double types for opset-11 Pad alone in ORT
template <typename T>
static void RunOpset11TypedTest(
    const std::vector<int64_t>& input_dims,
    const std::vector<T>& input,
    const std::vector<int64_t>& pads,
    T value,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output,
    std::string mode = "constant",
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& error_msg = "") {
  // ONNX domain opset-11
  OpTester test("Pad", 11);
  if (mode != "constant")
    test.AddAttribute("mode", mode);
  test.AddInput<T>("data", input_dims, input);
  test.AddInput<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads);
  test.AddInput<T>("value", {1}, {value});
  test.AddOutput<T>("output", output_dims, output);
  // NGraph and TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
  test.Run(expect, error_msg, {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

// There is only support for float type for opset-10 and MSDomain kernel in ORT
static void RunAllOpsetAllDomainPadTests(
    const std::vector<int64_t>& input_dims,
    const std::vector<float>& input,
    const std::vector<int64_t>& pads,
    float value,
    const std::vector<int64_t>& output_dims,
    const std::vector<float>& output,
    std::string mode = "constant",
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& error_msg = "") {
  // ONNX domain opset-10
  OpTester test1("Pad", 10);
  test1.AddInput<float>("data", input_dims, input);
  if (mode != "constant") test1.AddAttribute("mode", mode);
  test1.AddAttribute("pads", pads);
  test1.AddAttribute("value", value);
  test1.AddOutput<float>("output", output_dims, output);
  test1.Run(expect, error_msg);

  // ONNX domain opset-11
  RunOpset11TypedTest<float>(input_dims,
                             input,
                             pads,
                             value,
                             output_dims,
                             output,
                             mode, expect, error_msg);

#ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test3("Pad", 1, kMSDomain);
  if (mode != "constant") test3.AddAttribute("mode", mode);
  test3.AddInput<float>("data", input_dims, input);
  test3.AddInput<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads);
  test3.AddInput<float>("value", {1}, {value});
  test3.AddOutput<float>("output", output_dims, output);
  //TensorRT does not support pads as an input
  test3.Run(expect, error_msg, {kTensorrtExecutionProvider});

#endif
}

// Some of the tests can't run on TensorrtExecutionProvider because only constant mode and value 0 of "Pad" node is supported.
// Those tests will fallback to other EP.

TEST(TensorOpTest, Pad_Spec_Example) {
  RunAllOpsetAllDomainPadTests({3, 2},
                               {1.0f, 1.2f, 2.3f, 3.4f, 4.5f, 5.7f},
                               {0, 2, 0, 0},
                               0.f,
                               {3, 4},
                               {0.0f, 0.0f, 1.0f, 1.2f, 0.0f, 0.0f, 2.3f, 3.4f, 0.0f, 0.0f, 4.5f, 5.7f});
}

TEST(TensorOpTest, Pad_Constant_1D_int) {
  std::vector<int32_t> X = {1, 2, 3, 4, 5, 6};
  int32_t value = 1234;
  std::vector<int32_t> Y = {1234, 1234, 1, 2, 1234, 1234, 3, 4, 1234, 1234, 5, 6};
  RunOpset11TypedTest({3, 2},
                      X,
                      {0, 2, 0, 0},
                      value,
                      {3, 4},
                      Y);
}

TEST(TensorOpTest, Pad_Constant_1D_long) {
  std::vector<int64_t> X = {1, 2, 3, 4, 5, 6};
  int64_t value = 1234;
  std::vector<int64_t> Y = {1234, 1234, 1, 2, 1234, 1234, 3, 4, 1234, 1234, 5, 6};
  RunOpset11TypedTest({3, 2},
                      X,
                      {0, 2, 0, 0},
                      value,
                      {3, 4},
                      Y);
}

TEST(TensorOpTest, Pad_Constant_1D_double) {
  std::vector<double> X = {1., 2., 3., 4., 5., 6.};
  double value = 0.;
  std::vector<double> Y = {0., 0., 1., 2., 0., 0., 3., 4., 0., 0., 5., 6.};
  RunOpset11TypedTest({3, 2},
                      X,
                      {0, 2, 0, 0},
                      value,
                      {3, 4},
                      Y);
}

TEST(TensorOpTest, Pad_Constant_1D) {
  RunAllOpsetAllDomainPadTests({2},
                               {1.0f, 2.0f},
                               {1, 2},
                               1234.f,
                               {5},
                               {1234.0f, 1.0f, 2.0f, 1234.0f, 1234.0f});
}

TEST(TensorOpTest, Pad_Constant_1D_Zero) {
  RunAllOpsetAllDomainPadTests({2},
                               {1.0f, 2.0f},
                               {0, 0},
                               1234.f,
                               {2},
                               {1.0f, 2.0f});
}

TEST(TensorOpTest, Pad_Constant_2D) {
  RunAllOpsetAllDomainPadTests({2, 2},
                               {11.0f, 21.0f,
                                12.0f, 22.0f},
                               {1, 2, 1, 2},
                               1234.f,
                               {4, 6},
                               {1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f,
                                1234.0f, 1234.0f, 11.0f, 21.0f, 1234.0f, 1234.0f,
                                1234.0f, 1234.0f, 12.0f, 22.0f, 1234.0f, 1234.0f,
                                1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f, 1234.0f});
}

TEST(TensorOpTest, Pad_Constant_2D_negative_pads_1) {
  RunAllOpsetAllDomainPadTests({2, 3},
                               {11.0f, 21.0f, 31.0f,
                                12.0f, 22.0f, 32.0f},
                               {1, 2, 1, -1},
                               1234.f,
                               {4, 4},
                               {1234.0f, 1234.0f, 1234.0f, 1234.0f,
                                1234.0f, 1234.0f, 11.0f, 21.0f,
                                1234.0f, 1234.0f, 12.0f, 22.0f,
                                1234.0f, 1234.0f, 1234.0f, 1234.0f});
}

TEST(TensorOpTest, Pad_Constant_2D_negative_pads_2) {
  RunAllOpsetAllDomainPadTests({2, 3},
                               {11.0f, 21.0f, 31.0f,
                                12.0f, 22.0f, 32.0f},
                               {-1, 0, 0, 0},
                               1234.f,
                               {1, 3},
                               {12.0f, 22.0f, 32.0f});
}

TEST(TensorOpTest, Pad_Constant_3D_negative_pads) {
  RunAllOpsetAllDomainPadTests({1, 1, 3},
                               {0.f, 1.0f, 2.f},
                               {0, 0, -1, 0, 0, -1},
                               0.f,
                               {1, 1, 1},
                               {1.f});
}

TEST(TensorOpTest, Pad_Constant_4D_negative_pads) {
  // input_vals contains values from 0 to 99 (inclusive)
  std::vector<float> input_vals;
  input_vals.reserve(100);
  for (int i = 0; i < 100; ++i) {
    input_vals.push_back(static_cast<float>(i));
  }

  // holder for output_vals (expected)
  std::vector<float> output_vals;
  output_vals.reserve(21);

  float seed = 13;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 3; ++j) {
      output_vals.push_back(static_cast<float>(seed + j));
    }
    seed += 10;
  }

  // run tests
  RunAllOpsetAllDomainPadTests({1, 1, 10, 10},
                               input_vals,
                               {0, 0, -1, -3, 0, 0, -2, -4},
                               0.f,
                               {1, 1, 7, 3},
                               output_vals);
}

TEST(TensorOpTest, Pad_3D_complex) {
  RunAllOpsetAllDomainPadTests({2, 2, 2},
                               {111.0f, 112.0f,
                                121.0f, 122.0f,

                                211.0f, 212.0f,
                                221.0f, 222.0f},
                               {1, 0, 0, -1, 0, 0},
                               0.f,
                               {2, 2, 2},
                               {0.0f, 0.0f,
                                0.0f, 0.0f,

                                111.0f, 112.0f,
                                121.0f, 122.0f});
}

TEST(TensorOpTest, Pad_Edge_2D) {
  RunAllOpsetAllDomainPadTests({2, 3},
                               {11.0f, 21.0f, 31.0f,
                                12.0f, 22.0f, 32.0f},
                               {2, 2, 2, 2},
                               0.f,
                               {6, 7},
                               {11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                                11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                                11.0f, 11.0f, 11.0f, 21.0f, 31.0f, 31.0f, 31.0f,
                                12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                                12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f,
                                12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f},
                               "edge");
}

TEST(TensorOpTest, Pad_Edge_3D) {
  RunAllOpsetAllDomainPadTests({1, 2, 3},
                               {11.0f, 21.0f, 31.0f,
                                12.0f, 22.0f, 32.0f},
                               {1, 2, 2, 1, 2, 2},
                               0.f,
                               {3, 6, 7},
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
                                12.0f, 12.0f, 12.0f, 22.0f, 32.0f, 32.0f, 32.0f},
                               "edge");
}

TEST(TensorOpTest, Pad_Reflect_2D) {
  RunAllOpsetAllDomainPadTests({3, 3},
                               {11.0f, 21.0f, 31.0f,
                                12.0f, 22.0f, 32.0f,
                                13.0f, 23.0f, 33.0f},
                               {2, 2, 2, 2},
                               0.f,
                               {7, 7},
                               {33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                                32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                                31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f,
                                32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                                33.0f, 23.0f, 13.0f, 23.0f, 33.0f, 23.0f, 13.0f,
                                32.0f, 22.0f, 12.0f, 22.0f, 32.0f, 22.0f, 12.0f,
                                31.0f, 21.0f, 11.0f, 21.0f, 31.0f, 21.0f, 11.0f},
                               "reflect");
}

TEST(TensorOpTest, Pad_Constant_2D_int) {
  std::vector<int32_t> X = {11, 21, 31,
                            12, 22, 32};
  int32_t value = 0;
  std::vector<int32_t> Y = {11, 11, 11, 21, 31, 31, 31,
                            11, 11, 11, 21, 31, 31, 31,
                            11, 11, 11, 21, 31, 31, 31,
                            12, 12, 12, 22, 32, 32, 32,
                            12, 12, 12, 22, 32, 32, 32,
                            12, 12, 12, 22, 32, 32, 32};
  RunOpset11TypedTest({2, 3},
                      X,
                      {2, 2, 2, 2},
                      value,
                      {6, 7},
                      Y,
                      "edge");
}

TEST(TensorOpTest, Pad_Constant_2D_long) {
  std::vector<int64_t> X = {11, 21, 31,
                            12, 22, 32};
  int64_t value = 0;
  std::vector<int64_t> Y = {11, 11, 11, 21, 31, 31, 31,
                            11, 11, 11, 21, 31, 31, 31,
                            11, 11, 11, 21, 31, 31, 31,
                            12, 12, 12, 22, 32, 32, 32,
                            12, 12, 12, 22, 32, 32, 32,
                            12, 12, 12, 22, 32, 32, 32};
  RunOpset11TypedTest({2, 3},
                      X,
                      {2, 2, 2, 2},
                      value,
                      {6, 7},
                      Y,
                      "edge");
}

TEST(TensorOpTest, Pad_Constant_2D_double) {
  std::vector<double> X = {11., 21., 31.,
                           12., 22., 32.};
  double value = 0.;
  std::vector<double> Y = {11., 11., 11., 21., 31., 31., 31.,
                           11., 11., 11., 21., 31., 31., 31.,
                           11., 11., 11., 21., 31., 31., 31.,
                           12., 12., 12., 22., 32., 32., 32.,
                           12., 12., 12., 22., 32., 32., 32.,
                           12., 12., 12., 22., 32., 32., 32.};
  RunOpset11TypedTest({2, 3},
                      X,
                      {2, 2, 2, 2},
                      value,
                      {6, 7},
                      Y,
                      "edge");
}

/*
Example numpy for testing behavior

import numpy as np

a = np.zeros((2, 0))

b = np.pad(a, 1, 'constant')
print('constant')
print(b)
print(b.shape)

c = np.pad(a, ((1,1),(0,0)), 'reflect')  # allowed if we don't pad the dim with '0'. error otherwise
print('reflect')
print(c)
print(c.shape)

d = np.pad(a, 1, 'edge')
print('edge')
print(d)
print(d.shape)

Output:

constant
[[0. 0.]
 [0. 0.]
 [0. 0.]
 [0. 0.]]
(4, 2)
reflect
[]
(4, 0)
edge
[]
(4, 0)
*/

// test handling of input with a 0 for a dimension
TEST(TensorOpTest, Pad_Constant_DimWithZeroInput) {
  RunAllOpsetAllDomainPadTests({0},  // 1D
                               {},
                               {1, 1},
                               0.1f,
                               {2},
                               {0.1f, 0.1f});

  RunAllOpsetAllDomainPadTests({0},  // 1D empty pads
                               {},
                               {0, 0},
                               0.1f,
                               {0},
                               {});

  RunAllOpsetAllDomainPadTests({0},  // 1D offsetting pads
                               {},
                               {-1, 1},
                               0.1f,
                               {0},
                               {});

  RunAllOpsetAllDomainPadTests({2, 0},  // 2D
                               {},
                               {1, 1, 1, 1},
                               0.1f,
                               {4, 2},
                               {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});

  RunAllOpsetAllDomainPadTests({0, 2},
                               {},
                               {1, 1, 1, 1},
                               0.1f,
                               {2, 4},
                               {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});

  RunAllOpsetAllDomainPadTests({0, 2},
                               {},
                               {1, 0, 1, 0},  // empty pads for dim 1
                               0.1f,
                               {2, 2},
                               {0.1f, 0.1f, 0.1f, 0.1f});

  RunAllOpsetAllDomainPadTests({2, 0, 2},  // 3D
                               {},
                               {0, 1, 0, 0, 1, 0},
                               0.1f,
                               {2, 2, 2},
                               {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});
}

TEST(TensorOpTest, Pad_Edge_DimWithZeroInput) {
  RunAllOpsetAllDomainPadTests({0},  // 1D
                               {},
                               {1, 1},
                               0.1f,
                               {0},
                               {},
                               "edge");

  RunAllOpsetAllDomainPadTests({2, 0},  // 2D
                               {},
                               {1, 1, 1, 1},  // ignore pad for dims with value of 0 as there's no edge value to pad with
                               0.1f,
                               {4, 0},
                               {},
                               "edge");

  RunAllOpsetAllDomainPadTests({2, 2, 0},  // 3D
                               {},
                               {0, 1, 1, 0, 1, 1},
                               0.1f,
                               {2, 4, 0},
                               {},
                               "edge");
}

TEST(TensorOpTest, Pad_Reflect_DimWithZeroInput) {
  RunAllOpsetAllDomainPadTests({2, 0},  // 2D
                               {},
                               {1, 0, 1, 0},  // allowed if it doesn't pad the empty dim
                               0.1f,
                               {4, 0},
                               {},
                               "reflect");

  RunAllOpsetAllDomainPadTests({0, 2, 1},  // 3D
                               {},
                               {1, 1, 1, 1, 1, 1},  // not allowed if it pads the empty dim
                               0.1f,
                               {0, 4, 2},
                               {},
                               "reflect",
                               OpTester::ExpectResult::kExpectFailure,
                               "Cannot use 'reflect' mode to pad dimension with a value of 0. Input shape:{0,2,1}");
}

}  // namespace test
}  // namespace onnxruntime
