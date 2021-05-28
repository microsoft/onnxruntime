// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T, int opset>
static void RunOnnxOpsetTypedTest(
    const std::vector<int64_t>& input_dims,
    const std::vector<T>& input,
    const std::vector<int64_t>& pads,
    T value,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output,
    std::string mode = "constant",
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& error_msg = "") {
  // ONNX domain opset
  OpTester test("Pad", opset);
  if (mode != "constant")
    test.AddAttribute("mode", mode);
  test.AddInput<T>("data", input_dims, input);
  if (opset >= 11) {
    test.AddInput<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads);
    test.AddInput<T>("value", {1}, {value});
  } else {
    test.AddAttribute("pads", pads);
    test.AddAttribute("value", static_cast<float>(value));
  }
  test.AddOutput<T>("output", output_dims, output);
  if (opset >= 11) {
    // TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
    test.Run(expect, error_msg, {kTensorrtExecutionProvider});
  } else {
#if defined(OPENVINO_CONFIG_MYRIAD) || defined(OPENVINO_CONFIG_VAD_M)
    test.Run(expect, error_msg, {kOpenVINOExecutionProvider});
#else
    test.Run(expect, error_msg);
#endif
  }
}

template <typename T>
static void RunAllOpsetAllDomainPadTests(
    const std::vector<int64_t>& input_dims,
    const std::vector<T>& input,
    const std::vector<int64_t>& pads,
    T value,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output,
    std::string mode = "constant",
    OpTester::ExpectResult expect = OpTester::ExpectResult::kExpectSuccess,
    const std::string& error_msg = "") {
  // Test opset-11 and opset-13 kernels of Pad
  RunOnnxOpsetTypedTest<T, 11>(input_dims,
                               input,
                               pads,
                               value,
                               output_dims,
                               output,
                               mode, expect, error_msg);

  RunOnnxOpsetTypedTest<T, 13>(input_dims,
                               input,
                               pads,
                               value,
                               output_dims,
                               output,
                               mode, expect, error_msg);
}
template <>
void RunAllOpsetAllDomainPadTests<>(
    const std::vector<int64_t>& input_dims,
    const std::vector<double>& input,
    const std::vector<int64_t>& pads,
    double value,
    const std::vector<int64_t>& output_dims,
    const std::vector<double>& output,
    std::string mode,
    OpTester::ExpectResult expect,
    const std::string& error_msg) {
  // Test opset-10, opset-11 and opset-13 kernels of Pad (for double type)
  RunOnnxOpsetTypedTest<double, 10>(input_dims,
                                    input,
                                    pads,
                                    value,
                                    output_dims,
                                    output,
                                    mode, expect, error_msg);

  RunOnnxOpsetTypedTest<double, 11>(input_dims,
                                    input,
                                    pads,
                                    value,
                                    output_dims,
                                    output,
                                    mode, expect, error_msg);

  RunOnnxOpsetTypedTest<double, 13>(input_dims,
                                    input,
                                    pads,
                                    value,
                                    output_dims,
                                    output,
                                    mode, expect, error_msg);
}

// There is only support for float type for MSDomain kernel in ORT
template <>
void RunAllOpsetAllDomainPadTests<>(
    const std::vector<int64_t>& input_dims,
    const std::vector<float>& input,
    const std::vector<int64_t>& pads,
    float value,
    const std::vector<int64_t>& output_dims,
    const std::vector<float>& output,
    std::string mode,
    OpTester::ExpectResult expect,
    const std::string& error_msg) {
  // Test opset-10, opset-11 and opset-13 kernels of Pad (for float type)
  RunOnnxOpsetTypedTest<float, 10>(input_dims,
                                   input,
                                   pads,
                                   value,
                                   output_dims,
                                   output,
                                   mode, expect, error_msg);

  RunOnnxOpsetTypedTest<float, 11>(input_dims,
                                   input,
                                   pads,
                                   value,
                                   output_dims,
                                   output,
                                   mode, expect, error_msg);

  RunOnnxOpsetTypedTest<float, 13>(input_dims,
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
  test3.Run(expect, error_msg, {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});

#endif
}

// Some of the tests can't run on TensorrtExecutionProvider because only constant mode and value 0 of "Pad" node is supported.
// Those tests will fallback to other EP.

using PadTypes = ::testing::Types<float, double, int8_t, int32_t, int64_t, uint8_t, uint32_t, uint64_t>;

template <typename T>
class PadOpTest : public ::testing::Test {
};
TYPED_TEST_SUITE(PadOpTest, PadTypes);

TYPED_TEST(PadOpTest, Pad_Spec_Example) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2},
                                  {T(1), T(2), T(3), T(4), T(5), T(6)},
                                  {0, 2, 0, 0},
                                  T(0),
                                  {3, 4},
                                  {T(0), T(0), T(1), T(2), T(0), T(0), T(3), T(4), T(0), T(0), T(5), T(6)});
}

TYPED_TEST(PadOpTest, Pad_Constant_1D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2},
                                  {T(1), T(2)},
                                  {1, 2},
                                  T(123),
                                  {5},
                                  {T(123), T(1), T(2), T(123), T(123)});
}

TYPED_TEST(PadOpTest, Pad_Constant_1D_Zero) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2},
                                  {T(1), T(2)},
                                  {0, 0},
                                  T(123),
                                  {2},
                                  {T(1), T(2)});
}

TYPED_TEST(PadOpTest, Pad_Reflect_1D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2},
                                  {T(1), T(2), T(3), T(4), T(5), T(6)},
                                  {0, 1, 0, 1},
                                  T(0),
                                  {3, 4},
                                  {T(2), T(1), T(2), T(1), T(4), T(3), T(4), T(3), T(6), T(5), T(6), T(5)},
                                  "reflect");
}

TYPED_TEST(PadOpTest, Pad_Edge_1D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2},
                                  {T(1), T(2), T(3), T(4), T(5), T(6)},
                                  {0, 2, 0, 1},
                                  T(0),
                                  {3, 5},
                                  {T(1), T(1), T(1), T(2), T(2), T(3), T(3), T(3), T(4), T(4), T(5), T(5), T(5), T(6), T(6)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Constant_2D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 2},
                                  {T(11), T(21),
                                   T(12), T(22)},
                                  {1, 2, 1, 2},
                                  T(123),
                                  {4, 6},
                                  {T(123), T(123), T(123), T(123), T(123), T(123),
                                   T(123), T(123), T(11), T(21), T(123), T(123),
                                   T(123), T(123), T(12), T(22), T(123), T(123),
                                   T(123), T(123), T(123), T(123), T(123), T(123)});
}

TYPED_TEST(PadOpTest, Pad_Constant_2D_negative_pads_1) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3},
                                  {T(11), T(21), T(31),
                                   T(12), T(22), T(32)},
                                  {1, 2, 1, -1},
                                  T(123),
                                  {4, 4},
                                  {T(123), T(123), T(123), T(123),
                                   T(123), T(123), T(11), T(21),
                                   T(123), T(123), T(12), T(22),
                                   T(123), T(123), T(123), T(123)});
}

TYPED_TEST(PadOpTest, Pad_Constant_2D_negative_pads_2) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3},
                                  {T(11), T(21), T(31),
                                   T(12), T(22), T(32)},
                                  {-1, 0, 0, 0},
                                  T(123),
                                  {1, 3},
                                  {T(12), T(22), T(32)});
}

TYPED_TEST(PadOpTest, Pad_Constant_3D_negative_pads) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({1, 1, 3},
                                  {T(0), T(1), T(2)},
                                  {0, 0, -1, 0, 0, -1},
                                  T(0),
                                  {1, 1, 1},
                                  {T(1)});
}

TYPED_TEST(PadOpTest, Pad_Constant_4D_negative_pads) {
  using T = TypeParam;
  // input_vals contains values from 0 to 99 (inclusive)
  std::vector<T> input_vals;
  input_vals.reserve(100);
  for (int i = 0; i < 100; ++i) {
    input_vals.push_back(T(i));
  }

  // holder for output_vals (expected)
  std::vector<T> output_vals;
  output_vals.reserve(21);

  int seed = 13;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 3; ++j) {
      output_vals.push_back(T(seed + j));
    }
    seed += 10;
  }

  // run tests
  RunAllOpsetAllDomainPadTests<T>({1, 1, 10, 10},
                                  input_vals,
                                  {0, 0, -1, -3, 0, 0, -2, -4},
                                  T(0),
                                  {1, 1, 7, 3},
                                  output_vals);
}

TYPED_TEST(PadOpTest, Pad_3D_complex) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 2, 2},
                                  {T(11), T(12),
                                   T(21), T(22),

                                   T(111), T(112),
                                   T(121), T(122)},
                                  {1, 0, 0, -1, 0, 0},
                                  T(0),
                                  {2, 2, 2},
                                  {T(0), T(0),
                                   T(0), T(0),

                                   T(11), T(12),
                                   T(21), T(22)});
}

TYPED_TEST(PadOpTest, Pad_Edge_2D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3},
                                  {T(11), T(21), T(31),
                                   T(12), T(22), T(32)},
                                  {2, 2, 2, 2},
                                  T(0),
                                  {6, 7},
                                  {T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Edge_3D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({1, 2, 3},
                                  {T(11), T(21), T(31),
                                   T(12), T(22), T(32)},
                                  {1, 2, 2, 1, 2, 2},
                                  T(0),
                                  {3, 6, 7},
                                  {T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),

                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),

                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(11), T(11), T(11), T(21), T(31), T(31), T(31),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32),
                                   T(12), T(12), T(12), T(22), T(32), T(32), T(32)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Reflect_2D) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 3},
                                  {T(11), T(21), T(31),
                                   T(12), T(22), T(32),
                                   T(13), T(23), T(33)},
                                  {2, 2, 2, 2},
                                  T(0),
                                  {7, 7},
                                  {T(33), T(23), T(13), T(23), T(33), T(23), T(13),
                                   T(32), T(22), T(12), T(22), T(32), T(22), T(12),
                                   T(31), T(21), T(11), T(21), T(31), T(21), T(11),
                                   T(32), T(22), T(12), T(22), T(32), T(22), T(12),
                                   T(33), T(23), T(13), T(23), T(33), T(23), T(13),
                                   T(32), T(22), T(12), T(22), T(32), T(22), T(12),
                                   T(31), T(21), T(11), T(21), T(31), T(21), T(11)},
                                  "reflect");
}

TYPED_TEST(PadOpTest, Pad_Constant_3D_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, 1, 0, 1, 1, 0},
                                  T(31),
                                  {5, 4, 5},
                                  {T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31),

                                   T(31), T(31), T(31), T(31), T(31),
                                   T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(31), T(31), T(31), T(31), T(31),

                                   T(31), T(31), T(31), T(31), T(31),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(31), T(31), T(31), T(31), T(31),

                                   T(31), T(31), T(31), T(31), T(31),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(31), T(31), T(31), T(31), T(31),

                                   T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31),
                                   T(31), T(31), T(31), T(31), T(31)},
                                  "constant");
}

TYPED_TEST(PadOpTest, Pad_Edge_3D_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, 1, 0, 1, 1, 0},
                                  T(0),
                                  {5, 4, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(6), T(7), T(8), T(9), T(10),

                                   T(1), T(2), T(3), T(4), T(5),
                                   T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(6), T(7), T(8), T(9), T(10),

                                   T(11), T(12), T(13), T(14), T(15),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(16), T(17), T(18), T(19), T(20),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(26), T(27), T(28), T(29), T(30),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Edge_3D_Last_Pad_Slice_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, -1, 0, 1, 1, 0},
                                  T(0),
                                  {5, 2, 5},
                                  {T(6), T(7), T(8), T(9), T(10),
                                   T(6), T(7), T(8), T(9), T(10),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(6), T(7), T(8), T(9), T(10),

                                   T(16), T(17), T(18), T(19), T(20),
                                   T(16), T(17), T(18), T(19), T(20),

                                   T(26), T(27), T(28), T(29), T(30),
                                   T(26), T(27), T(28), T(29), T(30),

                                   T(26), T(27), T(28), T(29), T(30),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Edge_3D_Last_Slice_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, -1, 0, 1, 0, 0},
                                  T(0),
                                  {4, 2, 5},
                                  {T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Reflect_3D_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({3, 2, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, 1, 0, 1, 1, 0},
                                  T(0),
                                  {5, 4, 5},
                                  {T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(1), T(2), T(3), T(4), T(5),

                                   T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15),

                                   T(26), T(27), T(28), T(29), T(30),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(21), T(22), T(23), T(24), T(25),

                                   T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(11), T(12), T(13), T(14), T(15)},
                                  "reflect");
}

TYPED_TEST(PadOpTest, Pad_Reflect_3D_Last_Pad_Slice_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, -1, 0, 1, 1, 0},
                                  T(0),
                                  {4, 3, 5},
                                  {T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(21), T(22), T(23), T(24), T(25),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(6), T(7), T(8), T(9), T(10),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),
                                   T(21), T(22), T(23), T(24), T(25),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(6), T(7), T(8), T(9), T(10)},
                                  "reflect");
}

TYPED_TEST(PadOpTest, Pad_Reflect_3D_Last_Slice_Inner_No_Padding) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 3, 5},
                                  {T(1), T(2), T(3), T(4), T(5),
                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),
                                   T(16), T(17), T(18), T(19), T(20),
                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30)},
                                  {1, -1, 0, 1, 0, 0},
                                  T(0),
                                  {4, 2, 5},
                                  {T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15),

                                   T(21), T(22), T(23), T(24), T(25),
                                   T(26), T(27), T(28), T(29), T(30),

                                   T(6), T(7), T(8), T(9), T(10),
                                   T(11), T(12), T(13), T(14), T(15)},
                                  "reflect");
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
TYPED_TEST(PadOpTest, Pad_Constant_DimWithZeroInput) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({0},  // 1D
                                  {},
                                  {1, 1},
                                  T(1),
                                  {2},
                                  {T(1), T(1)});

  RunAllOpsetAllDomainPadTests<T>({0},  // 1D empty pads
                                  {},
                                  {0, 0},
                                  T(1),
                                  {0},
                                  {});

  RunAllOpsetAllDomainPadTests<T>({0},  // 1D offsetting pads
                                  {},
                                  {-1, 1},
                                  T(1),
                                  {0},
                                  {});

  RunAllOpsetAllDomainPadTests<T>({2, 0},  // 2D
                                  {},
                                  {1, 1, 1, 1},
                                  T(1),
                                  {4, 2},
                                  {T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1)});

  RunAllOpsetAllDomainPadTests<T>({0, 2},
                                  {},
                                  {1, 1, 1, 1},
                                  T(1),
                                  {2, 4},
                                  {T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1)});

  RunAllOpsetAllDomainPadTests<T>({0, 2},
                                  {},
                                  {1, 0, 1, 0},  // empty pads for dim 1
                                  T(1),
                                  {2, 2},
                                  {T(1), T(1), T(1), T(1)});

  RunAllOpsetAllDomainPadTests<T>({2, 0, 2},  // 3D
                                  {},
                                  {0, 1, 0, 0, 1, 0},
                                  T(1),
                                  {2, 2, 2},
                                  {T(1), T(1), T(1), T(1), T(1), T(1), T(1), T(1)});
}
// Added output shape verification b/w the output shape generated by operator specific ONNX inference and
// the output shape generated by operator specific ORT implementation. After adding this verification,
// this test logs warning as validation fails for 2 data types out of 8 data types i.e. Float and Double.
// Reason:
//  Pad ORT implementation output shape does not match with Pad ONNX inference function output shape.
//
// For Float and Double this test gets executed for 2 different opset version, 10 and 11. Specifically this
// test is failing for opset version 10.
//  Investigation Analysis: Different ONNX inference class/method gets executed per opset version. Main difference b/w the 2
//          pad operator ONNX inference class/method is:
//              Older Pad operator ONNX inference: Accepts "pads and values" as attribute.
//              Newer Pad operator ONNX inference: Accetps "pads and values" as input.
//          For newer version, "pads & values" fields have not been added as initializer, thus instead of shape
//          inference, rank inference gets triggered. Whereas, in older version shape inference gets executed
//          as "pads & values" fields have been added as attribute.
//      In order to remove the warning, shape inference methods needs to be fixed.

TYPED_TEST(PadOpTest, Pad_Edge_DimWithZeroInput) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({0},  // 1D
                                  {},
                                  {1, 1},
                                  T(1),
                                  {0},
                                  {},
                                  "edge");

  RunAllOpsetAllDomainPadTests<T>({2, 0},  // 2D
                                  {},
                                  {1, 1, 1, 1},  // ignore pad for dims with value of 0 as there's no edge value to pad with
                                  T(1),
                                  {4, 0},
                                  {},
                                  "edge");

  RunAllOpsetAllDomainPadTests<T>({2, 2, 0},  // 3D
                                  {},
                                  {0, 1, 1, 0, 1, 1},
                                  T(1),
                                  {2, 4, 0},
                                  {},
                                  "edge");
}

TYPED_TEST(PadOpTest, Pad_Reflect_DimWithZeroInput) {
  using T = TypeParam;
  RunAllOpsetAllDomainPadTests<T>({2, 0},  // 2D
                                  {},
                                  {1, 0, 1, 0},  // allowed if it doesn't pad the empty dim
                                  T(1),
                                  {4, 0},
                                  {},
                                  "reflect");

  RunAllOpsetAllDomainPadTests<T>({0, 2, 1},  // 3D
                                  {},
                                  {1, 1, 1, 1, 1, 1},  // not allowed if it pads the empty dim
                                  T(1),
                                  {0, 4, 2},
                                  {},
                                  "reflect",
                                  OpTester::ExpectResult::kExpectFailure,
                                  "Cannot use 'reflect' mode to pad dimension with a value of 0. Input shape:{0,2,1}");
}

TEST(PadOpTest, BoolType) {
  OpTester test("Pad", 13);
  test.AddAttribute("mode", "constant");
  test.AddInput<bool>("data", {3, 2}, {true, false, true, false, true, false});
  test.AddInput<int64_t>("pads", {4}, {0, 2, 0, 0});
  test.AddInput<bool>("value", {1}, {true});
  test.AddOutput<bool>("output", {3, 4}, {true, true, true, false, true, true, true, false, true, true, true, false});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
