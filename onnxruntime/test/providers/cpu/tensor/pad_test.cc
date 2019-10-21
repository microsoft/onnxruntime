// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTypedTest(
    const std::vector<int64_t>& input_dims,
    const std::vector<T>& input,
    const std::vector<int64_t>& pads,
    T value,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output,
    std::string mode = "constant") {

  // ONNX domain opset-11
  OpTester test("Pad", 11);
  if (mode != "constant") test.AddAttribute("mode", mode);
  test.AddInput<T>("data", input_dims, input);
  test.AddInput<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads);
  test.AddInput<T>("value", {1}, {value});
  test.AddOutput<T>("output", output_dims, output);
  // NGraph and TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider, kTensorrtExecutionProvider});
}

static void RunTest(
    const std::vector<int64_t>& input_dims,
    const std::vector<float>& input,
    const std::vector<int64_t>& pads,
    float value,
    const std::vector<int64_t>& output_dims,
    const std::vector<float>& output,
    std::string mode = "constant",
    const int opset_version = 10) {

  if ( opset_version <= 10){
	  // ONNX domain opset-10
	  OpTester test1("Pad", 10);
	  test1.AddInput<float>("data", input_dims, input);
	  if (mode != "constant") test1.AddAttribute("mode", mode);
	  test1.AddAttribute("pads", pads);
	  test1.AddAttribute("value", value);
	  test1.AddOutput<float>("output", output_dims, output);
	  test1.Run();
  }

  // ONNX domain opset-11
		RunTypedTest(input_dims,
								input,
								pads,
								value,
								output_dims,
								output,
								mode);

  #ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test3("Pad", 1, kMSDomain);
  if (mode != "constant") test3.AddAttribute("mode", mode);
  test3.AddInput<float>("data", input_dims, input);
  test3.AddInput<int64_t>("pads", {static_cast<int64_t>(pads.size())}, pads);
  test3.AddInput<float>("value", {1}, {value});
  test3.AddOutput<float>("output", output_dims, output);
  //TensorRT does not support pads as an input
  test3.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  #endif
}

// Some of the tests can't run on TensorrtExecutionProvider because only constant mode and value 0 of "Pad" node is supported.
// Those tests will fallback to other EP.

TEST(TensorOpTest, Pad_Spec_Example) {
  RunTest({3, 2},
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
	RunTypedTest({3, 2},
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
  RunTypedTest({3, 2},
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
	RunTypedTest({3, 2},
          X,
          {0, 2, 0, 0},
          value,
          {3, 4},
          Y);
}

TEST(TensorOpTest, Pad_Constant_1D) {
  RunTest({2},
          {1.0f, 2.0f},
          {1, 2},
          1234.f,
          {5},
          {1234.0f, 1.0f, 2.0f, 1234.0f, 1234.0f});
}

TEST(TensorOpTest, Pad_Constant_1D_Zero) {
  RunTest({2},
          {1.0f, 2.0f},
          {0, 0},
          1234.f,
          {2},
          {1.0f, 2.0f});
}

TEST(TensorOpTest, Pad_Constant_2D) {
  RunTest({2, 2},
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

TEST(TensorOpTest, Pad_Constant_2D_negative) {
  RunTest({2, 3},
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

TEST(TensorOpTest, Pad_3D_complex) {
  RunTest({2, 2, 2},
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
  RunTest({2, 3},
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
  RunTest({1, 2, 3},
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
  RunTest({3, 3},
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
	RunTypedTest({2, 3},
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
  RunTypedTest({2, 3},
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
	RunTypedTest({2, 3},
          X,
          {2, 2, 2, 2},
          value,
          {6, 7},
          Y,
          "edge");
}

}  // namespace test
}  // namespace onnxruntime
