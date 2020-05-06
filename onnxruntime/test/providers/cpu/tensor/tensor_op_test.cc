// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"


using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

using ExpectResult = OpTester::ExpectResult;

// Some of the tests can't run on TensorrtExecutionProvider because of unsupported data types.
// Those tests will fallback to other EPs.

TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  //TensorRT doesn't support dynamic shape tensor for now
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider, kTensorrtExecutionProvider});  // Nuphar only supports reshape shape from initializer
}

TEST(TensorOpTest, ReshapeWithEmptyDim) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {1, 1, 1}, std::vector<float>(1, 1.0f));
  test.AddInput<int64_t>("shape", {0}, {}, true);
  test.AddOutput<float>("reshaped", {}, std::vector<float>(1, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); // TensorRT doesn't support empty dimension
}

TEST(TensorOpTest, ReshapeWithEmptyInput) {
  OpTester test("Reshape");
  test.AddInput<float>("data", {0, 10}, std::vector<float>());
  test.AddInput<int64_t>("shape", {3}, {0, 10, 1}, false);
  test.AddOutput<float>("reshaped", {0, 10, 1}, std::vector<float>());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); // TensorRT doesn't support empty dimension
}

TEST(TensorOpTest, ReshapeWithEmptyInputAndDynamicShape) {
  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 0, -1}, false);
    test.AddOutput<float>("reshaped", {1, 0, 1}, {});
    test.Run(OpTester::ExpectResult::kExpectFailure, "The input tensor cannot be reshaped to the requested shape", {kTensorrtExecutionProvider}); // TensorRT doesn't support empty dimension
  }

  {
    OpTester test("Reshape");
    test.AddInput<float>("data", {1, 0}, std::vector<float>());
    test.AddInput<int64_t>("shape", {3}, {1, 1, -1}, false);
    test.AddOutput<float>("reshaped", {1, 1, 0}, {});
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}); // TensorRT doesn't support empty dimension
  }
}

TEST(TensorOpTest, ReshapeWithInitializer) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2}, true);
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, ShapeTest2D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<int64_t>("shape", {2}, {2, 3});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: volume of dimensions is not consistent with weights size
}

TEST(TensorOpTest, ShapeTest3D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddOutput<int64_t>("shape", {3}, {2, 3, 4});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT: volume of dimensions is not consistent with weights size
}

template <typename SrcType,
          typename DstType>
void TestCastOp(const std::initializer_list<SrcType>& input,
                const std::initializer_list<DstType>& output,
                const std::vector<int64_t>& dimensions,
                int64_t toType,
                ExpectResult expect_result = ExpectResult::kExpectSuccess,
                const std::string& expected_failure_string = "") {
  OpTester test("Cast", 9);
  test.AddAttribute("to", toType);
  test.AddInput<SrcType>("input", dimensions, input);
  test.AddOutput<DstType>("output", dimensions, output);
  test.Run(expect_result, expected_failure_string, {kTensorrtExecutionProvider});
}

template <typename SrcType>
void TestCastFromSrc() {
  std::initializer_list<SrcType> input_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const std::vector<int64_t> shape{3, 2, 2};

  auto float_output = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  TestCastOp(input_data, float_output, shape, TensorProto::FLOAT);

  auto double_output = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  TestCastOp(input_data, double_output, shape, TensorProto::DOUBLE);

  auto bool_output = {false, true, true, true, true, true, true, true, true, true, true, true};
  TestCastOp(input_data, bool_output, shape, TensorProto::BOOL);

  const std::initializer_list<uint8_t> uint8_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, uint8_t_output, shape, TensorProto::UINT8);

  const std::initializer_list<uint16_t> uint16_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, uint16_t_output, shape, TensorProto::UINT16);

  const std::initializer_list<uint32_t> uint32_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, uint32_t_output, shape, TensorProto::UINT32);

  const std::initializer_list<uint64_t> uint64_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, uint64_t_output, shape, TensorProto::UINT64);

  const std::initializer_list<int16_t> int16_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, int16_t_output, shape, TensorProto::INT16);

  const std::initializer_list<int32_t> int32_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, int32_t_output, shape, TensorProto::INT32);

  const std::initializer_list<int64_t> int64_t_output{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input_data, int64_t_output, shape, TensorProto::INT64);
};

TEST(TensorOpTest, Cast) {
  TestCastFromSrc<float>();
  TestCastFromSrc<double>();
  TestCastFromSrc<uint8_t>();
  TestCastFromSrc<uint16_t>();
  TestCastFromSrc<uint32_t>();
  TestCastFromSrc<uint64_t>();
  TestCastFromSrc<int8_t>();
  TestCastFromSrc<int16_t>();
  TestCastFromSrc<int32_t>();
  TestCastFromSrc<int64_t>();
}

TEST(TensorOpTest, CastFromBool) {
  auto bool_data = {false, true, true, true, true, true, true, true, true, true, false, true};
  const std::vector<int64_t> shape{3, 2, 2};

  const std::initializer_list<float> float_output = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  TestCastOp(bool_data, float_output, shape, TensorProto::FLOAT);

  const std::initializer_list<double> double_output = {0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0};
  TestCastOp(bool_data, double_output, shape, TensorProto::DOUBLE);

  auto bool_output = {false, true, true, true, true, true, true, true, true, true, false, true};
  TestCastOp(bool_data, bool_output, shape, TensorProto::BOOL);

  const std::initializer_list<uint8_t> uint8_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, uint8_t_output, shape, TensorProto::UINT8);

  const std::initializer_list<uint16_t> uint16_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, uint16_t_output, shape, TensorProto::UINT16);

  const std::initializer_list<uint32_t> uint32_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, uint32_t_output, shape, TensorProto::UINT32);

  const std::initializer_list<uint64_t> uint64_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, uint64_t_output, shape, TensorProto::UINT64);

  const std::initializer_list<int16_t> int16_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, int16_t_output, shape, TensorProto::INT16);

  const std::initializer_list<int32_t> int32_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, int32_t_output, shape, TensorProto::INT32);

  const std::initializer_list<int64_t> int64_t_output{0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1};
  TestCastOp(bool_data, int64_t_output, shape, TensorProto::INT64);

  const std::initializer_list<MLFloat16> float16_output{
      MLFloat16(math::floatToHalf(0.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(0.0f)),
      MLFloat16(math::floatToHalf(1.0f))};
  TestCastOp(bool_data, float16_output, shape, TensorProto::FLOAT16);
}

TEST(TensorOpTest, CastToFloat16) {
  const std::vector<int64_t> shape{3, 2, 2};
  std::initializer_list<float> float_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  const std::initializer_list<MLFloat16> float16_output{
      MLFloat16(math::floatToHalf(0.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(2.0f)),
      MLFloat16(math::floatToHalf(3.0f)),
      MLFloat16(math::floatToHalf(4.0f)),
      MLFloat16(math::floatToHalf(5.0f)),
      MLFloat16(math::floatToHalf(6.0f)),
      MLFloat16(math::floatToHalf(7.0f)),
      MLFloat16(math::floatToHalf(8.0f)),
      MLFloat16(math::floatToHalf(9.0f)),
      MLFloat16(math::floatToHalf(10.0f)),
      MLFloat16(math::floatToHalf(11.0f))};

  TestCastOp(float_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<uint8_t> uint8_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(uint8_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<uint16_t> uint16_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(uint16_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<uint32_t> uint32_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(uint32_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<uint64_t> uint64_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(uint64_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<int8_t> int8_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(int8_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<int16_t> int16_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(int16_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<int32_t> int32_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(int32_t_data, float16_output, shape, TensorProto::FLOAT16);

  std::initializer_list<int64_t> int64_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(int64_t_data, float16_output, shape, TensorProto::FLOAT16);
}

TEST(TensorOpTest, CastFromFloat16) {
  const std::vector<int64_t> shape{3, 2, 2};
  const std::initializer_list<float> float_output = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  const std::initializer_list<MLFloat16> input = {
      MLFloat16(math::floatToHalf(0.0f)),
      MLFloat16(math::floatToHalf(1.0f)),
      MLFloat16(math::floatToHalf(2.0f)),
      MLFloat16(math::floatToHalf(3.0f)),
      MLFloat16(math::floatToHalf(4.0f)),
      MLFloat16(math::floatToHalf(5.0f)),
      MLFloat16(math::floatToHalf(6.0f)),
      MLFloat16(math::floatToHalf(7.0f)),
      MLFloat16(math::floatToHalf(8.0f)),
      MLFloat16(math::floatToHalf(9.0f)),
      MLFloat16(math::floatToHalf(10.0f)),
      MLFloat16(math::floatToHalf(11.0f))};

  TestCastOp(input, float_output, shape, TensorProto::FLOAT);

  auto bool_data = {false, true, true, true, true, true, true, true, true, true, true, true};
  TestCastOp(input, bool_data, shape, TensorProto::BOOL);

  std::initializer_list<uint8_t> uint8_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint8_t_data, shape, TensorProto::UINT8);

  std::initializer_list<uint16_t> uint16_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint16_t_data, shape, TensorProto::UINT16);

  std::initializer_list<uint32_t> uint32_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint32_t_data, shape, TensorProto::UINT32);

  std::initializer_list<uint64_t> uint64_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, uint64_t_data, shape, TensorProto::UINT64);

  std::initializer_list<int8_t> int8_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int8_t_data, shape, TensorProto::INT8);

  std::initializer_list<int16_t> int16_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int16_t_data, shape, TensorProto::INT16);

  std::initializer_list<int32_t> int32_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int32_t_data, shape, TensorProto::INT32);

  std::initializer_list<int64_t> int64_t_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  TestCastOp(input, int64_t_data, shape, TensorProto::INT64);
}

TEST(TensorOpTest, CastFromString) {
  const std::vector<int64_t> shape{2, 2, 2};
  std::initializer_list<std::string> string_data = {"-inf", "+INF", "0.9767611f", "0.28280696f",
                                                    "-0.12019656f", "5.0f", "NaN", "nan"};
  const std::initializer_list<float> float_output = {-(std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity(),
                                                     0.9767611f, 0.28280696f,
                                                     -0.12019656f, 5.0f, NAN, NAN};
  TestCastOp(string_data, float_output, shape, TensorProto::FLOAT);

  std::initializer_list<std::string> int_16_string_data = {"0", "1", "2", "3", "4", "5", "-32768", "32767"};
  const std::initializer_list<int16_t> int_16_output = {0, 1, 2, 3, 4, 5, SHRT_MIN, SHRT_MAX};
  TestCastOp(int_16_string_data, int_16_output, shape, TensorProto::INT16);

  std::initializer_list<std::string> int_64_string_data = {"0", "1", "2", "3", "4", "5", "-9223372036854775808", "9223372036854775807"};
  const std::initializer_list<int64_t> int_64_output = {0, 1, 2, 3, 4, 5, LLONG_MIN, LLONG_MAX};
  TestCastOp(int_64_string_data, int_64_output, shape, TensorProto::INT64);
}

TEST(TensorOpTest, CastToString) {
  const std::vector<int64_t> shape{2, 2, 2};
  const std::initializer_list<float> float_input = {NAN, -1.f, 0.0391877927f, 0.296140194f, -0.120196559f, 5.0f,
                                                    -std::numeric_limits<float>::infinity(),
                                                    std::numeric_limits<float>::infinity()};

  // float output precision is 8, so the expected output differs slightly from the input due to that
  std::initializer_list<std::string> string_output = {"NaN", "-1", "0.039187793", "0.29614019",
                                                      "-0.12019656", "5", "-INF", "INF"};
  TestCastOp(float_input, string_output, shape, TensorProto::STRING);

  std::initializer_list<std::string> int_string_data = {"0", "1", "2", "3", "4", "5", "6", "7"};
  const std::initializer_list<int16_t> int_16_input = {0, 1, 2, 3, 4, 5, 6, 7};
  TestCastOp(int_16_input, int_string_data, shape, TensorProto::STRING);
}

void MeanVarianceNormalizationFunctionDefaultPerChannel() {
  const int64_t N = 2, C = 2, H = 2, W = 3;

  std::vector<float> N1C1 = {3.0f, -3.0f, -1.0f,
                             1.0f, 2.0f, -1.0f};
  std::vector<float> N1C2 = {-2.0f, -2.0f, -2.0f,
                             4.0f, 1.0f, 4.0f};
  std::vector<float> N2C1 = {
      0.0f,
      -2.0f,
      -2.0f,
      -4.0f,
      5.0f,
      7.0f,
  };
  std::vector<float> N2C2 = {
      5.0f,
      -5.0f,
      -5.0f,
      3.0f,
      4.0f,
      4.0f,
  };

  std::vector<float> X;
  X.reserve(N * C * H * W);
  X.insert(X.end(), N1C1.begin(), N1C1.end());
  X.insert(X.end(), N1C2.begin(), N1C2.end());
  X.insert(X.end(), N2C1.begin(), N2C1.end());
  X.insert(X.end(), N2C2.begin(), N2C2.end());

  std::vector<float> C1;
  C1.reserve(N * H * W);
  C1.insert(C1.end(), N1C1.begin(), N1C1.end());
  C1.insert(C1.end(), N2C1.begin(), N2C1.end());
  auto C1_meam_stdev = MeanStdev(C1);

  std::vector<float> C2;
  C2.reserve(N * H * W);
  C2.insert(C2.end(), N1C2.begin(), N1C2.end());
  C2.insert(C2.end(), N2C2.begin(), N2C2.end());
  auto C2_meam_stdev = MeanStdev(C2);

  std::vector<float> N1C1_result(N1C1), N1C2_result(N1C2),
      N2C1_result(N2C1), N2C2_result(N2C2);
  Normalize(N1C1_result, C1_meam_stdev, 1);
  Normalize(N2C1_result, C1_meam_stdev, 1);
  Normalize(N1C2_result, C2_meam_stdev, 1);
  Normalize(N2C2_result, C2_meam_stdev, 1);

  std::vector<float> result;
  result.reserve(N * C * H * W);
  result.insert(result.end(), N1C1_result.begin(), N1C1_result.end());
  result.insert(result.end(), N1C2_result.begin(), N1C2_result.end());
  result.insert(result.end(), N2C1_result.begin(), N2C1_result.end());
  result.insert(result.end(), N2C2_result.begin(), N2C2_result.end());

  OpTester test("MeanVarianceNormalization", 9);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

void MeanVarianceNormalizationFunctionAcrossChannels(std::vector<int64_t> axes) {
  const int64_t N = 2, C = 2, H = 2, W = 3;

  std::vector<float> X = {3.0f, -3.0f, -1.0f,
                          1.0f, 2.0f, -1.0f,
                          -2.0f, -2.0f, -2.0f,
                          4.0f, 1.0f, 4.0f,
                          0.0f, -2.0f, -2.0f,
                          -4.0f, 5.0f, 7.0f,
                          5.0f, -5.0f, -5.0f,
                          3.0f, 4.0f, 4.0f};
  auto mean_stdev = MeanStdev(X);

  std::vector<float> result(X);
  Normalize(result, mean_stdev, 1);

  OpTester test("MeanVarianceNormalization", 9);
  test.AddAttribute("axes", axes);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

TEST(TensorOpTest, MeanVarianceNormalizationCPUTest) {
  // axes: {0, 1, 2, 3} for across_channels
  MeanVarianceNormalizationFunctionAcrossChannels({0, 1, 2, 3});

  // Default (axes: {0, 2, 3}) for non across_channels
  MeanVarianceNormalizationFunctionDefaultPerChannel();
}

}  // namespace test
}  // namespace onnxruntime
