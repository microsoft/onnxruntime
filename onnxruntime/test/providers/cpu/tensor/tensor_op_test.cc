// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/tensor/crop.h"
#include "core/util/math.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

using ExpectResult = OpTester::ExpectResult;

TEST(TensorOpTest, Reshape) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2});
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider});  // Nuphar only supports reshape shape from initializer
}

TEST(TensorOpTest, ReshapeWithEmptyDim) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {1, 1, 1}, std::vector<float>(1, 1.0f));
  test.AddInput<int64_t>("shape", {0}, {}, true);
  test.AddOutput<float>("reshaped", {}, std::vector<float>(1, 1.0f));
  test.Run();
}

TEST(TensorOpTest, ReshapeWithInitializer) {
  OpTester test("Reshape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddInput<int64_t>("shape", {3}, {-1, 0, 2}, true);
  test.AddOutput<float>("reshaped", {1, 3, 2}, std::vector<float>(6, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Identity) {
  OpTester test("Identity");
  std::vector<float> X{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  test.AddInput<float>("input", {2, 3}, X);
  test.AddOutput<float>("output", {2, 3}, X);
  test.Run();
}

TEST(TensorOpTest, IdentityString) {
  OpTester test("Identity");
  std::vector<std::string> X{"this", "is", "a", "test", "for", "identity"};
  test.AddInput<std::string>("input", {2, 3}, X);
  test.AddOutput<std::string>("output", {2, 3}, X);
  test.Run();
}

TEST(TensorOpTest, ShapeTest2D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<int64_t>("shape", {2}, {2, 3});
  test.Run();
}

TEST(TensorOpTest, ShapeTest3D) {
  OpTester test("Shape");

  test.AddInput<float>("data", {2, 3, 4}, std::vector<float>(24, 1.0f));
  test.AddOutput<int64_t>("shape", {3}, {2, 3, 4});
  test.Run();
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
  test.Run(expect_result, expected_failure_string);
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
  std::initializer_list<std::string> string_data = {"-inf", "+INF", "2.0f", "3.0f", "4.0f", "5.0f", "NaN", "nan"};
  const std::initializer_list<float> float_output = {-(std::numeric_limits<float>::infinity()), std::numeric_limits<float>::infinity(), 2.0f, 3.0f, 4.0f, 5.0f, NAN, NAN};
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
  const std::initializer_list<float> float_input = {NAN, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
  std::initializer_list<std::string> string_output = {"NaN", "1", "2", "3", "4", "5", "-INF", "INF"};
  TestCastOp(float_input, string_output, shape, TensorProto::STRING);

  std::initializer_list<std::string> int_string_data = {"0", "1", "2", "3", "4", "5", "6", "7"};
  const std::initializer_list<int16_t> int_16_input = {0, 1, 2, 3, 4, 5, 6, 7};
  TestCastOp(int_16_input, int_string_data, shape, TensorProto::STRING);
}

#ifndef DISABLE_CONTRIB_OPS
TEST(TensorOpTest, CropBorderOnly) {
  const int N = 2, C = 1, H = 3, W = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                          2.0f, 3.0f, 4.0f, 5.0f,
                          3.0f, 4.0f, 5.0f, 6.0f,

                          4.0f, 5.0f, 6.0f, 7.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          6.0f, 7.0f, 8.0f, 9.0f};

  const std::vector<int64_t> border{0, 1, 2, 1};
  std::vector<float> output = {
      2.0f, 3.0f,

      5.0f, 6.0f};

  OpTester test("Crop");
  test.AddAttribute("border", border);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, (H - border[2] - border[0]), (W - border[3] - border[1])}, output);
  test.Run();
}

TEST(TensorOpTest, CropBorderAndScale) {
  const int N = 2, C = 1, H = 3, W = 4;
  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f,
                          2.0f, 3.0f, 4.0f, 5.0f,
                          3.0f, 4.0f, 5.0f, 6.0f,

                          4.0f, 5.0f, 6.0f, 7.0f,
                          5.0f, 6.0f, 7.0f, 8.0f,
                          6.0f, 7.0f, 8.0f, 9.0f};

  const std::vector<int64_t> border = {0, 0, 0, 0};
  const std::vector<int64_t> scale = {2, 2};

  std::vector<float> output = {
      1.0f, 2.0f,
      2.0f, 3.0f,

      4.0f, 5.0f,
      5.0f, 6.0f};

  OpTester test("Crop");
  test.AddAttribute("border", border);
  test.AddAttribute("scale", scale);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, scale[0], scale[1]}, output);
  test.Run();
}
#endif

std::pair<float, float> MeanStdev(std::vector<float>& v) {
  float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  float mean = sum / v.size();

  std::vector<float> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(),
                 std::bind(std::minus<float>(), std::placeholders::_1, mean));
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f);
  float stdev = std::sqrt(sq_sum / v.size());

  return std::make_pair(mean, stdev);
}

void Normalize(std::vector<float>& v,
               std::pair<float, float>& mean_stdev, bool normalize_variance) {
  float mean = mean_stdev.first;
  float stdev = mean_stdev.second;

  std::transform(v.begin(), v.end(), v.begin(),
                 std::bind(std::minus<float>(), std::placeholders::_1, mean));

  if (normalize_variance) {
    std::transform(v.begin(), v.end(), v.begin(),
                   std::bind(std::divides<float>(), std::placeholders::_1, stdev));
  }
}

void MeanVarianceNormalizationAcrossChannels(bool across_channels, bool normalize_variance) {
  const int64_t N = 2, C = 2, H = 2, W = 3;
  int64_t one = 1;
  int64_t zero = 0;

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
  Normalize(result, mean_stdev, normalize_variance);

  OpTester test("MeanVarianceNormalization", 7);
  test.AddAttribute("across_channels", across_channels ? one : zero);
  test.AddAttribute("normalize_variance", normalize_variance ? one : zero);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

void MeanVarianceNormalizationPerChannel(bool across_channels, bool normalize_variance) {
  const int64_t N = 2, C = 2, H = 2, W = 3;
  int64_t one = 1;
  int64_t zero = 0;

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
  Normalize(N1C1_result, C1_meam_stdev, normalize_variance);
  Normalize(N2C1_result, C1_meam_stdev, normalize_variance);
  Normalize(N1C2_result, C2_meam_stdev, normalize_variance);
  Normalize(N2C2_result, C2_meam_stdev, normalize_variance);

  std::vector<float> result;
  result.reserve(N * C * H * W);
  result.insert(result.end(), N1C1_result.begin(), N1C1_result.end());
  result.insert(result.end(), N1C2_result.begin(), N1C2_result.end());
  result.insert(result.end(), N2C1_result.begin(), N2C1_result.end());
  result.insert(result.end(), N2C2_result.begin(), N2C2_result.end());

  OpTester test("MeanVarianceNormalization", 7);
  test.AddAttribute("across_channels", across_channels ? one : zero);
  test.AddAttribute("normalize_variance", normalize_variance ? one : zero);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
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
  // across_channels: true, normalize_variance: true
  MeanVarianceNormalizationAcrossChannels(true, true);

  // across_channels: true, normalize_variance: false
  MeanVarianceNormalizationAcrossChannels(true, false);

  // across_channels: false, normalize_variance: false
  MeanVarianceNormalizationPerChannel(false, false);

  // across_channels: false, normalize_variance: true
  MeanVarianceNormalizationPerChannel(false, true);

  // axes: {0, 1, 2, 3} for across_channels
  MeanVarianceNormalizationFunctionAcrossChannels({0, 1, 2, 3});

  // Default (axes: {0, 2, 3}) for non across_channels
  MeanVarianceNormalizationFunctionDefaultPerChannel();
}

#ifndef DISABLE_CONTRIB_OPS
TEST(TensorOpTest, ImageScalerTest) {
  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {
      1.0f, 3.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      7.0f, 9.0f};

  float scale = 2.0f;
  std::vector<float> bias = {1.0f, 2.0f};

  std::vector<float> result = {
      3.0f, 7.0f,
      7.0f, 11.0f,

      8.0f, 12.0f,
      16.0f, 20.0f};

  OpTester test("ImageScaler");
  test.AddAttribute("scale", scale);
  test.AddAttribute("bias", bias);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}
#endif
}  // namespace test
}  // namespace onnxruntime
