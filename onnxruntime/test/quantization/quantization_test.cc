// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr float kEpsilon = 1e-1f;

template <typename T>
void TestQuantizeVectorAndValues(const std::vector<float>& values,
                                 const quantization::Params<T>& params,
                                 const std::vector<T>& expected) {
  std::vector<T> quant_values;
  quant_values.resize(values.size());

  // Pass in std::vector signature first:
  quantization::Quantize(values, quant_values, params);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected[i]);
  }

  // Next check pointer signature variant:
  quantization::Quantize(values.data(), quant_values.data(), params, values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected[i]);
  }
}

template <typename T>
void TestQuantizeLinearVectorAndValues(const std::vector<float>& values,
                                       const float expected_scale,
                                       const T expected_zero_point,
                                       const std::vector<T>& expected_quant_value) {
  std::vector<T> quant_values;
  quant_values.resize(values.size());

  // Pass in std::vector signature first:
  quantization::Params<T> params = quantization::QuantizeLinear(values, quant_values);
  EXPECT_NEAR(params.scale, expected_scale, kEpsilon);
  EXPECT_EQ(params.zero_point, expected_zero_point);
  for (size_t i = 0; i < quant_values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_quant_value[i]);
  }

  // Next check pointer signature variant:
  params = quantization::QuantizeLinear(values.data(), quant_values.data(), values.size());
  EXPECT_NEAR(params.scale, expected_scale, kEpsilon);
  EXPECT_EQ(params.zero_point, expected_zero_point);
  for (size_t i = 0; i < quant_values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_quant_value[i]);
  }
}

}  // namespace

//
// Int8 Tests:
//

TEST(Quantization, QuantizeFloat_Int8) {
  const float x = 1231.34f;

  quantization::Params<int8_t> params;
  params.zero_point = 0;
  params.scale = 0.123f;

  int8_t x_i8 = quantization::Quantize(x, params);
  EXPECT_EQ(x_i8, 127);
}

TEST(Quantization, QuantizeFloatValues_Int8) {
  std::vector<float> values = {-2.4f, 3.9f, 10.2f, 4.1f};

  quantization::Params<int8_t> params;
  params.zero_point = 0;
  params.scale = 0.5f;

  std::vector<int8_t> expected = {-5, 8, 20, 8};

  TestQuantizeVectorAndValues(values, params, expected);
}

TEST(Quantization, QuantizeLinear_Int8) {
  std::vector<float> values = {-3.412f, -12.42f, 1.032f, 2.32f, 9.8212f};
  const float expected_scale = 0.0872204f;
  const int8_t expected_zero_point = 14; 
  std::vector<int8_t> expected_values = {-25, -128, 26, 41, 127};

  TestQuantizeLinearVectorAndValues(values, expected_scale, expected_zero_point, expected_values);
}

TEST(Quantization, Dequantize_Int8) {
  //
  // TODO(kreeger): write me!
  //
}

//
// UInt8 Tests:
//

TEST(Quantization, QuantizeFloat_UInt8) {
  const float x = 1.25f;

  quantization::Params<uint8_t> params;
  params.zero_point = 85;
  params.scale = 0.0117647f;

  uint8_t x_u8 = quantization::Quantize(x, params);
  EXPECT_EQ(x_u8, 191);
}

TEST(Quantization, QuantizeFloatValues_UInt8) {
  std::vector<float> values = {-2.4f, 3.9f, 10.2f, 4.1f};

  quantization::Params<uint8_t> params;
  params.zero_point = 0;
  params.scale = 0.125f;

  std::vector<uint8_t> expected = {0, 31, 82, 33};

  TestQuantizeVectorAndValues(values, params, expected);
}

TEST(Quantization, QuantizeLinear_UInt8) {
  std::vector<float> values = {-3.412f, -12.42f, 1.032f, 2.32f, 9.8212f};
  const float expected_scale = 0.0872203931f;
  const uint8_t expected_zero_point = 142; 
  std::vector<uint8_t> expected_values = {103, 0, 154, 169, 255};

  TestQuantizeLinearVectorAndValues(values, expected_scale, expected_zero_point, expected_values);
}

//
// Invalid test state
//

TEST(Quantization, QuantizeMismatchedVectorSizes) {
  //
  // TODO(kreeger): write me!
  //
}

TEST(Quantization, QuantizeLinearMismatchedVectorSizes) {
  //
  // TODO(kreeger): write me!
  //
}

}  // namespace test
}  // namespace onnxruntime
