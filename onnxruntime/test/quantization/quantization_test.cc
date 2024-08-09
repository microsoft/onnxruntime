// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "core/quantization/quantization.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr float kEpsilon = 1e-1f;

template <typename T>
void TestQuantizeVectorAndValues(const std::vector<float>& values,
                                 const quantization::Params<T>& params,
                                 const std::vector<T>& expected_values) {
  ORT_ENFORCE(values.size() == expected_values.size(),
              "Input values and expected values must have the same length.");

  std::vector<T> quant_values;
  quant_values.resize(values.size());

  // Pass in std::vector signature first:
  quantization::Quantize(values, quant_values, params);
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_values[i]);
  }

  // Next check pointer signature variant:
  quantization::Quantize(values.data(),
                         quant_values.data(),
                         params,
                         values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_values[i]);
  }
}

template <typename T>
void TestQuantizeLinearVectorAndValues(const std::vector<float>& values,
                                       const float expected_scale,
                                       const T expected_zero_point,
                                       const std::vector<T>& expected_values) {
  ORT_ENFORCE(values.size() == expected_values.size(),
              "Input values and expected values must have the same length.");

  std::vector<T> quant_values;
  quant_values.resize(values.size());

  // Pass in std::vector signature first:
  quantization::Params<T> params = quantization::QuantizeLinear(
      values, quant_values);
  EXPECT_NEAR(params.scale, expected_scale, kEpsilon);
  EXPECT_EQ(params.zero_point, expected_zero_point);
  for (size_t i = 0; i < quant_values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_values[i]);
  }

  // Next check pointer signature variant:
  params = quantization::QuantizeLinear(
      values.data(), quant_values.data(), values.size());
  EXPECT_NEAR(params.scale, expected_scale, kEpsilon);
  EXPECT_EQ(params.zero_point, expected_zero_point);
  for (size_t i = 0; i < quant_values.size(); ++i) {
    EXPECT_EQ(quant_values[i], expected_values[i]);
  }
}

template <typename T>
void TestDequantizeVectorAndValues(const std::vector<T>& values,
                                   const quantization::Params<T>& params,
                                   const std::vector<float>& expected_values) {
  ORT_ENFORCE(values.size() == expected_values.size(),
              "Input values and expected values must have the same length.");
  std::vector<float> dequant_values;
  dequant_values.resize(values.size());

  // Pass in std::vector signature first:
  quantization::Dequantize(values, dequant_values, params);
  for (size_t i = 0; i < dequant_values.size(); ++i) {
    EXPECT_NEAR(dequant_values[i], expected_values[i], kEpsilon);
  }

  // Next check pointer signature variant:
  quantization::Dequantize(
      values.data(), dequant_values.data(), params, values.size());
  for (size_t i = 0; i < dequant_values.size(); ++i) {
    EXPECT_NEAR(dequant_values[i], expected_values[i], kEpsilon);
  }
}

template <typename T>
void EnsureQuantizedTensorParam(const float scale, const T zero_point) {
  TensorShape shape({1});

  // First, create the scale tensor:
  auto alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  IAllocatorUniquePtr<float> buffer = IAllocator::MakeUniquePtr<float>(alloc, shape.Size());
  float* float_data = buffer.get();
  float_data[0] = scale;
  Tensor scale_tensor(DataTypeImpl::GetType<float>(),
                      shape,
                      float_data,
                      alloc->Info(),
                      /*offset=*/0);

  // Next, create the zero_point tensor:
  IAllocatorUniquePtr<T> buffer2 = IAllocator::MakeUniquePtr<T>(alloc, shape.Size());
  T* typed_data = buffer2.get();
  typed_data[0] = zero_point;
  Tensor zero_point_tensor(DataTypeImpl::GetType<T>(),
                           shape,
                           typed_data,
                           alloc->Info(),
                           /*offset=*/0);

  quantization::Params<T> params =
      quantization::GetTensorQuantizationParams<T>(&scale_tensor,
                                                   &zero_point_tensor);

  EXPECT_EQ(params.scale, scale);
  EXPECT_EQ(params.zero_point, zero_point);
}

}  // namespace

TEST(Quantization, CreateQuantizationParamsFromTensors) {
  EnsureQuantizedTensorParam<int8_t>(0.134f, -2);
  EnsureQuantizedTensorParam<uint8_t>(0.0123f, 141);
}

//
// Int8 Tests:
//

TEST(Quantization, QuantizeFloat_Int8) {
  constexpr float x = 2.34f;
  quantization::Params<int8_t> params(/*scale=*/0.0872204f, /*zero_point=*/0);
  EXPECT_EQ(quantization::Quantize(x, params), 27);
}

TEST(Quantization, QuantizeFloatValues_Int8) {
  std::vector<float> values = {-2.4f, 3.9f, 10.2f, 4.1f};
  quantization::Params<int8_t> params(/*scale=*/0.5f, /*zero_point=*/0);
  std::vector<int8_t> expected = {-5, 8, 20, 8};
  TestQuantizeVectorAndValues(values, params, expected);
}

TEST(Quantization, QuantizeLinear_Int8) {
  std::vector<float> values = {-3.412f, -12.42f, 1.032f, 2.32f, 9.8212f};
  constexpr float expected_scale = 0.0872204f;
  constexpr int8_t expected_zero_point = 15;
  std::vector<int8_t> expected_values = {-24, -127, 27, 41, 127};

  TestQuantizeLinearVectorAndValues(values,
                                    expected_scale,
                                    expected_zero_point,
                                    expected_values);
}

TEST(Quantization, Dequantize_Int8) {
  constexpr int8_t x_i8 = 12;
  quantization::Params<int8_t> params(/*scale=*/0.0124f, /*zero_point=*/-1);
  EXPECT_NEAR(quantization::Dequantize(x_i8, params), 0.1612f, kEpsilon);
}

TEST(Quantization, DequantizeValues_Int8) {
  std::vector<int8_t> values = {-100, 23, 117, 2, -10};
  quantization::Params<int8_t> params(/*scale=*/0.0124f, /*zero_point=*/-1);
  std::vector<float> expected_values = {
      -1.22759998f, 0.297600001f, 1.46320009f, 0.0372000001f, -0.111600004f};

  TestDequantizeVectorAndValues(values, params, expected_values);
}

//
// UInt8 Tests:
//

TEST(Quantization, QuantizeFloat_UInt8) {
  constexpr float x = 1.25f;
  quantization::Params<uint8_t> params(/*scale=*/0.0117647f, /*zero_point=*/85);
  EXPECT_EQ(quantization::Quantize(x, params), 191);
}

TEST(Quantization, QuantizeFloatValues_UInt8) {
  std::vector<float> values = {-2.4f, 3.9f, 10.2f, 4.1f};
  quantization::Params<uint8_t> params(/*scale=*/0.125f, /*zero_point=*/0);
  std::vector<uint8_t> expected = {0, 31, 82, 33};

  TestQuantizeVectorAndValues(values, params, expected);
}

TEST(Quantization, QuantizeLinear_UInt8) {
  std::vector<float> values = {-3.412f, -12.42f, 1.032f, 2.32f, 9.8212f};
  constexpr float expected_scale = 0.0872203931f;
  constexpr uint8_t expected_zero_point = 142;
  std::vector<uint8_t> expected_values = {103, 0, 154, 169, 255};

  TestQuantizeLinearVectorAndValues(values,
                                    expected_scale,
                                    expected_zero_point,
                                    expected_values);
}

TEST(Quantization, Dequantize_UInt8) {
  constexpr uint8_t x_u8 = 200;
  quantization::Params<uint8_t> params(/*scale=*/0.0124f, /*zero_point=*/127);
  EXPECT_NEAR(quantization::Dequantize(x_u8, params), 0.9052f, kEpsilon);
}

TEST(Quantization, DequantizeValues_UInt8) {
  std::vector<uint8_t> values = {100, 223, 11, 153, 85};
  quantization::Params<uint8_t> params(/*scale=*/0.0124f, /*zero_point=*/128);
  std::vector<float> expected_values = {
      -0.347200006f, 1.17800009f, -1.45080006f, 0.310000002f, -0.533200026f};

  TestDequantizeVectorAndValues(values, params, expected_values);
}

//
// Invalid test state
//

#if !defined(ORT_NO_EXCEPTIONS)
TEST(Quantization, QuantizeMismatchedVectorSizes) {
  std::vector<float> values = {1.0f, 2.0f, 3.0f};
  std::vector<int8_t> quant_values;
  quant_values.resize(values.size() - 1);

  quantization::Params<int8_t> params(/*scale=*/0.125f, /*zero_point=*/0);

  EXPECT_THROW(quantization::Quantize(values, quant_values, params),
               OnnxRuntimeException);
}

TEST(Quantization, QuantizeLinearMismatchedVectorSizes) {
  std::vector<float> values = {1.0f, 2.0f, 3.0f};
  std::vector<int8_t> quant_values;
  quant_values.resize(values.size() - 1);

  EXPECT_THROW(quantization::QuantizeLinear(values, quant_values),
               OnnxRuntimeException);
}
#endif
}  // namespace test
}  // namespace onnxruntime
