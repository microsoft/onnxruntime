// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static std::vector<int64_t> PrefixingDims(const std::vector<int64_t>& dims, size_t number_dims)
{
  std::vector<int64_t> prefixed_dims;
  if (number_dims > dims.size()) prefixed_dims.resize(number_dims - dims.size(), 1);
  prefixed_dims.insert(prefixed_dims.end(), dims.begin(), dims.end());
  return prefixed_dims;
}

static int64_t CalcStrides(const std::vector<int64_t>& dims, std::vector<int64_t>& strides, bool clear1 = false)
{
  strides.clear();
  strides.resize(dims.size(), 1);
  for (int i = (int)dims.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  int64_t len = strides[0] * dims[0];
  if (clear1) {
    for (size_t i = 0, sz = strides.size(); i < sz; ++i) {
      if (dims[i] == 1) strides[i] = 0;
    }
  }
  return len;
}

template <typename T>
void
RunQLinearMathTestFromFloat(
    const char* op_name, std::function<float(float, float)> calc,
    const std::vector<float>& a, const std::vector<int64_t>& a_shape_origin, float A_scale, T A_zero_point,
    const std::vector<float>& b, const std::vector<int64_t>& b_shape_origin, float B_scale, T B_zero_point,
    float C_scale, T C_zero_point)
{
  size_t number_dims = std::max(a_shape_origin.size(), b_shape_origin.size());
  std::vector<int64_t> a_shape = PrefixingDims(a_shape_origin, number_dims);
  std::vector<int64_t> b_shape = PrefixingDims(b_shape_origin, number_dims);
  // calc broadcasting shaped
  std::vector<int64_t> c_shape(number_dims, 1);
  for (size_t axis = 0; axis < number_dims; ++axis) {
    if (a_shape[axis] != b_shape[axis] && (a_shape[axis] != 1 && b_shape[axis] != 1)) {
      throw std::runtime_error("Shapes can not be broadcasted");
    }
    c_shape[axis] = std::max(a_shape[axis], b_shape[axis]);
  }

  std::vector<int64_t> a_strides, b_strides, c_strides;
  auto c_size = CalcStrides(c_shape, c_strides, false);
  auto a_size = CalcStrides(a_shape, a_strides, true);
  auto b_size = CalcStrides(b_shape, b_strides, true);
  if (a_size != static_cast<int64_t>(a.size()) || b_size != static_cast<int64_t>(b.size())){
    throw std::runtime_error("Input size not match input shape!");
  }
  const float qmax = std::numeric_limits<T>::max();
  const float qmin = ((std::numeric_limits<T>::min() == -128) ? -127.0f : static_cast<float>(std::numeric_limits<T>::min()));

  OpTester test(op_name, 1, onnxruntime::kMSDomain);
  std::vector<T> a_quantized(a.size());
  for (size_t i = 0, sz = a.size(); i < sz; ++i) {
    a_quantized[i] = static_cast<T>(clamp(std::round(a[i] / A_scale + static_cast<int>(A_zero_point)), qmin, qmax));
  }
  test.template AddInput<T>("A", a_shape_origin, a_quantized);
  test.AddInput<float>("A_scale", {},  {A_scale});
  test.template AddInput<T>("A_zero_point", {}, {A_zero_point});

  std::vector<T> b_quantized(b.size());
  for (size_t i = 0, sz = b.size(); i < sz; ++i) {
    b_quantized[i] = static_cast<T>(clamp(std::round(b[i] / B_scale + static_cast<int>(B_zero_point)), qmin, qmax));
  }
  test.template AddInput<T>("B", b_shape_origin, b_quantized);
  test.AddInput<float>("B_scale", {}, {B_scale});
  test.template AddInput<T>("B_zero_point", {}, {B_zero_point});

  test.AddInput<float>("C_scale", {}, {C_scale});
  test.template AddInput<T>("C_zero_point", {}, {C_zero_point});
  std::vector<T> c(c_size);
  for (int64_t offset = 0; offset < c_size; ++offset) {
    int64_t remain = offset, a_offset = 0, b_offset = 0;
    for (size_t axis = 0; axis < number_dims; ++axis) {
      int64_t index = remain / c_strides[axis];
      remain = remain % c_strides[axis];
      a_offset += index * a_strides[axis];
      b_offset += index * b_strides[axis];
    }
    float a_dequantized = A_scale * (static_cast<int>(a_quantized[a_offset]) - static_cast<int>(A_zero_point));
    float b_dequantized = B_scale * (static_cast<int>(b_quantized[b_offset]) - static_cast<int>(B_zero_point));
    c[offset] = static_cast<T>(clamp(std::round(calc(a_dequantized, b_dequantized) / C_scale
                                     + static_cast<int>(C_zero_point)), qmin, qmax));
  }
  test.template AddOutput<T>("C", c_shape, c);

  test.Run();
}

TEST(QuantizeLinearContribMathOpTest, AddUInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  float A_scale = 2.0f / 255.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  float B_scale = 4.0f / 255.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 6.0f / 255.0f;
  uint8_t C_zero_point = 128;

  auto add_function = [](float a_dequantized, float b_dequantized) {
    return a_dequantized + b_dequantized;
  };

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    A, {2, 5}, A_scale, A_zero_point, B, {1, 5}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    A, {5, 2}, A_scale, A_zero_point, B, {5, 1}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    B, {5, 1}, B_scale, B_zero_point, A, {5, 2}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QuantizeLinearContribMathOpTest, AddInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  float A_scale = 2.0f / 255.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  float B_scale = 4.0f / 255.0f;
  int8_t B_zero_point = 0;
  float C_scale = 6.0f / 255.0f;
  int8_t C_zero_point = 0;

  auto add_function = [](float a_dequantized, float b_dequantized) {
    return a_dequantized + b_dequantized;
  };

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    A, {2, 5}, A_scale, A_zero_point, B, {1, 5}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    A, {5, 2}, A_scale, A_zero_point, B, {5, 1}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
    B, {5, 1}, B_scale, B_zero_point, A, {5, 2}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QuantizeLinearContribMathOpTest, MulUInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  float A_scale = 2.0f / 255.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  float B_scale = 4.0f / 255.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 4.0f / 255.0f;
  uint8_t C_zero_point = 128;

  auto mul_function = [](float a_dequantized, float b_dequantized) {
    return a_dequantized * b_dequantized;
  };

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    A, {2, 5}, A_scale, A_zero_point, B, {1, 5}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    A, {5, 2}, A_scale, A_zero_point, B, {5, 1}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    B, {5, 1}, B_scale, B_zero_point, A, {5, 2}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QuantizeLinearContribMathOpTest, MulInt8) {
  std::vector<float> A = {0.8f, 0.3f, 0.1f, -0.5f, -0.2f, -0.6f, -0.9f, 0.0f, -1.0f, 1.0f};
  float A_scale = 2.0f / 255.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {-2.0f, -1.0f, 2.0f, 0.3f, 0.9f};
  float B_scale = 4.0f / 255.0f;
  int8_t B_zero_point = 0;
  float C_scale = 4.0f / 255.0f;
  int8_t C_zero_point = 0;

  auto mul_function = [](float a_dequantized, float b_dequantized) {
    return a_dequantized * b_dequantized;
  };

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    A, {2, 5}, A_scale, A_zero_point, B, {1, 5}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    A, {5, 2}, A_scale, A_zero_point, B, {5, 1}, B_scale, B_zero_point, C_scale, C_zero_point);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
    B, {5, 1}, B_scale, B_zero_point, A, {5, 2}, A_scale, A_zero_point, C_scale, C_zero_point);
}

}  // namespace test
}  // namespace onnxruntime

