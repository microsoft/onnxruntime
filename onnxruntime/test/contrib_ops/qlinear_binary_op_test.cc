// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static std::vector<int64_t> PrefixingDims(const std::vector<int64_t>& dims, size_t number_dims) {
  std::vector<int64_t> prefixed_dims;
  if (number_dims > dims.size()) prefixed_dims.resize(number_dims - dims.size(), 1);
  prefixed_dims.insert(prefixed_dims.end(), dims.begin(), dims.end());
  return prefixed_dims;
}

static int64_t CalcStrides(const std::vector<int64_t>& dims, std::vector<int64_t>& strides, bool clear1 = false) {
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
static T clampi(int a, int min_value, int max_value) {
  return static_cast<T>(std::max(std::min(a, max_value), min_value));
}

template <typename T>
void RunQLinearMathTestFromFloat(
    const char* op_name, std::function<float(float, float)> calc,
    const std::vector<float>& a, const std::vector<int64_t>& a_shape_origin, float A_scale, T A_zero_point,
    const std::vector<float>& b, const std::vector<int64_t>& b_shape_origin, float B_scale, T B_zero_point,
    float C_scale, T C_zero_point,
    bool input_b_is_initializer = false,
    bool all_initializer_scale_zero_point = false) {
  size_t number_dims = std::max(a_shape_origin.size(), b_shape_origin.size());
  std::vector<int64_t> a_shape = PrefixingDims(a_shape_origin, number_dims);
  std::vector<int64_t> b_shape = PrefixingDims(b_shape_origin, number_dims);
  // calc broadcasting shaped
  std::vector<int64_t> c_shape(number_dims, 1);
  for (size_t axis = 0; axis < number_dims; ++axis) {
    if (a_shape[axis] != b_shape[axis] && (a_shape[axis] != 1 && b_shape[axis] != 1)) {
      ORT_THROW("Shapes can not be broadcasted");
    }
    c_shape[axis] = std::max(a_shape[axis], b_shape[axis]);
  }

  std::vector<int64_t> a_strides, b_strides, c_strides;
  auto c_size = CalcStrides(c_shape, c_strides, false);
  auto a_size = CalcStrides(a_shape, a_strides, true);
  auto b_size = CalcStrides(b_shape, b_strides, true);
  if (a_size != static_cast<int64_t>(a.size()) || b_size != static_cast<int64_t>(b.size())) {
    ORT_THROW("Input size not match input shape!");
  }
  constexpr int qmax = std::numeric_limits<T>::max();
  constexpr int qmin = std::numeric_limits<T>::min();

  OpTester test(op_name, 1, onnxruntime::kMSDomain);
  std::vector<T> a_quantized = Quantize<T>(a, A_scale, A_zero_point);
  test.template AddInput<T>("A", a_shape_origin, a_quantized);
  test.AddInput<float>("A_scale", {}, {A_scale}, all_initializer_scale_zero_point);
  test.template AddInput<T>("A_zero_point", {}, {A_zero_point}, all_initializer_scale_zero_point);

  std::vector<T> b_quantized = Quantize<T>(b, B_scale, B_zero_point);
  test.template AddInput<T>("B", b_shape_origin, b_quantized, input_b_is_initializer);
  test.AddInput<float>("B_scale", {}, {B_scale}, all_initializer_scale_zero_point);
  test.template AddInput<T>("B_zero_point", {}, {B_zero_point}, all_initializer_scale_zero_point);

  test.AddInput<float>("C_scale", {}, {C_scale}, all_initializer_scale_zero_point);
  test.template AddInput<T>("C_zero_point", {}, {C_zero_point}, all_initializer_scale_zero_point);
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
    c[offset] = clampi<T>(static_cast<int>(std::nearbyintf(calc(a_dequantized, b_dequantized) / C_scale)) + C_zero_point, qmin, qmax);
  }
  test.template AddOutput<T>("C", c_shape, c);

  test.Run();
}

// total 32 + 31 elements to cover all path
// for add() usage tensor A
static std::vector<float> A4Add = {
    0.00f, 0.25f, 0.50f, 0.75f, 1.00f, 1.25f, 1.50f, 1.75f,
    2.00f, 2.25f, 2.50f, 2.75f, 3.00f, 3.50f, 3.75f, 4.00f,
    -0.00f, -0.25f, -0.50f, -0.75f, -1.00f, -1.25f, -1.50f, -1.75f,
    -2.00f, -2.25f, -2.50f, -2.75f, -3.00f, -4.00f, -3.75f, -3.50f,
    0.00f, 0.25f, 0.50f, 0.75f, 1.00f, 1.25f, 1.50f, 1.75f,
    2.00f, 2.25f, 2.50f, 2.75f, 3.00f, 3.75f, 4.25f, 4.50f,
    -0.00f, -0.25f, -0.50f, -0.75f, -1.00f, -1.25f, -1.50f, -1.75f,
    -2.00f, -2.25f, -2.50f, -2.75f, -3.00f, 3.75f, 3.00f};

// for add() usage tensor B
static std::vector<float> B4Add = {
    4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
    -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
    4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
    -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
    4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
    -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
    4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
    -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -3.75f, -4.00f};

static auto add_function = [](float a_dequantized, float b_dequantized) {
  return a_dequantized + b_dequantized;
};

static auto mul_function = [](float a_dequantized, float b_dequantized) {
  return a_dequantized * b_dequantized;
};

TEST(QLinearBinaryOpTest, AddU8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {63}, B_scale, B_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {63}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {63}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddU8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 3, 7}, A_scale, A_zero_point,
                              B, {3, 1, 7}, B_scale, B_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 3, 7}, A_scale, A_zero_point,
                              B, {3, 1, 7}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 3, 7}, A_scale, A_zero_point,
                              B, {3, 1, 7}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddU8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 8.0f / 256.0f;
  uint8_t C_zero_point = 100;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {1}, B_scale, B_zero_point,
                              A, {63}, A_scale, A_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {1}, B_scale, B_zero_point,
                              A, {63}, A_scale, A_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {1}, B_scale, B_zero_point,
                              A, {63}, A_scale, A_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddU8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 8.0f / 256.0f;
  uint8_t C_zero_point = 100;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {3, 1, 1}, B_scale, B_zero_point,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {3, 1, 1}, B_scale, B_zero_point,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {3, 1, 1}, B_scale, B_zero_point,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddU8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {1}, B_scale, B_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {1}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {1}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddU8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              B, {1, 1, 3}, B_scale, B_zero_point,
                              C_scale, C_zero_point);

  // NNAPI will require all the scales and zero points be initializers
  // We also want to test the case input B is an initializer
  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              B, {1, 1, 3}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              false /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              B, {1, 1, 3}, B_scale, B_zero_point,
                              C_scale, C_zero_point,
                              true /* input_b_is_initializer */,
                              true /* all_initializer_scale_zero_point */);
}

TEST(QLinearBinaryOpTest, AddS8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  int8_t B_zero_point = 0;
  float C_scale = 16.0f / 256.0f;
  int8_t C_zero_point = -16;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {63}, B_scale, B_zero_point,
                              C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, AddS8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  int8_t B_zero_point = 0;
  float C_scale = 16.0f / 256.0f;
  int8_t C_zero_point = -16;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 3, 7}, A_scale, A_zero_point,
                              B, {3, 1, 7}, B_scale, B_zero_point,
                              C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, AddS8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {1}, B_scale, B_zero_point,
                              A, {63}, A_scale, A_zero_point,
                              C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, AddS8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              B, {3, 1, 1}, B_scale, B_zero_point,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, AddS8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {63}, A_scale, A_zero_point,
                              B, {1}, B_scale, B_zero_point,
                              C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, AddS8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
                              A, {3, 7, 3}, A_scale, A_zero_point,
                              B, {1, 1, 3}, B_scale, B_zero_point,
                              C_scale, C_zero_point);
}

//
// Tests for QLinearMul
//
TEST(QLinearBinaryOpTest, MulU8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 64.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_scale, A_zero_point, B, {63}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulU8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 0.00f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 128;
  float C_scale = 64.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 3, 7}, A_scale, A_zero_point, B, {3, 1, 7}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulU8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 8.0f / 256.0f;
  uint8_t C_zero_point = 100;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {1}, B_scale, B_zero_point, A, {63}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulU8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 8.0f / 256.0f;
  uint8_t C_zero_point = 100;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {3, 1, 1}, B_scale, B_zero_point, A, {3, 7, 3}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulU8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_scale, A_zero_point, B, {1}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulU8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  uint8_t A_zero_point = 128;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  uint8_t B_zero_point = 96;
  float C_scale = 16.0f / 256.0f;
  uint8_t C_zero_point = 128;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 7, 3}, A_scale, A_zero_point, B, {1, 1, 3}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  int8_t B_zero_point = 0;
  float C_scale = 64.0f / 256.0f;
  int8_t C_zero_point = -16;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_scale, A_zero_point, B, {63}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  int8_t B_zero_point = 0;
  float C_scale = 16.0f / 256.0f;
  int8_t C_zero_point = -16;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 3, 7}, A_scale, A_zero_point, B, {3, 1, 7}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {1}, B_scale, B_zero_point, A, {63}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {3, 1, 1}, B_scale, B_zero_point, A, {3, 7, 3}, A_scale, A_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_scale, A_zero_point, B, {1}, B_scale, B_zero_point, C_scale, C_zero_point);
}

TEST(QLinearBinaryOpTest, MulS8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  int8_t A_zero_point = 0;
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  int8_t B_zero_point = 16;
  float C_scale = 8.0f / 256.0f;
  int8_t C_zero_point = 10;

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 7, 3}, A_scale, A_zero_point, B, {1, 1, 3}, B_scale, B_zero_point, C_scale, C_zero_point);
}

}  // namespace test
}  // namespace onnxruntime
