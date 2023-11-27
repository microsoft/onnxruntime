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
    const std::vector<float>& a, const std::vector<int64_t>& a_shape_origin,
    const quantization::Params<T>& a_params,
    const std::vector<float>& b, const std::vector<int64_t>& b_shape_origin,
    const quantization::Params<T>& b_params,
    const quantization::Params<T>& c_params) {
  const auto run_test = [&](bool input_b_is_initializer,
                            bool all_initializer_scale_zero_point) {
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
    std::vector<T> a_quantized = QuantizeTestVector<T>(a, a_params);
    test.template AddInput<T>("A", a_shape_origin, a_quantized);
    test.AddInput<float>("A_scale", {}, {a_params.scale}, all_initializer_scale_zero_point);
    test.template AddInput<T>("A_zero_point", {}, {a_params.zero_point}, all_initializer_scale_zero_point);

    std::vector<T> b_quantized = QuantizeTestVector<T>(b, b_params);
    test.template AddInput<T>("B", b_shape_origin, b_quantized, input_b_is_initializer);
    test.AddInput<float>("B_scale", {}, {b_params.scale}, all_initializer_scale_zero_point);
    test.template AddInput<T>("B_zero_point", {}, {b_params.zero_point}, all_initializer_scale_zero_point);

    test.AddInput<float>("C_scale", {}, {c_params.scale}, all_initializer_scale_zero_point);
    test.template AddInput<T>("C_zero_point", {}, {c_params.zero_point}, all_initializer_scale_zero_point);
    std::vector<T> c(c_size);
    for (int64_t offset = 0; offset < c_size; ++offset) {
      int64_t remain = offset, a_offset = 0, b_offset = 0;
      for (size_t axis = 0; axis < number_dims; ++axis) {
        int64_t index = remain / c_strides[axis];
        remain = remain % c_strides[axis];
        a_offset += index * a_strides[axis];
        b_offset += index * b_strides[axis];
      }

      float a_dequantized = quantization::Dequantize(a_quantized[a_offset], a_params);
      float b_dequantized = quantization::Dequantize(b_quantized[b_offset], b_params);
      c[offset] = clampi<T>(static_cast<int>(std::nearbyintf(calc(a_dequantized, b_dequantized) / c_params.scale)) + c_params.zero_point, qmin, qmax);
    }

    float abs_error = 0.0f;

    // For quantized models, NNAPI's rounding is different than CPU provider
    // Sometimes the result is within +/-1 of result of CPU provider
    // For ONNX, we use rounding to nearest ties to even.
    // For NNAPI, it is using std::round which is HALF_AWAY_FROM_ZERO, see
    // https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/master/nn/common/operations/Quantize.cpp
    // Use 1 as abs_error which is the smallest possbile for uint8_t
    //
    // NOTE, for now the tolerance will only apply if the NNAPI is actually used,
    // if for any reason the execution falls back to CPU, we still expect an exact match
    // See, 'void Check<uint8_t>(...' in onnxruntime/test/providers/provider_test_utils.cc
#ifdef USE_NNAPI
    abs_error = 1.0f;
#endif

    test.template AddOutput<T>("C", c_shape, c, false /* sort_output */, 0.0f /* rel_error */, abs_error);

    test.Run();
  };

  run_test(false /* input_b_is_initializer */, false /* all_initializer_scale_zero_point */);

  // NNAPI will require all the scales and zero points be initializers
  run_test(false /* input_b_is_initializer */, true /* all_initializer_scale_zero_point */);

  // We also want to test the case input B is an initializer
  run_test(true /* input_b_is_initializer */, true /* all_initializer_scale_zero_point */);
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
//
//TEST(QLinearBinaryOpTest, AddU8VectorVectorFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  const std::vector<float>& B(B4Add);
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/128);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {63}, A_params,
//                              B, {63}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddU8VectorVectorBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  std::vector<float> B = {
//      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
//      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
//      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/128);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {3, 3, 7}, A_params,
//                              B, {3, 1, 7}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddU8ScalarVectorFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  std::vector<float> B = {0.25f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/100);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              B, {1}, B_params,
//                              A, {63}, A_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddU8ScalarVectorBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  std::vector<float> B = {0.25f, -0.25f, -0.00f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/100);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              B, {3, 1, 1}, B_params,
//                              A, {3, 7, 3}, A_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddU8VectorScalarFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  std::vector<float> B = {0.25f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {63}, A_params,
//                              B, {1}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddU8VectorScalarBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
//  std::vector<float> B = {0.25f, -0.25f, -0.00f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {3, 7, 3}, A_params,
//                              B, {1, 1, 3}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8VectorVectorFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  const std::vector<float>& B(B4Add);
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/0);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/-16);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {63}, A_params,
//                              B, {63}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8VectorVectorBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  std::vector<float> B = {
//      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
//      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
//      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
//  float B_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/0);
//  float C_scale = 16.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/-16);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {3, 3, 7}, A_params,
//                              B, {3, 1, 7}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8ScalarVectorFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  std::vector<float> B = {0.25f};
//  float B_scale = 2.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              B, {1}, B_params,
//                              A, {63}, A_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8ScalarVectorBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  std::vector<float> B = {0.25f, -0.25f, -0.00f};
//  float B_scale = 2.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              B, {3, 1, 1}, B_params,
//                              A, {3, 7, 3}, A_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8VectorScalarFull) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  std::vector<float> B = {0.25f};
//  float B_scale = 2.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {63}, A_params,
//                              B, {1}, B_params,
//                              C_params);
//}
//
//TEST(QLinearBinaryOpTest, AddS8VectorScalarBroadcast) {
//  const std::vector<float>& A(A4Add);
//  float A_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
//  std::vector<float> B = {0.25f, -0.25f, -0.00f};
//  float B_scale = 2.0f / 256.0f;
//  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
//  float C_scale = 8.0f / 256.0f;
//  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);
//
//  RunQLinearMathTestFromFloat("QLinearAdd", add_function,
//                              A, {3, 7, 3}, A_params,
//                              B, {1, 1, 3}, B_params,
//                              C_params);
//}
//
//
// Tests for QLinearMul
//
TEST(QLinearBinaryOpTest, MulU8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/128);
  float C_scale = 64.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_params,
                              B, {63}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulU8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 0.00f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/128);
  float C_scale = 64.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 3, 7}, A_params,
                              B, {3, 1, 7}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulU8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/100);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {1}, B_params,
                              A, {63}, A_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulU8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/100);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {3, 1, 1}, B_params,
                              A, {3, 7, 3}, A_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulU8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  std::vector<float> B = {0.25f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
  float C_scale = 16.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_params,
                              B, {1}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulU8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> A_params(A_scale, /*zero_point=*/128);
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<uint8_t> B_params(B_scale, /*zero_point=*/96);
  float C_scale = 16.0f / 256.0f;
  quantization::Params<uint8_t> C_params(C_scale, /*zero_point=*/128);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 7, 3}, A_params,
                              B, {1, 1, 3}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8VectorVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  const std::vector<float>& B(B4Add);
  float B_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/0);
  float C_scale = 64.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/-16);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_params,
                              B, {63}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8VectorVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  std::vector<float> B = {
      4.00f, 0.25f, 0.00f, -0.25f, 0.50f, -0.25f, -0.00f, 0.25f,
      -1.50f, -2.25f, 2.50f, 3.75f, -3.75f, -4.00f, 5.00f, 5.50f,
      -0.50f, -1.25f, 0.75f, 1.25f, 2.25f};
  float B_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/0);
  float C_scale = 16.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/-16);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 3, 7}, A_params,
                              B, {3, 1, 7}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8ScalarVectorFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {1}, B_params,
                              A, {63}, A_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8ScalarVectorBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              B, {3, 1, 1}, B_params,
                              A, {3, 7, 3}, A_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8VectorScalarFull) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  std::vector<float> B = {0.25f};
  float B_scale = 2.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {63}, A_params,
                              B, {1}, B_params,
                              C_params);
}

TEST(QLinearBinaryOpTest, MulS8VectorScalarBroadcast) {
  const std::vector<float>& A(A4Add);
  float A_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> A_params(A_scale, /*zero_point=*/0);
  std::vector<float> B = {0.25f, -0.25f, -0.00f};
  float B_scale = 2.0f / 256.0f;
  quantization::Params<int8_t> B_params(B_scale, /*zero_point=*/16);
  float C_scale = 8.0f / 256.0f;
  quantization::Params<int8_t> C_params(C_scale, /*zero_point=*/10);

  RunQLinearMathTestFromFloat("QLinearMul", mul_function,
                              A, {3, 7, 3}, A_params,
                              B, {1, 1, 3}, B_params,
                              C_params);
}

}  // namespace test
}  // namespace onnxruntime
