// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/test_environment.h"
#include "test/unittest_util/conversion.h"
#include "default_providers.h"

#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"

namespace onnxruntime {
namespace test {

class DynamicQuantMatMulFp8SessionTester : public OpTester {
 public:
  using BaseTester::ExecuteModel;
  using BaseTester::FillFeedsAndOutputNames;
  using BaseTester::SetTestFunctionCalled;
  using OpTester::BuildModel;
  using OpTester::OpTester;
};

template <typename Fp8T>
struct Fp8TensorProtoType;

template <>
struct Fp8TensorProtoType<Float8E4M3FN> {
  static constexpr int64_t value = ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN;
};

template <>
struct Fp8TensorProtoType<Float8E4M3FNUZ> {
  static constexpr int64_t value = ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FNUZ;
};

template <>
struct Fp8TensorProtoType<Float8E5M2> {
  static constexpr int64_t value = ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2;
};

template <>
struct Fp8TensorProtoType<Float8E5M2FNUZ> {
  static constexpr int64_t value = ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E5M2FNUZ;
};

template <typename Fp8T>
float Fp8MaxAbs();

template <>
float Fp8MaxAbs<Float8E4M3FN>() {
  return 448.0f;
}

template <>
float Fp8MaxAbs<Float8E4M3FNUZ>() {
  return 448.0f;
}

template <>
float Fp8MaxAbs<Float8E5M2>() {
  return 57344.0f;
}

template <>
float Fp8MaxAbs<Float8E5M2FNUZ>() {
  return 57344.0f;
}

template <typename Fp8T>
float QuantizeDequantize(float value, float scale) {
  return Fp8T(value / scale, true).ToFloat() * scale;
}

template <typename Fp8T>
std::vector<float> ComputeRowwiseAQuantized(const std::vector<float>& a_data,
                                            int64_t m,
                                            int64_t k,
                                            int64_t block_size_k) {
  const int64_t blocks_k = k / block_size_k;
  std::vector<float> result(a_data.size());
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t block_k = 0; block_k < blocks_k; ++block_k) {
      const int64_t k_begin = block_k * block_size_k;
      const int64_t k_end = k_begin + block_size_k;
      float max_abs = 0.0f;
      for (int64_t kk = k_begin; kk < k_end; ++kk) {
        max_abs = std::max(max_abs, std::fabs(a_data[static_cast<size_t>(row * k + kk)]));
      }
      const float scale = max_abs == 0.0f ? 1.0f : max_abs / Fp8MaxAbs<Fp8T>();
      for (int64_t kk = k_begin; kk < k_end; ++kk) {
        const size_t index = static_cast<size_t>(row * k + kk);
        result[index] = QuantizeDequantize<Fp8T>(a_data[index], scale);
      }
    }
  }
  return result;
}

template <typename Fp8T>
std::vector<float> ComputeConstantBQuantized(const std::vector<float>& b_data,
                                             int64_t k,
                                             int64_t n,
                                             int64_t block_size_k,
                                             int64_t block_size_n) {
  const int64_t blocks_k = k / block_size_k;
  const int64_t blocks_n = n / block_size_n;
  std::vector<float> result(b_data.size());
  for (int64_t block_n = 0; block_n < blocks_n; ++block_n) {
    const int64_t n_begin = block_n * block_size_n;
    const int64_t n_end = n_begin + block_size_n;
    for (int64_t block_k = 0; block_k < blocks_k; ++block_k) {
      const int64_t k_begin = block_k * block_size_k;
      const int64_t k_end = k_begin + block_size_k;
      float max_abs = 0.0f;
      for (int64_t kk = k_begin; kk < k_end; ++kk) {
        for (int64_t nn = n_begin; nn < n_end; ++nn) {
          max_abs = std::max(max_abs, std::fabs(b_data[static_cast<size_t>(kk * n + nn)]));
        }
      }
      const float scale = max_abs == 0.0f ? 1.0f : max_abs / Fp8MaxAbs<Fp8T>();
      for (int64_t kk = k_begin; kk < k_end; ++kk) {
        for (int64_t nn = n_begin; nn < n_end; ++nn) {
          const size_t index = static_cast<size_t>(kk * n + nn);
          result[index] = QuantizeDequantize<Fp8T>(b_data[index], scale);
        }
      }
    }
  }
  return result;
}

template <typename Fp8T>
std::vector<float> ComputeRuntimeBQuantized(const std::vector<Fp8T>& b_data,
                                            const std::vector<float>& b_scale,
                                            int64_t k,
                                            int64_t n,
                                            int64_t block_size_k,
                                            int64_t block_size_n) {
  const int64_t blocks_k = k / block_size_k;
  std::vector<float> result(b_data.size());
  for (int64_t kk = 0; kk < k; ++kk) {
    const int64_t block_k = kk / block_size_k;
    for (int64_t nn = 0; nn < n; ++nn) {
      const int64_t block_n = nn / block_size_n;
      const size_t index = static_cast<size_t>(kk * n + nn);
      result[index] = b_data[index].ToFloat() * b_scale[static_cast<size_t>(block_n * blocks_k + block_k)];
    }
  }
  return result;
}

std::vector<float> ComputeMatMul(const std::vector<float>& a_data,
                                 const std::vector<float>& b_data,
                                 int64_t m,
                                 int64_t n,
                                 int64_t k,
                                 float y_scale = 1.0f) {
  std::vector<float> y_data(static_cast<size_t>(m * n), 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        sum += a_data[static_cast<size_t>(row * k + kk)] * b_data[static_cast<size_t>(kk * n + col)];
      }
      y_data[static_cast<size_t>(row * n + col)] = sum * y_scale;
    }
  }
  return y_data;
}

template <typename Fp8T>
void AddZeroPoint(OpTester& test, const char* name, const std::vector<int64_t>& shape, size_t count, bool initializer) {
  test.AddInput<Fp8T>(name, shape, std::vector<Fp8T>(count, Fp8T(0.0f)), initializer);
}

template <typename Fp8T>
void RunConstantBInputs() {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const auto a_quantized = ComputeRowwiseAQuantized<Fp8T>(a_data, M, K, 128);
  const auto b_quantized = ComputeConstantBQuantized<Fp8T>(b_data, K, N, 128, 128);
  const auto y_data = ComputeMatMul(a_quantized, b_quantized, M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("fp8_type", Fp8TensorProtoType<Fp8T>::value);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

template <typename Fp8T>
void RunRuntimeFp8BInput() {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Fp8T> b_data(static_cast<size_t>(K * N), Fp8T(-0.5f));
  std::vector<float> b_scale{1.0f};
  const auto a_quantized = ComputeRowwiseAQuantized<Fp8T>(a_data, M, K, 128);
  const auto b_quantized = ComputeRuntimeBQuantized(b_data, b_scale, K, N, 128, 128);
  const auto y_data = ComputeMatMul(a_quantized, b_quantized, M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Fp8T>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  AddZeroPoint<Fp8T>(test, "B_zero_point", {N / 128, K / 128}, b_scale.size(), false);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

void RunDynamicQuantMatMulFp8WithSharedPrepack(DynamicQuantMatMulFp8SessionTester& test,
                                               OrtValue& shared_b,
                                               PrepackedWeightsContainer& prepacked_weights_container,
                                               size_t& shared_prepack_count) {
  test.SetTestFunctionCalled();

  auto& model = test.BuildModel();
  Status status = model.MainGraph().Resolve();
  ASSERT_TRUE(status.IsOK()) << status;

  std::unordered_map<std::string, OrtValue> feeds;
  std::vector<std::string> output_names;
  test.FillFeedsAndOutputNames(feeds, output_names);

  SessionOptions so;
  status = so.AddInitializer("B", &shared_b);
  ASSERT_TRUE(status.IsOK()) << status;

  InferenceSession session{so, GetEnvironment()};
  status = session.AddPrePackedWeightsContainer(&prepacked_weights_container);
  ASSERT_TRUE(status.IsOK()) << status;

  status = session.RegisterExecutionProvider(DefaultCpuExecutionProvider());
  ASSERT_TRUE(status.IsOK()) << status;

  test.ExecuteModel(model,
                    session,
                    OpTester::ExpectResult::kExpectSuccess,
                    "",
                    nullptr,
                    feeds,
                    output_names,
                    kCpuExecutionProvider);
  shared_prepack_count = session.GetSessionState().GetUsedSharedPrePackedWeightCounter();
}

TEST(DynamicQuantMatMulFp8, WithConstantBInputs) {
  RunConstantBInputs<Float8E4M3FN>();
}

TEST(DynamicQuantMatMulFp8, WithConstantBInputsE4M3FNUZ) {
  RunConstantBInputs<Float8E4M3FNUZ>();
}

TEST(DynamicQuantMatMulFp8, WithConstantBInputsE5M2) {
  RunConstantBInputs<Float8E5M2>();
}

TEST(DynamicQuantMatMulFp8, WithConstantBInputsE5M2FNUZ) {
  RunConstantBInputs<Float8E5M2FNUZ>();
}

TEST(DynamicQuantMatMulFp8, RuntimeFp8BInput) {
  RunRuntimeFp8BInput<Float8E4M3FN>();
}

TEST(DynamicQuantMatMulFp8, RuntimeFp8BInputE4M3FNUZ) {
  RunRuntimeFp8BInput<Float8E4M3FNUZ>();
}

TEST(DynamicQuantMatMulFp8, RuntimeFp8BInputE5M2) {
  RunRuntimeFp8BInput<Float8E5M2>();
}

TEST(DynamicQuantMatMulFp8, RuntimeFp8BInputE5M2FNUZ) {
  RunRuntimeFp8BInput<Float8E5M2FNUZ>();
}

TEST(DynamicQuantMatMulFp8, WithOmittedOutputQuantizationInputs) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, 128),
                                    ComputeRuntimeBQuantized(b_data, b_scale, K, N, 128, 128), M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {N / 128, K / 128}, b_scale.size(), false);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, WithOnlyYScale) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;
  constexpr float YScale = 0.5f;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  std::vector<float> y_scale{YScale};
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, 128),
                                    ComputeRuntimeBQuantized(b_data, b_scale, K, N, 128, 128), M, N, K, YScale);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {N / 128, K / 128}, b_scale.size(), false);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, RejectsNonZeroYZeroPoint) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(1.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {N / 128, K / 128}, b_scale.size(), false);
  test.AddOptionalInputEdge<float>();
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 supports symmetric quantization only; Y zero point values must be zero.");
}

TEST(DynamicQuantMatMulFp8, WithRuntimeBInputsBf16Scales) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale_float{1.0f};
  std::vector<BFloat16> b_scale = MakeBFloat16({1.0f});
  std::vector<BFloat16> y_scale = MakeBFloat16({1.0f});
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, 128),
                                    ComputeRuntimeBQuantized(b_data, b_scale_float, K, N, 128, 128), M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<BFloat16>("B_scale", {N / 128, K / 128}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {N / 128, K / 128}, b_scale.size(), false);
  test.AddInput<BFloat16>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, Float16Output) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, 128),
                                    ComputeConstantBQuantized<Float8E4M3FN>(b_data, K, N, 128, 128), M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("A", {M, K}, FloatsToMLFloat16s(a_data));
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<MLFloat16>("Y", {M, N}, FloatsToMLFloat16s(y_data));
  test.Run();
}

TEST(DynamicQuantMatMulFp8, BFloat16Output) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, 128),
                                    ComputeConstantBQuantized<Float8E4M3FN>(b_data, K, N, 128, 128), M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<BFloat16>("A", {M, K}, FloatsToBFloat16s(a_data));
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<BFloat16>("Y", {M, N}, FloatsToBFloat16s(y_data));
  test.Run();
}

TEST(DynamicQuantMatMulFp8, RejectsNonConstantB) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires runtime B input to be FP8.");
}

TEST(DynamicQuantMatMulFp8, RejectsRuntimeFp8BZeroPointTypeMismatch) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E5M2> b_zp{Float8E5M2(0.0f)};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  test.AddInput<Float8E5M2>("B_zero_point", {N / 128, K / 128}, b_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires B and B zero point FP8 types to match.");
}

TEST(DynamicQuantMatMulFp8, RejectsConstantFp8BZeroPointTypeMismatch) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E5M2> b_zp{Float8E5M2(0.0f)};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  test.AddInput<Float8E5M2>("B_zero_point", {N / 128, K / 128}, b_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires B and B zero point FP8 types to match.");
}

TEST(DynamicQuantMatMulFp8, RejectsNonZeroBZeroPoint) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(1.0f)};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {N / 128, K / 128}, b_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 supports symmetric quantization only; B zero point values must be zero.");
}

TEST(DynamicQuantMatMulFp8, NonDefaultBlockSizesWithPartialM) {
  constexpr int64_t M = 9;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;
  constexpr int64_t BlockK = 2;
  constexpr int64_t BlockN = 2;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      a_data[static_cast<size_t>(m * K + k)] = static_cast<float>((m + 1) * (k + 1)) / 16.0f;
    }
  }

  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.0f));
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      b_data[static_cast<size_t>(k * N + n)] = Float8E4M3FN(k == n ? 1.0f : 0.0f);
    }
  }
  std::vector<float> b_scale{1.0f, 2.0f,
                             3.0f, 4.0f};
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, BlockK),
                                    ComputeRuntimeBQuantized(b_data, b_scale, K, N, BlockK, BlockN), M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_k", BlockK);
  test.AddAttribute<int64_t>("block_size_n", BlockN);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / BlockN, K / BlockK}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {N / BlockN, K / BlockK}, b_scale.size(), false);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, SharedPrepackedWeightsRestoresPackedBMetadata) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;
  constexpr int64_t BlockSize = 2;

  std::vector<float> a_data{
      1.0f, 2.0f, 3.0f, 4.0f,
      -1.0f, -2.0f, -3.0f, -4.0f,
      0.5f, 1.0f, -0.5f, -1.0f,
      4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<float> b_data{
      2.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 2.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 2.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 2.0f};
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E5M2>(a_data, M, K, BlockSize),
                                    ComputeConstantBQuantized<Float8E5M2>(b_data, K, N, BlockSize, BlockSize),
                                    M, N, K);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("fp8_type", Fp8TensorProtoType<Float8E5M2>::value);
  test.AddAttribute<int64_t>("block_size_k", BlockSize);
  test.AddAttribute<int64_t>("block_size_n", BlockSize);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 1e-5f);

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({K, N}), b_data.data(),
                       OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  SessionOptions so;
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());

  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t prepack_count_session_1 = 0;
  size_t shared_prepack_count = 0;
  test.Config(so).ConfigEps(cpu_ep()).RunWithConfig(&prepack_count_session_1, &shared_prepack_count);
  ASSERT_EQ(shared_prepack_count, static_cast<size_t>(0));
  ASSERT_EQ(test.GetNumPrePackedWeightsShared(), prepack_count_session_1);
  ASSERT_GT(prepack_count_session_1, static_cast<size_t>(0));

  size_t prepack_count_session_2 = 0;
  test.Config(so).ConfigEps(cpu_ep()).RunWithConfig(&prepack_count_session_2, &shared_prepack_count);
  ASSERT_EQ(prepack_count_session_2, prepack_count_session_1);
  ASSERT_EQ(shared_prepack_count, prepack_count_session_2);
}

TEST(DynamicQuantMatMulFp8, SharedPrepackedWeightsWithComputedBScalesReuseCorrectly) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;
  constexpr int64_t BlockSize = 2;

  const std::vector<float> a_data{
      1.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> b_data{
      0.31f, -0.37f, 0.83f, -1.70f,
      1.20f, -2.30f, 3.40f, -4.50f,
      5.50f, -6.25f, 7.75f, -8.50f,
      9.00f, -10.50f, 11.25f, -12.75f};
  const auto y_data = ComputeMatMul(ComputeRowwiseAQuantized<Float8E4M3FN>(a_data, M, K, BlockSize),
                                    ComputeConstantBQuantized<Float8E4M3FN>(b_data, K, N, BlockSize, BlockSize),
                                    M, N, K);

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({K, N}), b_data.data(),
                       OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  PrepackedWeightsContainer prepacked_weights_container;

  DynamicQuantMatMulFp8SessionTester test_1("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test_1.AddAttribute<int64_t>("block_size_k", BlockSize);
  test_1.AddAttribute<int64_t>("block_size_n", BlockSize);
  test_1.AddInput<float>("A", {M, K}, a_data);
  test_1.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test_1.AddOutput<float>("Y", {M, N}, y_data);
  test_1.SetOutputAbsErr("Y", 1e-5f);

  size_t shared_prepack_count = 0;
  RunDynamicQuantMatMulFp8WithSharedPrepack(test_1, b, prepacked_weights_container, shared_prepack_count);
  ASSERT_EQ(shared_prepack_count, static_cast<size_t>(0));
  ASSERT_EQ(prepacked_weights_container.GetNumberOfElements(), static_cast<size_t>(1));

  DynamicQuantMatMulFp8SessionTester test_2("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test_2.AddAttribute<int64_t>("block_size_k", BlockSize);
  test_2.AddAttribute<int64_t>("block_size_n", BlockSize);
  test_2.AddInput<float>("A", {M, K}, a_data);
  test_2.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test_2.AddOutput<float>("Y", {M, N}, y_data);
  test_2.SetOutputAbsErr("Y", 1e-5f);

  RunDynamicQuantMatMulFp8WithSharedPrepack(test_2, b, prepacked_weights_container, shared_prepack_count);
  ASSERT_GT(shared_prepack_count, static_cast<size_t>(0));
  ASSERT_EQ(prepacked_weights_container.GetNumberOfElements(), static_cast<size_t>(1));
}

TEST(DynamicQuantMatMulFp8, RejectsMalformedBScaleShapeBeforeReadingScaleData) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.5f));
  std::vector<MLFloat16> b_scale = FloatsToMLFloat16s({1.0f});
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_k", 4);
  test.AddAttribute<int64_t>("block_size_n", 4);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<MLFloat16>("B_scale", {1}, b_scale);
  AddZeroPoint<Float8E4M3FN>(test, "B_zero_point", {1, 1}, 1, false);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires B scale to be a 2D tensor.");
}

TEST(DynamicQuantMatMulFp8, RejectsRuntimeFp8BWithoutBScale) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.5f));
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_k", 4);
  test.AddAttribute<int64_t>("block_size_n", 4);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires B scale when B is already FP8.");
}

TEST(DynamicQuantMatMulFp8, ZeroMInput) {
  constexpr int64_t M = 0;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data{};
  std::vector<float> b_data(static_cast<size_t>(K * N), 0.0f);
  std::vector<float> y_data{};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, ZeroKInput) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 0;

  std::vector<float> a_data{};
  std::vector<float> b_data{};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, ZeroKInputRejectsInvalidYScaleShape) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 0;

  std::vector<float> a_data{};
  std::vector<float> b_data{};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);
  std::vector<float> y_scale{1.0f};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddInput<float>("Y_scale", {1}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires Y scale input to be a scalar.");
}

TEST(DynamicQuantMatMulFp8, ZeroKInputRejectsInvalidYScaleValue) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 0;

  std::vector<float> a_data{};
  std::vector<float> b_data{};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);
  std::vector<float> y_scale{0.0f};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Y scale values to be finite and positive.");
}

TEST(DynamicQuantMatMulFp8, ZeroKInputRejectsInvalidYScaleType) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 0;

  std::vector<float> a_data{};
  std::vector<float> b_data{};
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);
  std::vector<int32_t> y_scale{1};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddInput<int32_t>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Type Error");
}

TEST(DynamicQuantMatMulFp8, ZeroNInput) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 0;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<Float8E4M3FN> b_data{};
  std::vector<float> y_data{};
  std::vector<float> b_scale{};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, ZeroNInputRejectsInvalidYScaleValue) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 0;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<Float8E4M3FN> b_data{};
  std::vector<float> y_data{};
  std::vector<float> b_scale{};
  std::vector<float> y_scale{0.0f};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {N / 128, K / 128}, b_scale);
  test.AddOptionalInputEdge<Float8E4M3FN>();
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Y scale values to be finite and positive.");
}

TEST(DynamicQuantMatMulFp8, ZeroNInputWithConstantNonFp8B) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 0;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<float> b_data{};
  std::vector<float> y_data{};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(DISABLE_FLOAT8_TYPES)
