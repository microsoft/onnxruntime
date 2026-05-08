// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

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

float QuantizeDequantizeE4M3(float value, float scale) {
  return Float8E4M3FN(value / scale, true).ToFloat() * scale;
}

std::vector<float> ComputeExpectedIdentityAWithQuantizedB(gsl::span<const float> b_data,
                                                          gsl::span<const float> b_scale,
                                                          int64_t k,
                                                          int64_t n,
                                                          int64_t block_size_k,
                                                          int64_t block_size_n) {
  const int64_t blocks_n = n / block_size_n;
  std::vector<float> expected(b_data.size());
  for (int64_t row = 0; row < k; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const int64_t scale_idx = (row / block_size_k) * blocks_n + (col / block_size_n);
      const size_t data_idx = static_cast<size_t>(row * n + col);
      expected[data_idx] = QuantizeDequantizeE4M3(b_data[data_idx], b_scale[scale_idx]);
    }
  }
  return expected;
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
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, WithOmittedOutputQuantizationInputs) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, WithOnlyYScale) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  constexpr float y_scale_value = 0.5f;
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K) * y_scale_value;
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{y_scale_value};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, RejectsNonZeroYZeroPoint) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(1.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddOptionalInputEdge<float>();
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 supports symmetric quantization only; Y zero point values must be zero.");
}

TEST(DynamicQuantMatMulFp8, WithConstantBInputsBf16Scales) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<BFloat16> a_scale = MakeBFloat16({1.0f});
  std::vector<BFloat16> b_scale = MakeBFloat16({1.0f});
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<BFloat16> y_scale = MakeBFloat16({1.0f});
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<BFloat16>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<BFloat16>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<BFloat16>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, Float16Output) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("A", {M, K}, FloatsToMLFloat16s(a_data));
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<MLFloat16>("Y", {M, N}, FloatsToMLFloat16s(y_data));
  test.Run();
}

TEST(DynamicQuantMatMulFp8, BFloat16Output) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<BFloat16>("A", {M, K}, FloatsToBFloat16s(a_data));
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(DynamicQuantMatMulFp8, RuntimeFp8BInput) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  const float expected_value = 0.25f * -0.5f * static_cast<float>(K);
  std::vector<float> y_data(static_cast<size_t>(M * N), expected_value);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.5f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, RejectsRuntimeFp8BZeroPointTypeMismatch) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E5M2> b_zp{Float8E5M2(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale);
  test.AddInput<Float8E5M2>("B_zero_point", {K / 128, N / 128}, b_zp);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(DynamicQuantMatMulFp8, RejectsNonZeroAZeroPoint) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(-0.5f));
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(1.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 supports symmetric quantization only; A zero point values must be zero.");
}

TEST(DynamicQuantMatMulFp8, RejectsNonZeroBZeroPoint) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<float> b_data(static_cast<size_t>(K * N), -0.5f);
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(1.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 supports symmetric quantization only; B zero point values must be zero.");
}

TEST(DynamicQuantMatMulFp8, NonDefaultBlockSizesWithPartialM) {
  constexpr int64_t M = 9;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      a_data[static_cast<size_t>(m * K + k)] = (m >= 4 && m <= 7) ? 0.04f : static_cast<float>(1 << k);
    }
  }
  for (int64_t k = 0; k < K; ++k) {
    a_data[static_cast<size_t>(3 * K + k)] = 64.0f;
  }

  std::vector<float> b_data(static_cast<size_t>(K * N), 0.0f);
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      b_data[static_cast<size_t>(k * N + n)] = (k == n) ? 1.0f : 0.0f;
    }
  }

  const std::vector<float> a_scale{1.0f, 1.0f,
                                   0.01f, 0.01f,
                                   1.0f, 1.0f};
  const std::vector<float> b_scale{1.0f, 1.0f,
                                   1.0f, 1.0f};
  std::vector<Float8E4M3FN> a_zp(a_scale.size(), Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> b_zp(b_scale.size(), Float8E4M3FN(0.0f));
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += a_data[static_cast<size_t>(m * K + k)] * b_data[static_cast<size_t>(k * N + n)];
      }
      y_data[static_cast<size_t>(m * N + n)] = sum;
    }
  }

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_m", 4);
  test.AddAttribute<int64_t>("block_size_k", 2);
  test.AddAttribute<int64_t>("block_size_n", 2);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {3, 2}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {3, 2}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {2, 2}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {2, 2}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.01f);
  test.Run();
}

TEST(DynamicQuantMatMulFp8, SharedPrepackedWeightsRestoresPackedBMetadata) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

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

  std::vector<float> y_data(a_data.size());
  for (size_t i = 0; i < a_data.size(); ++i) {
    y_data[i] = 2.0f * a_data[i];
  }

  std::vector<float> a_scale{1.0f, 1.0f,
                             1.0f, 1.0f};
  std::vector<float> b_scale{1.0f, 1.0f,
                             1.0f, 1.0f};
  std::vector<Float8E5M2> a_zp(a_scale.size(), Float8E5M2(0.0f));
  std::vector<Float8E5M2> b_zp(b_scale.size(), Float8E5M2(0.0f));
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E5M2> y_zp{Float8E5M2(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_m", 2);
  test.AddAttribute<int64_t>("block_size_k", 2);
  test.AddAttribute<int64_t>("block_size_n", 2);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {2, 2}, a_scale);
  test.AddInput<Float8E5M2>("A_zero_point", {2, 2}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {2, 2}, b_scale, true /*initializer*/);
  test.AddInput<Float8E5M2>("B_zero_point", {2, 2}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E5M2>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.SetOutputAbsErr("Y", 0.01f);

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

TEST(DynamicQuantMatMulFp8, SharedPrepackedWeightsWithDifferentBScaleKeepCorrectSemantics) {
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

  std::vector<float> a_scale{1.0f, 1.0f,
                             1.0f, 1.0f};
  std::vector<Float8E4M3FN> a_zp(a_scale.size(), Float8E4M3FN(0.0f));
  std::vector<Float8E4M3FN> b_zp(a_scale.size(), Float8E4M3FN(0.0f));
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  std::vector<float> b_scale_1{1.0f, 1.0f,
                               1.0f, 1.0f};
  std::vector<float> b_scale_2{0.10f, 0.25f,
                               0.50f, 2.00f};
  std::vector<float> y_data_1 = ComputeExpectedIdentityAWithQuantizedB(b_data, b_scale_1, K, N,
                                                                       BlockSize, BlockSize);
  std::vector<float> y_data_2 = ComputeExpectedIdentityAWithQuantizedB(b_data, b_scale_2, K, N,
                                                                       BlockSize, BlockSize);

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({K, N}), b_data.data(),
                       OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  PrepackedWeightsContainer prepacked_weights_container;

  DynamicQuantMatMulFp8SessionTester test_1("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test_1.AddAttribute<int64_t>("block_size_m", BlockSize);
  test_1.AddAttribute<int64_t>("block_size_k", BlockSize);
  test_1.AddAttribute<int64_t>("block_size_n", BlockSize);
  test_1.AddInput<float>("A", {M, K}, a_data);
  test_1.AddInput<float>("A_scale", {2, 2}, a_scale);
  test_1.AddInput<Float8E4M3FN>("A_zero_point", {2, 2}, a_zp);
  test_1.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test_1.AddInput<float>("B_scale", {2, 2}, b_scale_1, true /*initializer*/);
  test_1.AddInput<Float8E4M3FN>("B_zero_point", {2, 2}, b_zp, true /*initializer*/);
  test_1.AddInput<float>("Y_scale", {}, y_scale);
  test_1.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test_1.AddOutput<float>("Y", {M, N}, y_data_1);
  test_1.SetOutputAbsErr("Y", 1e-5f);

  size_t shared_prepack_count = 0;
  RunDynamicQuantMatMulFp8WithSharedPrepack(test_1, b, prepacked_weights_container, shared_prepack_count);
  ASSERT_EQ(shared_prepack_count, static_cast<size_t>(0));
  ASSERT_EQ(prepacked_weights_container.GetNumberOfElements(), static_cast<size_t>(1));

  DynamicQuantMatMulFp8SessionTester test_2("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test_2.AddAttribute<int64_t>("block_size_m", BlockSize);
  test_2.AddAttribute<int64_t>("block_size_k", BlockSize);
  test_2.AddAttribute<int64_t>("block_size_n", BlockSize);
  test_2.AddInput<float>("A", {M, K}, a_data);
  test_2.AddInput<float>("A_scale", {2, 2}, a_scale);
  test_2.AddInput<Float8E4M3FN>("A_zero_point", {2, 2}, a_zp);
  test_2.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test_2.AddInput<float>("B_scale", {2, 2}, b_scale_2, true /*initializer*/);
  test_2.AddInput<Float8E4M3FN>("B_zero_point", {2, 2}, b_zp, true /*initializer*/);
  test_2.AddInput<float>("Y_scale", {}, y_scale);
  test_2.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test_2.AddOutput<float>("Y", {M, N}, y_data_2);
  test_2.SetOutputAbsErr("Y", 1e-5f);

  RunDynamicQuantMatMulFp8WithSharedPrepack(test_2, b, prepacked_weights_container, shared_prepack_count);
  ASSERT_EQ(shared_prepack_count, static_cast<size_t>(0));
  ASSERT_EQ(prepacked_weights_container.GetNumberOfElements(), static_cast<size_t>(2));
}

TEST(DynamicQuantMatMulFp8, RejectsMismatchedAScaleBatchPrefix) {
  constexpr int64_t Batch = 2;
  constexpr int64_t Seq = 3;
  constexpr int64_t M = 128;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(Batch * Seq * M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.5f));
  std::vector<float> y_data(static_cast<size_t>(Batch * Seq * M * N), 0.0f);

  std::vector<float> a_scale(static_cast<size_t>(Seq * Batch), 1.0f);
  std::vector<Float8E4M3FN> a_zp(a_scale.size(), Float8E4M3FN(0.0f));
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  const std::vector<std::string> a_dim_params{"batch", "seq", "128", "128"};
  const std::vector<std::string> a_scale_dim_params{"seq", "batch", "1", "1"};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {Batch, Seq, M, K}, a_data, false, &a_dim_params);
  test.AddInput<float>("A_scale", {Seq, Batch, 1, 1}, a_scale, false, &a_scale_dim_params);
  test.AddInput<Float8E4M3FN>("A_zero_point", {Seq, Batch, 1, 1}, a_zp, false, &a_scale_dim_params);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {1, 1}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {1, 1}, b_zp);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {Batch, Seq, M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires A scale batch dimensions to match Y.");
}

TEST(DynamicQuantMatMulFp8, RejectsMalformedAScaleShapeBeforeReadingScaleData) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.5f));
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<MLFloat16> a_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<MLFloat16> b_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<MLFloat16> y_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_m", 4);
  test.AddAttribute<int64_t>("block_size_k", 4);
  test.AddAttribute<int64_t>("block_size_n", 4);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<MLFloat16>("A_scale", {1}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {1}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<MLFloat16>("B_scale", {1, 1}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {1, 1}, b_zp);
  test.AddInput<MLFloat16>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires A scale to have rank >= 2.");
}

TEST(DynamicQuantMatMulFp8, RejectsMalformedBScaleShapeBeforeReadingScaleData) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 4;
  constexpr int64_t K = 4;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.25f);
  std::vector<Float8E4M3FN> b_data(static_cast<size_t>(K * N), Float8E4M3FN(0.5f));
  std::vector<float> y_data(static_cast<size_t>(M * N), 0.0f);

  std::vector<MLFloat16> a_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<MLFloat16> b_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<MLFloat16> y_scale = FloatsToMLFloat16s({1.0f});
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("block_size_m", 4);
  test.AddAttribute<int64_t>("block_size_k", 4);
  test.AddAttribute<int64_t>("block_size_n", 4);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<MLFloat16>("A_scale", {1, 1}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {1, 1}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<MLFloat16>("B_scale", {1}, b_scale);
  test.AddInput<Float8E4M3FN>("B_zero_point", {1}, b_zp);
  test.AddInput<MLFloat16>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires B scale to be a 2D tensor.");
}

TEST(DynamicQuantMatMulFp8, ZeroMInput) {
  constexpr int64_t M = 0;
  constexpr int64_t N = 128;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<float> b_data(static_cast<size_t>(K * N), 0.0f);
  std::vector<float> y_data{};

  std::vector<float> a_scale{};
  std::vector<float> b_scale{1.0f};
  std::vector<Float8E4M3FN> a_zp{};
  std::vector<Float8E4M3FN> b_zp{Float8E4M3FN(0.0f)};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddShapeToTensorData(false);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {1}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{0.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<int32_t> y_scale{1};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data, true /*initializer*/);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<int32_t>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Type Error");
}

TEST(DynamicQuantMatMulFp8, ZeroNInput) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 0;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<Float8E4M3FN> b_data{};
  std::vector<float> y_data{};

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
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

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{0.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<Float8E4M3FN>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Y scale values to be finite and positive.");
}

TEST(DynamicQuantMatMulFp8, ZeroNInputRejectsNonFp8B) {
  constexpr int64_t M = 128;
  constexpr int64_t N = 0;
  constexpr int64_t K = 128;

  std::vector<float> a_data(static_cast<size_t>(M * K), 0.0f);
  std::vector<float> b_data{};
  std::vector<float> y_data{};

  std::vector<float> a_scale{1.0f};
  std::vector<float> b_scale{};
  std::vector<Float8E4M3FN> a_zp{Float8E4M3FN(0.0f)};
  std::vector<Float8E4M3FN> b_zp{};
  std::vector<float> y_scale{1.0f};
  std::vector<Float8E4M3FN> y_zp{Float8E4M3FN(0.0f)};

  OpTester test("DynamicQuantMatMulFp8", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("A", {M, K}, a_data);
  test.AddInput<float>("A_scale", {M / 128, K / 128}, a_scale);
  test.AddInput<Float8E4M3FN>("A_zero_point", {M / 128, K / 128}, a_zp);
  test.AddInput<float>("B", {K, N}, b_data);
  test.AddInput<float>("B_scale", {K / 128, N / 128}, b_scale, true /*initializer*/);
  test.AddInput<Float8E4M3FN>("B_zero_point", {K / 128, N / 128}, b_zp, true /*initializer*/);
  test.AddInput<float>("Y_scale", {}, y_scale);
  test.AddInput<Float8E4M3FN>("Y_zero_point", {}, y_zp);
  test.AddOutput<float>("Y", {M, N}, y_data);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "DynamicQuantMatMulFp8 requires runtime B input to be FP8.");
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(DISABLE_FLOAT8_TYPES)
