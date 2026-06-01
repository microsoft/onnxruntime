// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
using namespace std;
namespace onnxruntime {
namespace test {

template <typename T>
void L1Normalization() {
  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("p", (int64_t)1);

  vector<T> input = {5.93932154F, 7.4367043F, 6.42487038F, 5.90394865F,
                     4.81289319F, 6.81304702F, 4.9382849F, 9.02595701F,
                     9.67296484F, 4.45097367F, 8.12552534F, 5.76005428F,

                     6.11240105F, 9.33036974F, 1.63932452F, 1.7841637F,
                     1.18196558F, 8.49357861F, 8.00341076F, 8.83010933F,
                     9.80756508F, 8.19242708F, 5.15331426F, 8.02476259F};
  vector<int64_t> input_dims = {2, 3, 4};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {0.2907843F, 0.3976693F, 0.3296719F, 0.28535331F,
                               0.23563529F, 0.36431994F, 0.25339247F, 0.43624816F,
                               0.47358041F, 0.23801075F, 0.41693563F, 0.27839852F,

                               0.35740998F, 0.3586345F, 0.11079474F, 0.09572189F,
                               0.06911299F, 0.32647048F, 0.54091538F, 0.47374282F,
                               0.57347703F, 0.31489502F, 0.34828988F, 0.43053529F};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run();
}

TEST(LpNormalizationTest, L1Normalization) {
  L1Normalization<float>();
  L1Normalization<double>();
}

template <typename T>
void L2Normalization() {
  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)1);
  test.AddAttribute("p", (int64_t)2);

  vector<T> input = {5.93932154F, 7.4367043F, 6.42487038F, 5.90394865F,
                     4.81289319F, 6.81304702F, 4.9382849F, 9.02595701F,
                     9.67296484F, 4.45097367F, 8.12552534F, 5.76005428F,

                     6.11240105F, 9.33036974F, 1.63932452F, 1.7841637F,
                     1.18196558F, 8.49357861F, 8.00341076F, 8.83010933F,
                     9.80756508F, 8.19242708F, 5.15331426F, 8.02476259F};
  vector<int64_t> input_dims = {2, 3, 4};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {
      0.48173351F, 0.67457895F, 0.55987147F, 0.48285641F,
      0.39036983F, 0.61800737F, 0.4303285F, 0.73819091F,
      0.78456626F, 0.40374513F, 0.70806873F, 0.47108796F,

      0.52617536F, 0.62021826F, 0.16971778F, 0.14788607F,
      0.10174744F, 0.56459419F, 0.82858584F, 0.73191164F,
      0.8442671F, 0.54457572F, 0.53351794F, 0.66515792F};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run();
}

TEST(LpNormalizationTest, L2Normalization) {
  L2Normalization<float>();
  L2Normalization<double>();
}

template <typename T>
void LpNormalizationDefaultAxisAndP() {
  OpTester test("LpNormalization");
  vector<T> input = {
      0.0f, 0.5f, 2.0f, 2.0f,
      1.0f, 0.5f, 2.0f, 2.5f,
      1.0f, 1.5f, 3.0f, 3.0f,
      1.5f, 2.0f, 3.5f, 3.5f};

  vector<int64_t> input_dims = {16};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {
      0.0f, 0.059028134f, 0.236112535f, 0.236112535f,
      0.118056267f, 0.059028134f, 0.236112535f, 0.295140654f,
      0.118056267f, 0.177084401f, 0.354168802f, 0.354168802f,
      0.177084401f, 0.236112535f, 0.413196921f, 0.413196921f};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run();
}

TEST(LpNormalizationTest, LpNormalizationDefaultAxisAndP) {
  LpNormalizationDefaultAxisAndP<float>();
  LpNormalizationDefaultAxisAndP<double>();
}

template <typename T>
void L1NormalizationWithValidNegativeAxis() {
  OpTester test("LpNormalization");
  test.AddAttribute("axis", static_cast<int64_t>(-2));
  test.AddAttribute("p", static_cast<int64_t>(1));

  vector<T> input = {5.93932154F, 7.4367043F, 6.42487038F, 5.90394865F,
                     4.81289319F, 6.81304702F, 4.9382849F, 9.02595701F,
                     9.67296484F, 4.45097367F, 8.12552534F, 5.76005428F,

                     6.11240105F, 9.33036974F, 1.63932452F, 1.7841637F,
                     1.18196558F, 8.49357861F, 8.00341076F, 8.83010933F,
                     9.80756508F, 8.19242708F, 5.15331426F, 8.02476259F};
  vector<int64_t> input_dims = {2, 3, 4};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {0.2907843F, 0.3976693F, 0.3296719F, 0.28535331F,
                               0.23563529F, 0.36431994F, 0.25339247F, 0.43624816F,
                               0.47358041F, 0.23801075F, 0.41693563F, 0.27839852F,

                               0.35740998F, 0.3586345F, 0.11079474F, 0.09572189F,
                               0.06911299F, 0.32647048F, 0.54091538F, 0.47374282F,
                               0.57347703F, 0.31489502F, 0.34828988F, 0.43053529F};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run();
}

TEST(LpNormalizationTest, L1NormalizationWithValidNegativeAxis) {
  L1NormalizationWithValidNegativeAxis<float>();
  L1NormalizationWithValidNegativeAxis<double>();
}

template <typename T>
void L1NormalizationWithZeroNorm() {
  OpTester test("LpNormalization");
  test.AddAttribute("p", static_cast<int64_t>(1));

  // With default axis (axis = -1), one of the norms will be evaluated to zero
  // for the following input
  vector<T> input = {2.f, 2.f, 0.f, 0.f};
  vector<int64_t> input_dims = {2, 2};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {0.5f, 0.5f, 0.f, 0.f};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(LpNormalizationTest, L1NormalizationWithZeroNorm) {
  L1NormalizationWithZeroNorm<float>();
  L1NormalizationWithZeroNorm<double>();
}

template <typename T>
void L2NormalizationWithZeroNorm() {
  OpTester test("LpNormalization");

  // With default axis (axis = -1), one of the norms will be evaluated to zero
  // for the following input
  vector<T> input = {1.f, 0.f, 0.f, 0.f};
  vector<int64_t> input_dims = {2, 2};
  test.AddInput<T>("input", input_dims, input);

  vector<T> expected_output = {1.f, 0.f, 0.f, 0.f};
  test.AddOutput<T>("Y", input_dims, expected_output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(LpNormalizationTest, L2NormalizationWithZeroNorm) {
  L2NormalizationWithZeroNorm<float>();
  L2NormalizationWithZeroNorm<double>();
}

TEST(LpNormalizationTest, L2Normalization_FP16) {
  // FP16 is only supported on CUDA/ROCM EPs. Skip before building the model so the
  // OpTester is never constructed without a Run() (a Debug-build DebugTrap otherwise fires).
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)-1);  // normalize along last axis
  test.AddAttribute("p", (int64_t)2);

  // Use axis length 128 with magnitudes ~100 so sum-of-squares (~128*10000 = 1.28M)
  // exceeds FP16 max (65504). This locks in the float-accumulation fix: without it
  // the sum overflows to inf and the output is all zeros.
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 128;
  vector<float> input_f(kRows * kCols);
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      // Vary values per row to get different norms
      input_f[r * kCols + c] = 80.0f + static_cast<float>(c % 7) * 5.0f + static_cast<float>(r) * 10.0f;
    }
  }

  // Compute expected output in float
  vector<float> expected_f(kRows * kCols);
  for (int64_t r = 0; r < kRows; ++r) {
    float sum_sq = 0.0f;
    for (int64_t c = 0; c < kCols; ++c) {
      float v = input_f[r * kCols + c];
      sum_sq += v * v;
    }
    float norm = std::sqrt(sum_sq);
    for (int64_t c = 0; c < kCols; ++c) {
      expected_f[r * kCols + c] = input_f[r * kCols + c] / norm;
    }
  }

  vector<int64_t> dims = {kRows, kCols};
  vector<MLFloat16> input(input_f.size());
  for (size_t i = 0; i < input_f.size(); ++i) input[i] = MLFloat16(input_f[i]);
  test.AddInput<MLFloat16>("input", dims, input);

  vector<MLFloat16> expected(expected_f.size());
  for (size_t i = 0; i < expected_f.size(); ++i) expected[i] = MLFloat16(expected_f[i]);
  test.AddOutput<MLFloat16>("Y", dims, expected);

  SessionOptions so;
  ASSERT_TRUE(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1").IsOK());
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(LpNormalizationTest, L1Normalization_FP16) {
  // FP16 is only supported on CUDA/ROCM EPs. Skip before building the model so the
  // OpTester is never constructed without a Run() (a Debug-build DebugTrap otherwise fires).
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)-1);  // normalize along last axis
  test.AddAttribute("p", (int64_t)1);

  // Use axis length 128 with magnitudes ~200 so sum (~128*200 = 25600)
  // is large enough to stress FP16 precision.
  constexpr int64_t kRows = 2;
  constexpr int64_t kCols = 128;
  vector<float> input_f(kRows * kCols);
  for (int64_t r = 0; r < kRows; ++r) {
    for (int64_t c = 0; c < kCols; ++c) {
      input_f[r * kCols + c] = 150.0f + static_cast<float>(c % 11) * 10.0f + static_cast<float>(r) * 20.0f;
    }
  }

  // Compute expected output in float
  vector<float> expected_f(kRows * kCols);
  for (int64_t r = 0; r < kRows; ++r) {
    float sum_abs = 0.0f;
    for (int64_t c = 0; c < kCols; ++c) {
      sum_abs += std::abs(input_f[r * kCols + c]);
    }
    for (int64_t c = 0; c < kCols; ++c) {
      expected_f[r * kCols + c] = input_f[r * kCols + c] / sum_abs;
    }
  }

  vector<int64_t> dims = {kRows, kCols};
  vector<MLFloat16> input(input_f.size());
  for (size_t i = 0; i < input_f.size(); ++i) input[i] = MLFloat16(input_f[i]);
  test.AddInput<MLFloat16>("input", dims, input);

  vector<MLFloat16> expected(expected_f.size());
  for (size_t i = 0; i < expected_f.size(); ++i) expected[i] = MLFloat16(expected_f[i]);
  test.AddOutput<MLFloat16>("Y", dims, expected);

  SessionOptions so;
  ASSERT_TRUE(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1").IsOK());
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(LpNormalizationTest, L2Normalization_LastAxis) {
  // Test normalization along the last axis (axis=-1), which is the most common
  // use case for LpNormalization in attention patterns (e.g. Qwen3.5 L2-norm).
  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)-1);
  test.AddAttribute("p", (int64_t)2);

  // 2x4 input, normalize each row
  vector<float> input = {3.0f, 4.0f, 0.0f, 0.0f,
                         1.0f, 2.0f, 2.0f, 0.0f};
  vector<int64_t> dims = {2, 4};
  test.AddInput<float>("input", dims, input);

  // Row 0: norm = 5, Row 1: norm = 3
  vector<float> expected = {0.6f, 0.8f, 0.0f, 0.0f,
                            1.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 0.0f};
  test.AddOutput<float>("Y", dims, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(LpNormalizationTest, L2Normalization_Axis0) {
  // Test normalization along axis 0
  OpTester test("LpNormalization");
  test.AddAttribute("axis", (int64_t)0);
  test.AddAttribute("p", (int64_t)2);

  vector<float> input = {3.0f, 1.0f,
                         4.0f, 2.0f};
  vector<int64_t> dims = {2, 2};
  test.AddInput<float>("input", dims, input);

  // Col 0: norm = 5, Col 1: norm = sqrt(5)
  float s5 = std::sqrt(5.0f);
  vector<float> expected = {3.0f / 5.0f, 1.0f / s5,
                            4.0f / 5.0f, 2.0f / s5};
  test.AddOutput<float>("Y", dims, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
