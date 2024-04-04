// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
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

}  // namespace test
}  // namespace onnxruntime
