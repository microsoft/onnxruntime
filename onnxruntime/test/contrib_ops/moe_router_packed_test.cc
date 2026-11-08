// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

template <typename T>
void CheckRoutes(int rows, bool tied) {
  constexpr int width = 256;
  constexpr int experts = 4;
  constexpr int k = 2;
  OpTester test("MoERouterPacked", 1, kMSDomain);
  test.AddAttribute("k", int64_t{k});
  test.AddAttribute("router_scale", 2.5f);
  test.AddInput<T>("input", {rows, width}, std::vector<T>(rows * width, T(0.f)));
  test.AddInput<T>("weight", {experts, width}, std::vector<T>(experts * width, T(0.f)), true);
  test.AddInput<float>("bias", {experts}, tied ? std::vector<float>(experts, 0.f) : std::vector<float>{0.f, 3.f, 2.f, 1.f}, true);
  std::vector<int32_t> indices;
  for (int row = 0; row < rows; ++row) {
    indices.push_back(tied ? 0 : 1);
    indices.push_back(tied ? 1 : 2);
  }
  test.AddOutput<int32_t>("indices", {rows, k}, indices);
  test.AddOutput<float>("weights", {rows, k}, std::vector<float>(rows * k, 1.25f));
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void CheckProjection() {
  constexpr int width = 256;
  constexpr int experts = 4;
  std::vector<T> input(width, T(0.f)), weight(experts * width, T(0.f));
  input[0] = T(1.f);
  for (int expert = 0; expert < experts; ++expert) weight[expert * width] = T(float(expert - 1));
  const float first = 1.f / (1.f + std::exp(1.f));
  const float sum = first + 0.5f;
  OpTester test("MoERouterPacked", 1, kMSDomain);
  test.AddAttribute("k", int64_t{2});
  test.AddAttribute("router_scale", 2.5f);
  test.AddInput<T>("input", {1, width}, input);
  test.AddInput<T>("weight", {experts, width}, weight, true);
  test.AddInput<float>("bias", {experts}, {4.f, 3.f, 0.f, 0.f}, true);
  test.AddOutput<int32_t>("indices", {1, 2}, {0, 1});
  test.AddOutput<float>("weights", {1, 2}, {first / sum * 2.5f, 0.5f / sum * 2.5f});
  test.SetOutputTolerance(1e-6f, 1e-6f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T>
void CheckMaximumK() {
  constexpr int width = 256;
  constexpr int experts = 512;
  constexpr int k = 32;
  OpTester test("MoERouterPacked", 1, kMSDomain);
  test.AddAttribute("k", int64_t{k});
  test.AddAttribute("router_scale", 2.5f);
  test.AddInput<T>("input", {1, width}, std::vector<T>(width, T(0.f)));
  test.AddInput<T>("weight", {experts, width}, std::vector<T>(experts * width, T(0.f)), true);
  test.AddInput<float>("bias", {experts}, std::vector<float>(experts, 0.f), true);
  std::vector<int32_t> indices(k);
  std::iota(indices.begin(), indices.end(), 0);
  test.AddOutput<int32_t>("indices", {1, k}, indices);
  test.AddOutput<float>("weights", {1, k}, std::vector<float>(k, 2.5f / k));
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MoERouterPackedTest, SelectionBiasDoesNotChangeMixtureWeights) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {0, 1, 8, 16}) {
    CheckRoutes<BFloat16>(rows, false);
    CheckRoutes<MLFloat16>(rows, false);
  }
}

TEST(MoERouterPackedTest, TiesSelectLowerExpertIndices) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckRoutes<BFloat16>(8, true);
  CheckRoutes<MLFloat16>(8, true);
}

TEST(MoERouterPackedTest, ProjectionUsesUnbiasedAffinities) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckProjection<BFloat16>();
  CheckProjection<MLFloat16>();
}

TEST(MoERouterPackedTest, MaximumExpertsAndK) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  CheckMaximumK<BFloat16>();
  CheckMaximumK<MLFloat16>();
}

TEST(MoERouterPackedTest, NonfiniteBiasPreservesDeterministicRanking) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int width = 256;
  const float inf = std::numeric_limits<float>::infinity();
  OpTester test("MoERouterPacked", 1, kMSDomain);
  test.AddAttribute("k", int64_t{4});
  test.AddInput<BFloat16>("input", {1, width}, std::vector<BFloat16>(width, BFloat16(0.f)));
  test.AddInput<BFloat16>("weight", {4, width}, std::vector<BFloat16>(4 * width, BFloat16(0.f)), true);
  test.AddInput<float>("bias", {4}, {-inf, inf, 0.f, std::numeric_limits<float>::quiet_NaN()}, true);
  test.AddOutput<int32_t>("indices", {1, 4}, {1, 2, 0, 3});
  test.AddOutput<float>("weights", {1, 4}, {0.25f, 0.25f, 0.25f, 0.25f});
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(MoERouterPackedTest, RejectsUnalignedRouterWidth) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester test("MoERouterPacked", 1, kMSDomain);
  test.AddAttribute("k", int64_t{1});
  test.AddInput<BFloat16>("input", {1, 255}, std::vector<BFloat16>(255, BFloat16(0.f)));
  test.AddInput<BFloat16>("weight", {1, 255}, std::vector<BFloat16>(255, BFloat16(0.f)), true);
  test.AddInput<float>("bias", {1}, {0.f}, true);
  test.AddOutput<int32_t>("indices", {1, 1}, {0});
  test.AddOutput<float>("weights", {1, 1}, {1.f});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectFailure, "width divisible by 256",
           {}, nullptr, &providers);
}

}  // namespace
}  // namespace onnxruntime::test
#endif
