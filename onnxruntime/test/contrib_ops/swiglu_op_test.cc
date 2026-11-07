// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

float SwiGluRef(float g, float l, float limit, float alpha, float beta) {
  if (limit > 0.0f) {
    g = std::min(g, limit);
    l = std::min(std::max(l, -limit), limit);
  }
  return g / (1.0f + std::exp(-alpha * g)) * (l + beta);
}

// Runs SwiGLU on CUDA with either separate gate/up inputs or one `[rows, 2 * cols]` input.
void RunSwiGLUTest(bool fused_input, float limit, float alpha, float beta) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  if (!cuda_ep) {
    GTEST_SKIP() << "CUDA is not available";
  }

  constexpr int64_t rows = 3;
  constexpr int64_t cols = 37;
  std::mt19937 generator(11);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  std::vector<float> gate(rows * cols);
  std::vector<float> up(rows * cols);
  for (float& v : gate) v = dist(generator);
  for (float& v : up) v = dist(generator);

  std::vector<float> expected(rows * cols);
  for (size_t i = 0; i < expected.size(); ++i) expected[i] = SwiGluRef(gate[i], up[i], limit, alpha, beta);

  OpTester tester("SwiGLU", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<float>("limit", limit);
  tester.AddAttribute<float>("activation_alpha", alpha);
  tester.AddAttribute<float>("activation_beta", beta);
  if (fused_input) {
    std::vector<float> merged(rows * 2 * cols);
    for (int64_t r = 0; r < rows; ++r) {
      std::copy_n(gate.begin() + r * cols, cols, merged.begin() + r * 2 * cols);
      std::copy_n(up.begin() + r * cols, cols, merged.begin() + r * 2 * cols + cols);
    }
    tester.AddInput<float>("gate", {rows, 2 * cols}, merged);
  } else {
    tester.AddInput<float>("gate", {rows, cols}, gate);
    tester.AddInput<float>("up", {rows, cols}, up);
  }
  tester.AddOutput<float>("output", {rows, cols}, expected, false, 1e-5f, 1e-5f);

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace

TEST(SwiGLUCudaTest, SeparateInputs) { RunSwiGLUTest(false, 0.0f, 1.0f, 0.0f); }
TEST(SwiGLUCudaTest, SingleMergedInput) { RunSwiGLUTest(true, 0.0f, 1.0f, 0.0f); }
TEST(SwiGLUCudaTest, ClampAlphaBeta) { RunSwiGLUTest(true, 3.0f, 1.702f, 1.0f); }

}  // namespace test
}  // namespace onnxruntime
