// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void RunRouterCase(const std::vector<float>& bias, int64_t n_group, int64_t group_topk,
                   const std::vector<int64_t>& selected_experts) {
  constexpr int64_t rows = 3;
  const int64_t experts = static_cast<int64_t>(bias.size());
  const size_t count = static_cast<size_t>(rows * experts);
  std::vector<T> expected(count, T{-10000.0f});
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t expert : selected_experts) {
      expected[static_cast<size_t>(row * experts + expert)] = T{std::log(0.5f)};
    }
  }

  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  if (auto cuda_ep = DefaultCudaExecutionProvider();
      cuda_ep && (!std::is_same_v<T, BFloat16> || CudaHasBF16Support())) {
    providers.push_back(std::move(cuda_ep));
  }
  for (auto& provider : providers) {
    SCOPED_TRACE(provider->Type());
    SCOPED_TRACE("experts=" + std::to_string(experts) + ", group_topk=" + std::to_string(group_topk));
    OpTester tester("MoEGroupRouter", 1, kMSDomain);
    tester.AddAttribute("top_k", static_cast<int64_t>(selected_experts.size()));
    tester.AddAttribute("n_group", n_group);
    tester.AddAttribute("topk_group", int64_t{1});
    tester.AddAttribute("group_topk", group_topk);
    tester.AddInput<T>("logits", {rows, experts}, std::vector<T>(count, T{0.0f}));
    tester.AddInput<float>("expert_bias", {experts}, bias);
    tester.AddOutput<T>("router_probs", {rows, experts}, expected, false, 0.001f, 0.001f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(provider));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

template <typename T>
void RunRouterPaths() {
  std::vector<float> bias(256, -0.5f);
  bias[100] = 0.5f;
  bias[128] = bias[129] = 0.3f;
  RunRouterCase<T>(bias, 2, 2, {128});

  bias.assign(512, -0.5f);
  bias[100] = 0.5f;
  bias[256] = bias[257] = bias[258] = 0.3f;
  RunRouterCase<T>(bias, 2, 3, {256, 257});
  RunRouterCase<T>(bias, 2, 259, {256, 257});

  bias.assign(12, -0.5f);
  bias[1] = 0.5f;
  bias[4] = bias[5] = 0.3f;
  RunRouterCase<T>(bias, 3, 8, {4, 5});

  bias.assign(128, -0.5f);
  bias[1] = 0.5f;
  bias[16] = bias[17] = 0.3f;
  RunRouterCase<T>(bias, 8, 2, {16, 17});
}

void RunRouterAttributeFailure(std::string_view attribute, int64_t value,
                               bool use_cuda, const std::string& expected_error) {
  SCOPED_TRACE(std::string(attribute) + "=" + std::to_string(value));
  auto provider = use_cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
  ASSERT_NE(provider, nullptr);
  SCOPED_TRACE(provider->Type());
  OpTester tester("MoEGroupRouter", 1, kMSDomain);
  tester.AddAttribute("top_k", attribute == "top_k" ? value : int64_t{1});
  tester.AddAttribute("n_group", attribute == "n_group" ? value : int64_t{2});
  tester.AddAttribute("topk_group", attribute == "topk_group" ? value : int64_t{1});
  tester.AddAttribute("group_topk", attribute == "group_topk" ? value : int64_t{2});
  tester.AddInput<float>("logits", {1, 8}, std::vector<float>(8, 0.0f));
  tester.AddInput<float>("expert_bias", {8}, std::vector<float>(8, 0.0f));
  tester.AddOutput<float>("router_probs", {1, 8}, std::vector<float>(8, -10000.0f));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(provider));
  tester.Run(OpTester::ExpectResult::kExpectFailure, expected_error, {}, nullptr, &eps);
}

constexpr std::array<std::string_view, 4> kRouterAttributes = {
    "top_k", "n_group", "topk_group", "group_topk"};

template <typename T>
void RunCompactCase(bool projected, int rows, bool dense, bool maximum = false, bool nonfinite = false) {
  const int width = projected ? 512 : 4;
  const int experts = maximum ? 512 : 4;
  const int k = maximum ? 32 : nonfinite ? 4
                                         : 2;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> bias = maximum     ? std::vector<float>(experts, 0.f)
                            : nonfinite ? std::vector<float>{-inf, inf, 0.f, NAN}
                                        : std::vector<float>{4.f, 3.f, 0.f, 0.f};
  std::vector<T> input(rows * width, T{0.f}), projection(experts * width, T{0.f});
  if (!maximum && !nonfinite) {
    // Nonzero products in every lane and both width tiles exercise the projection reduction.
    for (int row = 0; row < rows; ++row) {
      for (int i = 0; i < width; ++i) input[row * width + i] = T{projected ? 1.f : float(i - 1)};
    }
    for (int e = 0; e < experts; ++e) {
      for (int i = 0; i < width; ++i) projection[e * width + i] = T{float(e - 1) / width};
    }
  }
  std::vector<int32_t> ids;
  std::vector<float> weights;
  std::vector<T> log_scores(rows * experts, T{-10000.f});
  const float first = 1.f / (1.f + std::exp(1.f));
  for (int row = 0; row < rows; ++row) {
    for (int rank = 0; rank < k; ++rank) {
      const int expert = maximum ? rank : nonfinite ? std::array<int, 4>{1, 2, 0, 3}[rank]
                                                    : rank;
      const float score = maximum || nonfinite ? 0.5f : rank == 0 ? first
                                                                  : 0.5f;
      ids.push_back(expert);
      weights.push_back(maximum || nonfinite ? 2.5f / k : score / (first + 0.5f) * 2.5f);
      log_scores[row * experts + expert] = T{std::log(score)};
    }
  }
  for (bool cuda : {false, true}) {
    auto provider = cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
    if (!provider || (cuda && std::is_same_v<T, BFloat16> && !CudaHasBF16Support())) continue;
    SCOPED_TRACE(provider->Type());
    SCOPED_TRACE("projected=" + std::to_string(projected) + ", rows=" + std::to_string(rows));
    OpTester test("MoEGroupRouter", 1, kMSDomain);
    test.AddAttribute("top_k", int64_t{k});
    test.AddAttribute("router_scale", 2.5f);
    test.AddInput<T>("logits", {rows, projected ? width : experts}, input);
    test.AddInput<float>("expert_bias", {experts}, bias, true);
    if (projected) test.AddInput<T>("projection_weight", {experts, width}, projection, true);
    if (dense) {
      test.AddOutput<T>("router_probs", {rows, experts}, log_scores);
    } else {
      test.AddOptionalOutputEdge<T>();
    }
    test.AddOutput<int32_t>("expert_indices", {rows, k}, ids);
    test.AddOutput<float>("expert_weights", {rows, k}, weights);
    test.SetOutputTolerance(0.001f, 0.001f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

template <typename T>
void RunCompactPaths() {
  for (bool projected : {false, true}) {
    for (int rows : {0, 1, 8}) {
      for (bool dense : {false, true}) RunCompactCase<T>(projected, rows, dense);
    }
    RunCompactCase<T>(projected, 1, false, false, true);
  }
  RunCompactCase<T>(true, 1, false, true);
}

void RunCompactFailure(int experts, int k, int n_group, int topk_group,
                       bool projected, bool compact, float scale, const std::string& error) {
  for (bool cuda : {false, true}) {
    auto provider = cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
    if (!provider) continue;
    SCOPED_TRACE(provider->Type());
    SCOPED_TRACE(testing::Message() << "experts=" << experts << " k=" << k
                                    << " n_group=" << n_group << " topk_group=" << topk_group);
    const int width = projected ? 256 : experts;
    OpTester test("MoEGroupRouter", 1, kMSDomain);
    test.AddAttribute("top_k", int64_t{k});
    test.AddAttribute("n_group", int64_t{n_group});
    test.AddAttribute("topk_group", int64_t{topk_group});
    test.AddAttribute("group_topk", int64_t{1});
    test.AddAttribute("router_scale", scale);
    test.AddInput<float>("logits", {1, width}, std::vector<float>(width, 0.f));
    test.AddInput<float>("expert_bias", {experts}, std::vector<float>(experts, 0.f));
    if (projected) {
      test.AddInput<float>("projection_weight", {experts, width},
                           std::vector<float>(experts * width, 0.f));
    }
    test.AddOutput<float>("router_probs", {1, experts}, std::vector<float>(experts, 0.f));
    if (compact) {
      test.AddOutput<int32_t>("expert_indices", {1, k}, std::vector<int32_t>(k, 0));
      test.AddOutput<float>("expert_weights", {1, k}, std::vector<float>(k, 0.f));
    }
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(provider));
    test.Run(OpTester::ExpectResult::kExpectFailure, error, {}, nullptr, &eps);
  }
}

template <typename T>
void RunFullScoresCase(bool projected, int rows, int width, int experts, int k, bool nonfinite = false) {
  std::mt19937 generator(20261111 + width + experts);
  std::uniform_real_distribution<float> distribution(-0.125f, 0.125f);
  const int input_width = projected ? width : experts;
  std::vector<T> input(rows * input_width), projection(experts * width);
  std::vector<float> bias(experts), scores(rows * experts), mixing(rows * experts);
  for (auto& value : input) value = T{distribution(generator)};
  for (auto& value : projection) value = T{distribution(generator)};
  for (auto& value : bias) value = distribution(generator);
  if (nonfinite) {
    ASSERT_FALSE(projected);
    ASSERT_EQ(experts, 4);
    const float inf = std::numeric_limits<float>::infinity();
    input = {T{inf}, T{-inf}, T{NAN}, T{0.f}};
    bias = {0.f, inf, -inf, NAN};
  }
  for (int row = 0; row < rows; ++row) {
    for (int expert = 0; expert < experts; ++expert) {
      double logit = 0;
      if (projected) {
        for (int col = 0; col < width; ++col) {
          logit += static_cast<double>(static_cast<float>(input[row * width + col])) *
                   static_cast<float>(projection[expert * width + col]);
        }
      } else {
        logit = static_cast<float>(input[row * experts + expert]);
      }
      const size_t index = static_cast<size_t>(row * experts + expert);
      mixing[index] = static_cast<float>(1.0 / (1.0 + std::exp(-logit)));
      scores[index] = mixing[index] + bias[expert];
    }
  }
  for (bool cuda : {false, true}) {
    auto provider = cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
    if (!provider || (cuda && std::is_same_v<T, BFloat16> && !CudaHasBF16Support())) continue;
    SCOPED_TRACE(provider->Type());
    SCOPED_TRACE(testing::Message() << "projected=" << projected << " rows=" << rows
                                    << " width=" << width << " experts=" << experts << " k=" << k);
    OpTester test("MoEGroupRouter", 1, kMSDomain);
    test.AddAttribute("top_k", int64_t{k});
    test.AddInput<T>("logits", {rows, input_width}, input);
    test.AddInput<float>("expert_bias", {experts}, bias, true);
    if (projected) test.AddInput<T>("projection_weight", {experts, width}, projection, true);
    test.AddOptionalOutputEdge<T>();
    test.AddOptionalOutputEdge<int32_t>();
    test.AddOptionalOutputEdge<float>();
    test.AddOutput<float>("selection_scores", {rows, experts}, scores);
    test.AddOutput<float>("router_weights", {rows, experts}, mixing);
    test.SetOutputTolerance(1e-6f, 1e-6f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(provider));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

template <typename T>
void RunFullScoresPaths() {
  for (int rows : {0, 1, 3}) {
    RunFullScoresCase<T>(false, rows, 0, 9, 1);
    RunFullScoresCase<T>(true, rows, 256, 9, 1);
    RunFullScoresCase<T>(true, rows, 512, 33, 9);
  }
  RunFullScoresCase<T>(true, 3, 2048, 512, 32);
  // Neither truncation to k nor a normalization may be applied to the full arrays.
  RunFullScoresCase<T>(true, 1, 256, 33, 33);
  RunFullScoresCase<T>(false, 1, 0, 4, 2, true);
}

void RunFullScoresFailure(bool missing_scores, bool missing_weights, bool legacy,
                          int n_group, float scale, const std::string& error) {
  for (bool cuda : {false, true}) {
    auto provider = cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
    if (!provider) continue;
    SCOPED_TRACE(provider->Type());
    OpTester test("MoEGroupRouter", 1, kMSDomain);
    test.AddAttribute("top_k", int64_t{1});
    test.AddAttribute("n_group", int64_t{n_group});
    test.AddAttribute("router_scale", scale);
    test.AddInput<float>("logits", {1, 4}, std::vector<float>(4, 0.f));
    test.AddInput<float>("expert_bias", {4}, std::vector<float>(4, 0.f));
    if (legacy) {
      test.AddOutput<float>("router_probs", {1, 4}, std::vector<float>(4, 0.f));
    } else {
      test.AddOptionalOutputEdge<float>();
    }
    test.AddOptionalOutputEdge<int32_t>();
    test.AddOptionalOutputEdge<float>();
    if (missing_scores) {
      test.AddOptionalOutputEdge<float>();
    } else {
      test.AddOutput<float>("selection_scores", {1, 4}, std::vector<float>(4, .5f));
    }
    if (!missing_weights) test.AddOutput<float>("router_weights", {1, 4}, std::vector<float>(4, .5f));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(provider));
    test.Run(OpTester::ExpectResult::kExpectFailure, error, {}, nullptr, &eps);
  }
}

}  // namespace

TEST(MoEGroupRouterTest, FullScoresFloat) { RunFullScoresPaths<float>(); }
TEST(MoEGroupRouterTest, FullScoresFloat16) { RunFullScoresPaths<MLFloat16>(); }
TEST(MoEGroupRouterTest, FullScoresBFloat16) { RunFullScoresPaths<BFloat16>(); }

TEST(MoEGroupRouterTest, FullScoresRejectIncompatibleOutputsAndAttributes) {
  RunFullScoresFailure(true, false, false, 1, 1.f, "must be requested together");
  RunFullScoresFailure(false, true, false, 1, 1.f, "must be requested together");
  RunFullScoresFailure(false, false, true, 1, 1.f, "cannot be combined");
  RunFullScoresFailure(false, false, false, 2, 1.f, "requires one group");
  RunFullScoresFailure(false, false, false, 1, 2.f, "requires router_scale=1");
}

TEST(MoEGroupRouterTest, CompactAndProjectedFloat) { RunCompactPaths<float>(); }
TEST(MoEGroupRouterTest, CompactAndProjectedFloat16) { RunCompactPaths<MLFloat16>(); }
TEST(MoEGroupRouterTest, CompactAndProjectedBFloat16) { RunCompactPaths<BFloat16>(); }

TEST(MoEGroupRouterTest, CompactOutputsMustBePaired) {
  for (bool cuda : {false, true}) {
    auto provider = cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider();
    if (!provider) continue;
    for (bool missing_indices : {false, true}) {
      OpTester test("MoEGroupRouter", 1, kMSDomain);
      test.AddAttribute("top_k", int64_t{1});
      test.AddInput<float>("logits", {1, 4}, std::vector<float>(4, 0.f));
      test.AddInput<float>("expert_bias", {4}, std::vector<float>(4, 0.f));
      test.AddOptionalOutputEdge<float>();
      if (missing_indices) {
        test.AddOptionalOutputEdge<int32_t>();
        test.AddOutput<float>("expert_weights", {1, 1}, {1.f});
      } else {
        test.AddOutput<int32_t>("expert_indices", {1, 1}, {0});
      }
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      // Re-create the provider for each session.
      eps.push_back(cuda ? DefaultCudaExecutionProvider() : DefaultCpuExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectFailure,
               "expert_indices and expert_weights must be requested together", {}, nullptr, &eps);
    }
  }
}

TEST(MoEGroupRouterTest, CompactRejectsGroupedRouting) {
  for (int topk_group : {1, 2}) {
    RunCompactFailure(4, 1, 2, topk_group, false, true, 1.f, "compact routing requires one group");
  }
}

TEST(MoEGroupRouterTest, ProjectionRequiresCompactOutputs) {
  RunCompactFailure(4, 1, 1, 1, true, false, 1.f, "projection_weight requires compact outputs");
}

TEST(MoEGroupRouterTest, ScaleRequiresCompactOutputs) {
  RunCompactFailure(4, 1, 1, 1, false, false, 1.5f, "router_scale requires compact outputs");
}

TEST(MoEGroupRouterTest, CompactRejectsNonprojectedBounds) {
  RunCompactFailure(64, 33, 1, 1, false, true, 1.f, "top_k<=32");
  RunCompactFailure(513, 1, 1, 1, false, true, 1.f, "512");
}

TEST(MoEGroupRouterTest, GeneralAndSpecializedPathsFloat) {
  RunRouterPaths<float>();
}

TEST(MoEGroupRouterTest, GeneralAndSpecializedPathsFloat16) {
  RunRouterPaths<MLFloat16>();
}

TEST(MoEGroupRouterTest, GeneralAndSpecializedPathsBFloat16) {
  RunRouterPaths<BFloat16>();
}

TEST(MoEGroupRouterTest, NonPositiveAttributesAreRejected) {
  for (bool use_cuda : {false, true}) {
    if (use_cuda && !DefaultCudaExecutionProvider()) {
      continue;
    }
    for (auto attribute : kRouterAttributes) {
      for (int64_t value : {int64_t{0}, int64_t{-1}, std::numeric_limits<int64_t>::min()}) {
        RunRouterAttributeFailure(attribute, value, use_cuda, std::string(attribute) + " must be positive");
      }
    }
  }
}

TEST(MoEGroupRouterTest, CudaAttributesAboveIntMaxAreRejected) {
  if (!DefaultCudaExecutionProvider()) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  for (auto attribute : kRouterAttributes) {
    for (int64_t value : {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1,
                          (int64_t{1} << 32) + 1, std::numeric_limits<int64_t>::max()}) {
      RunRouterAttributeFailure(attribute, value, true, "MoEGroupRouter CUDA attributes must not exceed INT_MAX");
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
