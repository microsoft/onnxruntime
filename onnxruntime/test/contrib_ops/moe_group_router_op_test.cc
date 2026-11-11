// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
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

}  // namespace

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
