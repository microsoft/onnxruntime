// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace onnxruntime::test {
#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace {
void ConfigureIQ2QMoE(OpTester& tester, int topk, const char* activation = "relu2") {
  tester.AddAttribute("quant_type", std::string("iq2_xxs"));
  tester.AddAttribute("activation_type", std::string(activation));
  tester.AddAttribute("expert_weight_bits", int64_t{2});
  tester.AddAttribute("block_size", int64_t{256});
  tester.AddAttribute("weights_prepacked", int64_t{0});
  tester.AddAttribute("k", static_cast<int64_t>(topk));
}

void AddPreselectedRoutes(OpTester& tester, int rows, int topk,
                          const std::vector<int32_t>& indices, const std::vector<float>& weights) {
  for (int index = 7; index <= 20; ++index) tester.AddOptionalInputEdge<BFloat16>();
  tester.AddInput<int32_t>("indices", {rows, topk}, indices);
  tester.AddInput<float>("weights", {rows, topk}, weights);
}

template <typename Scale>
void TestReusedExperts(int rows, const std::string& expected_error = "",
                       bool fused_router = false, int router_width = 256, float router_scale = 5.f) {
  constexpr int intermediate = 256, topk = 2;
  const int experts = fused_router ? 3 : 2;
  const int pairs = rows * topk;
  OpTester tester("QMoE", 1, kMSDomain);
  ConfigureIQ2QMoE(tester, topk);
  if (fused_router) tester.AddAttribute("router_scale", router_scale);
  tester.AddInput<BFloat16>("X", {rows, 1024}, std::vector<BFloat16>(rows * 1024, BFloat16(0.03125f)));
  std::vector<int32_t> indices(pairs);
  std::vector<float> weights(pairs);
  for (int r = 0; r < rows; ++r) {
    indices[r * 2] = fused_router ? 2 : r & 1;
    indices[r * 2 + 1] = fused_router ? 0 : 1 - (r & 1);
    weights[r * 2] = fused_router ? router_scale / 2.f : (r & 1) ? 2.f
                                                                 : 0.5f;
    weights[r * 2 + 1] = fused_router ? router_scale / 2.f : (r & 1) ? 0.5f
                                                                     : 2.f;
  }
  tester.AddOptionalInputEdge<BFloat16>();
  // Grid one is [43,8,8,8,8,8,8,8]. Scale 0.019 distinguishes full-operand from scale-first rounding.
  auto make_packed = [](int bytes) {
    std::vector<uint8_t> data(bytes, 0);
    for (int record = 0; record < bytes; record += 8) {
      for (int i = 0; i < 4; ++i) data[record + i] = 1;
    }
    return data;
  };
  tester.AddInput<uint8_t>("fc1", {experts, intermediate, 256}, make_packed(experts * intermediate * 256));
  auto make_scale = [experts](int elements) {
    std::vector<Scale> data;
    data.reserve(elements * experts);
    for (int e = 0; e < experts; ++e) {
      const auto bits = MLFloat16(e == 0 ? 0.019f : 0.023f).val;
      data.insert(data.end(), elements, Scale::FromBits(bits));
    }
    return data;
  };
  tester.AddInput<Scale>("scale1", {experts, intermediate, 4}, make_scale(intermediate * 4));
  tester.AddOptionalInputEdge<BFloat16>();
  tester.AddInput<uint8_t>("fc2", {experts, 1024, 64}, make_packed(experts * 1024 * 64));
  tester.AddInput<Scale>("scale2", {experts, 1024, 1}, make_scale(1024));
  if (fused_router) {
    for (int index = 7; index <= 22; ++index) tester.AddOptionalInputEdge<BFloat16>();
    tester.AddInput<BFloat16>("router_input", {rows, router_width},
                              std::vector<BFloat16>(rows * router_width, BFloat16(1.f)));
    tester.AddInput<BFloat16>("router_weight", {experts, router_width},
                              std::vector<BFloat16>(experts * router_width, BFloat16(0.f)));
    // Equal sigmoid scores select experts 2 and 0, but their mixing weights remain equal.
    tester.AddInput<float>("router_bias", {experts}, {1.f, -2.f, 2.f});
  } else {
    AddPreselectedRoutes(tester, rows, topk, indices, weights);
  }
  std::vector<float> expert_hidden(experts), expert_partial(experts);
  for (int e = 0; e < experts; ++e) {
    const float d = MLFloat16(e == 0 ? 0.019f : 0.023f).ToFloat();
    const float sum8 = BFloat16(d * 0.125f * 43.f).ToFloat() + 7.f * BFloat16(d).ToFloat();
    const float fc1 = BFloat16(4.f * sum8).ToFloat();
    expert_hidden[e] = BFloat16(fc1 * fc1).ToFloat();
    expert_partial[e] = BFloat16(32.f * expert_hidden[e] * sum8).ToFloat();
  }
  const BFloat16 sum(fused_router ? router_scale / 2.f * expert_partial[2] + router_scale / 2.f * expert_partial[0]
                                  : 0.5f * expert_partial[0] + 2.f * expert_partial[1]);
  tester.AddOutput<BFloat16>("Y", {rows, 1024}, std::vector<BFloat16>(rows * 1024, sum));
  tester.SetOutputTolerance(0.f, 0.f);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(expected_error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
             expected_error, {}, nullptr, &providers);
}

void RunCuda(OpTester& tester) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

class MoETester : public OpTester {
 public:
  MoETester() : OpTester("QMoE", 1, kMSDomain) {
    SetCustomOutputVerifier([this](const std::vector<OrtValue>& actual, const std::string& provider) {
      auto& expected = GetOutputData();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        CheckOrtValuesAreEqual(expected[i].def.Name(), expected[i].data, actual[i], expected[i].validation_params,
                               provider);
        const auto& want = expected[i].data.Get<Tensor>();
        if (!want.IsDataType<BFloat16>()) continue;
        const auto& got = actual[i].Get<Tensor>();
        ASSERT_EQ(want.Shape(), got.Shape());
        ASSERT_TRUE(got.IsDataType<BFloat16>());
        // The base BF16 checker checks the expected, not actual, NaN.
        for (int64_t j = 0; j < want.Shape().Size(); ++j) {
          if (std::isnan(want.Data<BFloat16>()[j].ToFloat())) {
            EXPECT_TRUE(std::isnan(got.Data<BFloat16>()[j].ToFloat())) << expected[i].def.Name() << ", " << j;
          }
        }
      }
    });
  }
};

// Independently decode grid one, including sign parity and the raw FP16 super-scale.
float DecodeFixture(const std::vector<uint8_t>& packed, const std::vector<MLFloat16>& scales, int row, int width,
                    int k) {
  const int record = (row * (width / 32) + k / 32) * 8;
  const int group = (k % 32) / 8, lane = k % 8;
  uint32_t bits = 0;
  for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(packed[record + 4 + b]) << (8 * b);
  const int signs = (bits >> (group * 7)) & 127;
  int parity = 0;
  for (int b = 0; b < 7; ++b) parity ^= (signs >> b) & 1;
  const bool negative = lane == 7 ? parity != 0 : ((signs >> lane) & 1) != 0;
  const float magnitude = packed[record + group] == 1 && lane == 0 ? 43.f : 8.f;
  const float scale = scales[row * (width / 256) + k / 256].ToFloat() * (0.5f + static_cast<float>(bits >> 28)) * 0.25f;
  return BFloat16((negative ? -magnitude : magnitude) * scale).ToFloat();
}

void TestSparseRoutes(int topk, int invalid_id = -2, bool duplicate = false) {
  constexpr int rows = 3, intermediate = 512;
  const int experts = topk + 3, pairs = rows * topk;
  std::vector<BFloat16> x(rows * 1024);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < 1024; ++k) x[r * 1024 + k] = BFloat16(k % 8 == 0 ? (r + 1) * (1 + k / 256) / 1024.f : 0.f);
  std::vector<int32_t> indices(pairs);
  std::vector<float> weights(pairs);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < topk; ++k) {
      indices[r * topk + k] = (r * 2 + k) % experts;
      weights[r * topk + k] = (k + 1) * 0.125f;
    }
  if (invalid_id != -2) indices[topk] = invalid_id;
  if (duplicate) indices[topk + 1] = indices[topk];
  auto make_packed = [](int count, int width, bool signed_weights) {
    std::vector<uint8_t> data(static_cast<size_t>(count) * (width / 32) * 8);
    for (int n = 0; n < count; ++n)
      for (int b = 0; b < width / 32; ++b) {
        const int offset = (n * (width / 32) + b) * 8;
        std::fill_n(data.begin() + offset, 4, uint8_t{1});
        // All eight signs are negative (seven bits plus odd parity).
        if (signed_weights && (n + b / 8) % 3 == 0) {
          data[offset + 4] = data[offset + 5] = data[offset + 6] = 255;
          data[offset + 7] = 15;
        }
      }
    return data;
  };
  auto make_scales = [experts](int height, int blocks) {
    std::vector<MLFloat16> scales(experts * height * blocks);
    for (int e = 0; e < experts; ++e)
      for (int n = 0; n < height; ++n)
        for (int b = 0; b < blocks; ++b)
          scales[(e * height + n) * blocks + b] = MLFloat16((1 + e % 3 + n % 2 + b) / 64.f);
    return scales;
  };
  const auto fc1 = make_packed(experts * intermediate, 1024, false);
  const auto fc2 = make_packed(experts * 1024, intermediate, true);
  const auto sc1 = make_scales(intermediate, 4), sc2 = make_scales(1024, intermediate / 256);
  std::vector<BFloat16> hidden(pairs * intermediate), y(rows * 1024);
  std::vector<float> part(pairs * 1024);
  for (int r = 0; r < rows; ++r) {
    const bool valid = !(r == 1 && (invalid_id != -2 || duplicate));
    for (int k = 0; k < topk; ++k) {
      const int p = r * topk + k, e = indices[p];
      if (!valid) {
        std::fill_n(hidden.begin() + p * intermediate, intermediate, BFloat16(std::numeric_limits<float>::quiet_NaN()));
        std::fill_n(part.begin() + p * 1024, 1024, std::numeric_limits<float>::quiet_NaN());
        continue;
      }
      for (int n = 0; n < intermediate; ++n) {
        float dot = 0.f;
        for (int i = 0; i < 1024; ++i)
          dot = std::fma(x[r * 1024 + i].ToFloat(), DecodeFixture(fc1, sc1, e * intermediate + n, 1024, i), dot);
        const float relu = std::max(BFloat16(dot).ToFloat(), 0.f);
        hidden[p * intermediate + n] = BFloat16(relu * relu);
      }
      for (int n = 0; n < 1024; ++n) {
        float dot = 0.f;
        for (int i = 0; i < intermediate; ++i)
          dot = std::fma(hidden[p * intermediate + i].ToFloat(), DecodeFixture(fc2, sc2, e * 1024 + n, intermediate, i),
                         dot);
        part[p * 1024 + n] = BFloat16(dot).ToFloat();
      }
    }
    for (int n = 0; n < 1024; ++n) {
      float sum = 0.f;
      for (int k = 0; k < topk; ++k) sum = std::fma(weights[r * topk + k], part[(r * topk + k) * 1024 + n], sum);
      y[r * 1024 + n] = BFloat16(sum);
    }
  }
  MoETester tester;
  ConfigureIQ2QMoE(tester, topk);
  tester.AddInput<BFloat16>("X", {rows, 1024}, x);
  tester.AddOptionalInputEdge<BFloat16>();
  tester.AddInput<uint8_t>("fc1", {experts, intermediate, 256}, fc1);
  tester.AddInput<MLFloat16>("scale1", {experts, intermediate, 4}, sc1);
  tester.AddOptionalInputEdge<BFloat16>();
  tester.AddInput<uint8_t>("fc2", {experts, 1024, intermediate / 4}, fc2);
  tester.AddInput<MLFloat16>("scale2", {experts, 1024, intermediate / 256}, sc2);
  AddPreselectedRoutes(tester, rows, topk, indices, weights);
  tester.AddOutput<BFloat16>("Y", {rows, 1024}, y);
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester);
}

void TestOrderedCancellation(int topk) {
  const int experts = topk + 1;
  OpTester tester("QMoE", 1, kMSDomain);
  ConfigureIQ2QMoE(tester, topk);
  tester.AddInput<BFloat16>("X", {1, 1024}, std::vector<BFloat16>(1024, BFloat16(1.f / 1024)));
  std::vector<int32_t> indices(topk);
  for (int k = 0; k < topk; ++k) indices[k] = experts - 1 - k;
  std::vector<float> weights(topk, 0.f);
  weights[0] = 0x1p28f;
  weights[1] = 48.f;
  weights[2] = -0x1p28f;
  tester.AddOptionalInputEdge<BFloat16>();
  // Grid zero has eight 8s: decoded operands are exactly 1/16.
  tester.AddInput<uint8_t>("fc1", {experts, 256, 256}, std::vector<uint8_t>(experts * 256 * 256, 0));
  tester.AddInput<MLFloat16>("scale1", {experts, 256, 4},
                             std::vector<MLFloat16>(experts * 256 * 4, MLFloat16(0.0625f)));
  tester.AddOptionalInputEdge<BFloat16>();
  tester.AddInput<uint8_t>("fc2", {experts, 1024, 64}, std::vector<uint8_t>(experts * 1024 * 64, 0));
  tester.AddInput<MLFloat16>("scale2", {experts, 1024, 1}, std::vector<MLFloat16>(experts * 1024, MLFloat16(0.0625f)));
  AddPreselectedRoutes(tester, 1, topk, indices, weights);
  float ordered = 0.f, reversed = 0.f, bf16_sum = 0.f;
  for (int k = 0; k < topk; ++k) {
    ordered = std::fma(weights[k], 0.0625f, ordered);
    reversed = std::fma(weights[topk - 1 - k], 0.0625f, reversed);
    bf16_sum = BFloat16(std::fma(weights[k], 0.0625f, bf16_sum)).ToFloat();
  }
  ASSERT_EQ(ordered, 4.f);
  ASSERT_NE(BFloat16(ordered).val, BFloat16(reversed).val);
  ASSERT_NE(BFloat16(ordered).val, BFloat16(bf16_sum).val);
  tester.AddOutput<BFloat16>("Y", {1, 1024}, std::vector<BFloat16>(1024, BFloat16(ordered)));
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester);
}
}  // namespace

TEST(QMoEIQ2XXSTest, ReusedExpertsAndIndependentFP16Scales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 8}) TestReusedExperts<MLFloat16>(rows);
}

TEST(QMoEIQ2XXSTest, FusedBiasCorrectedRouterPreservesUncorrectedWeightsAndScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 8}) TestReusedExperts<MLFloat16>(rows, "", true);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(QMoEIQ2XXSTest, FusedRouterRejectsUnsupportedWidthAndNonfiniteScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestReusedExperts<MLFloat16>(1, "positive int32 H divisible by 256", true, 128);
  TestReusedExperts<MLFloat16>(1, "router_scale must be finite", true, 256,
                               std::numeric_limits<float>::quiet_NaN());
}
#endif

TEST(QMoEIQ2XXSTest, SparseRoutesAndOrderedFP32Accumulation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSparseRoutes(2);
  TestSparseRoutes(9);
  TestOrderedCancellation(3);
  TestOrderedCancellation(9);
}

TEST(QMoEIQ2XXSTest, InvalidAndDuplicateRoutesProduceNaNRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSparseRoutes(2, -1);
  TestSparseRoutes(2, 5);
  TestSparseRoutes(2, -2, true);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(QMoEIQ2XXSTest, RejectsUnsupportedScaleTypeAndRowCount) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestReusedExperts<BFloat16>(1, "requires FP16 super-scales");
  TestReusedExperts<MLFloat16>(9, "requires X[R,1024]");
}

TEST(QMoEIQ2XXSTest, RejectsUnsupportedCPUInsteadOfIntegerDecoding) {
  OpTester tester("QMoE", 1, kMSDomain);
  ConfigureIQ2QMoE(tester, 1, "relu");
  tester.AddInput<MLFloat16>("X", {1, 1024}, std::vector<MLFloat16>(1024, MLFloat16(1.f / 1024)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc1", {1, 256, 256}, std::vector<uint8_t>(256 * 256, 0));
  tester.AddInput<MLFloat16>("scale1", {1, 256, 4}, std::vector<MLFloat16>(256 * 4, MLFloat16(0.0625f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<uint8_t>("fc2", {1, 1024, 64}, std::vector<uint8_t>(1024 * 64, 0));
  tester.AddInput<MLFloat16>("scale2", {1, 1024, 1}, std::vector<MLFloat16>(1024, MLFloat16(0.0625f)));
  AddPreselectedRoutes(tester, 1, 1, {0}, {1.f});
  tester.AddOutput<MLFloat16>("Y", {1, 1024}, std::vector<MLFloat16>(1024, MLFloat16(0.0625f)));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCpuExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectFailure, "CPU QMoE does not support IQ2_XXS",
             {}, nullptr, &providers);
}
#endif
#endif
}  // namespace onnxruntime::test
