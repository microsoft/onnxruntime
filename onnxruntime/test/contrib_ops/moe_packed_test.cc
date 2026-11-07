// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace {
void Configure(OpTester& tester, int topk = 2, const char* codec = "iq2_xxs") {
  tester.AddAttribute("quant_type", std::string(codec));
  tester.AddAttribute("k", static_cast<int64_t>(topk));
}

std::vector<uint8_t> NativeBlocks(const std::vector<uint8_t>& records,
                                  const std::vector<MLFloat16>& scales) {
  EXPECT_EQ(records.size(), scales.size() * 64);
  std::vector<uint8_t> result(scales.size() * 66);
  for (size_t block = 0; block < scales.size(); ++block) {
    result[block * 66] = static_cast<uint8_t>(scales[block].val);
    result[block * 66 + 1] = static_cast<uint8_t>(scales[block].val >> 8);
    std::copy_n(records.begin() + block * 64, 64, result.begin() + block * 66 + 2);
  }
  return result;
}

void RunCuda(OpTester& tester, const std::string& error = "") {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(error.empty() ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure,
             error, {}, nullptr, &providers);
}

void TestReusedExperts(int rows, bool fused = false, const std::string& error = "",
                       int width = 256, float scale = 5.f) {
  constexpr int intermediate = 256, topk = 2;
  const int experts = fused ? 3 : 2;
  OpTester tester("MoEPacked", 1, kMSDomain);
  Configure(tester);
  if (fused) tester.AddAttribute("router_scale", scale);
  tester.AddInput<BFloat16>("X", {rows, 1024}, std::vector<BFloat16>(rows * 1024, BFloat16(0.03125f)));
  auto records = [](int bytes) {
    std::vector<uint8_t> data(bytes, 0);
    for (int record = 0; record < bytes; record += 8)
      std::fill_n(data.begin() + record, 4, uint8_t{1});
    return data;
  };
  auto scales = [experts](int per_expert) {
    std::vector<MLFloat16> data;
    for (int e = 0; e < experts; ++e)
      data.insert(data.end(), per_expert, MLFloat16(e == 0 ? 0.019f : 0.023f));
    return data;
  };
  tester.AddInput<uint8_t>("FC1", {experts, intermediate, 264},
                           NativeBlocks(records(experts * intermediate * 256), scales(intermediate * 4)));
  tester.AddInput<uint8_t>("FC2", {experts, 1024, 66},
                           NativeBlocks(records(experts * 1024 * 64), scales(1024)));
  if (fused) {
    tester.AddOptionalInputEdge<int32_t>();
    tester.AddOptionalInputEdge<float>();
    tester.AddInput<BFloat16>("router_input", {rows, width},
                              std::vector<BFloat16>(rows * width, BFloat16(1.f)));
    tester.AddInput<BFloat16>("router_weight", {experts, width},
                              std::vector<BFloat16>(experts * width, BFloat16(0.f)));
    tester.AddInput<float>("router_bias", {experts}, {1.f, -2.f, 2.f});
  } else {
    std::vector<int32_t> indices(rows * topk);
    std::vector<float> weights(rows * topk);
    for (int r = 0; r < rows; ++r) {
      indices[r * 2] = r & 1;
      indices[r * 2 + 1] = 1 - (r & 1);
      weights[r * 2] = (r & 1) ? 2.f : 0.5f;
      weights[r * 2 + 1] = (r & 1) ? 0.5f : 2.f;
    }
    tester.AddInput<int32_t>("indices", {rows, topk}, indices);
    tester.AddInput<float>("weights", {rows, topk}, weights);
  }
  std::vector<float> partial(experts);
  for (int e = 0; e < experts; ++e) {
    const float d = MLFloat16(e == 0 ? 0.019f : 0.023f).ToFloat();
    const float sum8 = BFloat16(d * 0.125f * 43.f).ToFloat() + 7.f * BFloat16(d).ToFloat();
    const float projected = BFloat16(4.f * sum8).ToFloat();
    const float hidden = BFloat16(projected * projected).ToFloat();
    partial[e] = BFloat16(32.f * hidden * sum8).ToFloat();
  }
  const BFloat16 y(fused ? scale / 2.f * partial[2] + scale / 2.f * partial[0]
                         : 0.5f * partial[0] + 2.f * partial[1]);
  tester.AddOutput<BFloat16>("Y", {rows, 1024}, std::vector<BFloat16>(rows * 1024, y));
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester, error);
}

class NaNTester : public OpTester {
 public:
  NaNTester() : OpTester("MoEPacked", 1, kMSDomain) {
    SetCustomOutputVerifier([this](const std::vector<OrtValue>& actual, const std::string& provider) {
      auto& expected = GetOutputData();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        CheckOrtValuesAreEqual(expected[i].def.Name(), expected[i].data, actual[i],
                               expected[i].validation_params, provider);
        const auto& want = expected[i].data.Get<Tensor>();
        const auto& got = actual[i].Get<Tensor>();
        ASSERT_EQ(want.Shape(), got.Shape());
        ASSERT_TRUE(got.IsDataType<BFloat16>());
        for (int64_t j = 0; j < want.Shape().Size(); ++j) {
          if (std::isnan(want.Data<BFloat16>()[j].ToFloat())) {
            EXPECT_TRUE(std::isnan(got.Data<BFloat16>()[j].ToFloat())) << j;
          }
        }
      }
    });
  }
};

// Independent fixture decoder: grid one, sign parity, FP16 scale and complete-operand rounding.
float Decode(const std::vector<uint8_t>& records, const std::vector<MLFloat16>& scales,
             int row, int width, int k) {
  const int record = (row * (width / 32) + k / 32) * 8;
  const int group = (k % 32) / 8, lane = k % 8;
  uint32_t bits = 0;
  for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(records[record + 4 + b]) << (8 * b);
  const int signs = (bits >> (group * 7)) & 127;
  int parity = 0;
  for (int b = 0; b < 7; ++b) parity ^= (signs >> b) & 1;
  const bool negative = lane == 7 ? parity != 0 : ((signs >> lane) & 1) != 0;
  const float magnitude = records[record + group] == 1 && lane == 0 ? 43.f : 8.f;
  const float scale = scales[row * (width / 256) + k / 256].ToFloat() *
                      (0.5f + static_cast<float>(bits >> 28)) * 0.25f;
  return BFloat16((negative ? -magnitude : magnitude) * scale).ToFloat();
}

void TestSparse(int topk, int invalid_id = -2, bool duplicate = false) {
  constexpr int rows = 3, intermediate = 512;
  const int experts = topk + 3, pairs = rows * topk;
  std::vector<BFloat16> x(rows * 1024);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < 1024; ++k)
      x[r * 1024 + k] = BFloat16(k % 8 == 0 ? (r + 1) * (1 + k / 256) / 1024.f : 0.f);
  std::vector<int32_t> indices(pairs);
  std::vector<float> weights(pairs);
  for (int r = 0; r < rows; ++r)
    for (int k = 0; k < topk; ++k) {
      indices[r * topk + k] = (r * 2 + k) % experts;
      weights[r * topk + k] = (k + 1) * 0.125f;
    }
  if (invalid_id != -2) indices[topk] = invalid_id;
  if (duplicate) indices[topk + 1] = indices[topk];
  auto records = [](int count, int width, bool negative) {
    std::vector<uint8_t> data(static_cast<size_t>(count) * (width / 32) * 8);
    for (int n = 0; n < count; ++n)
      for (int b = 0; b < width / 32; ++b) {
        const int offset = (n * (width / 32) + b) * 8;
        std::fill_n(data.begin() + offset, 4, uint8_t{1});
        if (negative && (n + b / 8) % 3 == 0) {
          data[offset + 4] = data[offset + 5] = data[offset + 6] = 255;
          data[offset + 7] = 15;
        }
      }
    return data;
  };
  auto scales = [experts](int height, int blocks) {
    std::vector<MLFloat16> data(experts * height * blocks);
    for (int e = 0; e < experts; ++e)
      for (int n = 0; n < height; ++n)
        for (int b = 0; b < blocks; ++b)
          data[(e * height + n) * blocks + b] = MLFloat16((1 + e % 3 + n % 2 + b) / 64.f);
    return data;
  };
  const auto fc1 = records(experts * intermediate, 1024, false);
  const auto fc2 = records(experts * 1024, intermediate, true);
  const auto sc1 = scales(intermediate, 4), sc2 = scales(1024, intermediate / 256);
  std::vector<BFloat16> hidden(pairs * intermediate), y(rows * 1024);
  std::vector<float> part(pairs * 1024);
  for (int r = 0; r < rows; ++r) {
    const bool valid = !(r == 1 && (invalid_id != -2 || duplicate));
    for (int k = 0; k < topk; ++k) {
      const int p = r * topk + k, e = indices[p];
      if (!valid) {
        std::fill_n(part.begin() + p * 1024, 1024, std::numeric_limits<float>::quiet_NaN());
        continue;
      }
      for (int n = 0; n < intermediate; ++n) {
        float dot = 0.f;
        for (int i = 0; i < 1024; ++i)
          dot = std::fma(x[r * 1024 + i].ToFloat(), Decode(fc1, sc1, e * intermediate + n, 1024, i), dot);
        const float relu = std::max(BFloat16(dot).ToFloat(), 0.f);
        hidden[p * intermediate + n] = BFloat16(relu * relu);
      }
      for (int n = 0; n < 1024; ++n) {
        float dot = 0.f;
        for (int i = 0; i < intermediate; ++i)
          dot = std::fma(hidden[p * intermediate + i].ToFloat(),
                         Decode(fc2, sc2, e * 1024 + n, intermediate, i), dot);
        part[p * 1024 + n] = BFloat16(dot).ToFloat();
      }
    }
    for (int n = 0; n < 1024; ++n) {
      float sum = 0.f;
      for (int k = 0; k < topk; ++k)
        sum = std::fma(weights[r * topk + k], part[(r * topk + k) * 1024 + n], sum);
      y[r * 1024 + n] = BFloat16(sum);
    }
  }
  NaNTester tester;
  Configure(tester, topk);
  tester.AddInput<BFloat16>("X", {rows, 1024}, x);
  tester.AddInput<uint8_t>("FC1", {experts, intermediate, 264}, NativeBlocks(fc1, sc1));
  tester.AddInput<uint8_t>("FC2", {experts, 1024, (intermediate / 256) * 66}, NativeBlocks(fc2, sc2));
  tester.AddInput<int32_t>("indices", {rows, topk}, indices);
  tester.AddInput<float>("weights", {rows, topk}, weights);
  tester.AddOutput<BFloat16>("Y", {rows, 1024}, y);
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester);
}

void TestCancellation(int topk) {
  const int experts = topk + 1;
  OpTester tester("MoEPacked", 1, kMSDomain);
  Configure(tester, topk);
  tester.AddInput<BFloat16>("X", {1, 1024}, std::vector<BFloat16>(1024, BFloat16(1.f / 1024)));
  tester.AddInput<uint8_t>("FC1", {experts, 256, 264},
                           NativeBlocks(std::vector<uint8_t>(experts * 256 * 256),
                                        std::vector<MLFloat16>(experts * 256 * 4, MLFloat16(0.0625f))));
  tester.AddInput<uint8_t>("FC2", {experts, 1024, 66},
                           NativeBlocks(std::vector<uint8_t>(experts * 1024 * 64),
                                        std::vector<MLFloat16>(experts * 1024, MLFloat16(0.0625f))));
  std::vector<int32_t> indices(topk);
  for (int k = 0; k < topk; ++k) indices[k] = experts - 1 - k;
  std::vector<float> weights(topk, 0.f);
  weights[0] = 0x1p28f;
  weights[1] = 48.f;
  weights[2] = -0x1p28f;
  float ordered = 0.f, reversed = 0.f, bf16_sum = 0.f;
  for (int k = 0; k < topk; ++k) {
    ordered = std::fma(weights[k], 0.0625f, ordered);
    reversed = std::fma(weights[topk - 1 - k], 0.0625f, reversed);
    bf16_sum = BFloat16(std::fma(weights[k], 0.0625f, bf16_sum)).ToFloat();
  }
  ASSERT_EQ(ordered, 4.f);
  ASSERT_NE(BFloat16(ordered).val, BFloat16(reversed).val);
  ASSERT_NE(BFloat16(ordered).val, BFloat16(bf16_sum).val);
  tester.AddInput<int32_t>("indices", {1, topk}, indices);
  tester.AddInput<float>("weights", {1, topk}, weights);
  tester.AddOutput<BFloat16>("Y", {1, 1024}, std::vector<BFloat16>(1024, BFloat16(ordered)));
  tester.SetOutputTolerance(0.f, 0.f);
  RunCuda(tester);
}

void TestInvalidContract(int topk, const char* codec, int mode, const std::string& error) {
  OpTester tester("MoEPacked", 1, kMSDomain);
  Configure(tester, topk, codec);
  tester.AddInput<BFloat16>("X", {1, 1024}, std::vector<BFloat16>(1024));
  tester.AddInput<uint8_t>("FC1", {2, 256, 264}, std::vector<uint8_t>(2 * 256 * 264));
  tester.AddInput<uint8_t>("FC2", {2, 1024, 66}, std::vector<uint8_t>(2 * 1024 * 66));
  if (mode == 0 || mode == 2) {
    tester.AddInput<int32_t>("indices", {1, 2}, {0, 1});
    tester.AddInput<float>("weights", {1, 2}, {0.5f, 0.5f});
  } else if (mode == 1) {
    tester.AddOptionalInputEdge<int32_t>();
    tester.AddOptionalInputEdge<float>();
  }
  if (mode == 1 || mode == 2) {
    tester.AddInput<BFloat16>("router_input", {1, 256}, std::vector<BFloat16>(256));
    if (mode == 2) {
      tester.AddInput<BFloat16>("router_weight", {2, 256}, std::vector<BFloat16>(512));
      tester.AddInput<float>("router_bias", {2}, {0.f, 0.f});
    }
  }
  tester.AddOutput<BFloat16>("Y", {1, 1024}, std::vector<BFloat16>(1024));
  RunCuda(tester, error);
}
}  // namespace

TEST(MoEPackedTest, ReusedExpertsAndFullOperandRounding) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 8}) TestReusedExperts(rows);
}

TEST(MoEPackedTest, FusedRouterPreservesUncorrectedWeightsAndScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 8}) TestReusedExperts(rows, true);
}

TEST(MoEPackedTest, SparseRoutesAndOrderedFP32Mixing) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSparse(2);
  TestSparse(9);
  TestCancellation(3);
  TestCancellation(9);
}

TEST(MoEPackedTest, InvalidAndDuplicateRoutesPoisonOnlyTheirRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSparse(2, -1);
  TestSparse(2, 5);
  TestSparse(2, -2, true);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MoEPackedTest, RejectsUnsupportedRowsWidthAndScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestReusedExperts(9, false, "requires X[R,1024]");
  TestReusedExperts(1, true, "positive int32 H divisible by 256", 128);
  TestReusedExperts(1, true, "router_scale must be finite", 256, std::numeric_limits<float>::quiet_NaN());
}

TEST(MoEPackedTest, RejectsUnsupportedCodecAndTopk) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestInvalidContract(2, "q8_0", 0, "requires quant_type='iq2_xxs'");
  TestInvalidContract(0, "iq2_xxs", 0, "1 <= k <= 32");
  TestInvalidContract(33, "iq2_xxs", 0, "1 <= k <= 32");
}

TEST(MoEPackedTest, RejectsIncompleteAndMixedRoutingModes) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestInvalidContract(2, "iq2_xxs", 1, "requires all three router inputs");
  TestInvalidContract(2, "iq2_xxs", 2, "no preselected routes");
  TestInvalidContract(2, "iq2_xxs", 3, "requires indices and weights");
}
#endif
#endif
}  // namespace onnxruntime::test
