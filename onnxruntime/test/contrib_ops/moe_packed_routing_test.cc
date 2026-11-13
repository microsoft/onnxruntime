// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

template <typename T>
struct FullRoutingInputs {
  std::vector<T> scores;
  std::vector<T> weights;
};

template <typename T>
FullRoutingInputs<T> MakeFullRoutingInputs(int rows, int experts, int k,
                                           const std::vector<int32_t>& indices,
                                           const std::vector<float>& weights) {
  FullRoutingInputs<T> result{
      std::vector<T>(rows * experts, T(-1.f)),
      std::vector<T>(rows * experts, T(std::numeric_limits<float>::quiet_NaN()))};
  for (int row = 0; row < rows; ++row) {
    for (int rank = 0; rank < k; ++rank) {
      const size_t index = static_cast<size_t>(row * experts + indices[row * k + rank]);
      result.scores.at(index) = T(static_cast<float>(k - rank));
      result.weights.at(index) = T(weights[row * k + rank]);
    }
  }
  return result;
}

void RunMoeCuda(OpTester& test) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.SetOutputTolerance(0, 0);
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  ASSERT_FALSE(test.GetFetches().empty()) << "The CUDA test must actually execute";
}

template <typename T = BFloat16, typename TR = T>
void TestFullScoreReluSquared(int rows, int topk, int malformed = 0,
                              int hidden_size = 1024,
                              const char* activation_type = "relu2", bool normalize = false,
                              int selection_case = 0, int experts = 32, int inter_size = 256,
                              int grid_offset = -1) {
  const int gate_row_bytes = (hidden_size / 256) * 66;
  std::vector<uint8_t> gate(experts * inter_size * gate_row_bytes);
  const int down_row_bytes = (inter_size / 256) * 66;
  std::vector<uint8_t> down(experts * hidden_size * down_row_bytes);
  std::vector<T> input(rows * hidden_size), expected(rows * hidden_size);
  std::vector<int32_t> indices(rows * topk);
  std::vector<float> weights(rows * topk);
  std::vector<float> sum8(experts);
  for (int expert = 0; expert < experts; ++expert) {
    const MLFloat16 d(expert & 1 ? .023f : .019f);
    const int code = grid_offset < 0 ? 1 : expert * 8 + grid_offset;
    const uint64_t grid = contrib::gguf_quant::iq2xxs_grid[code];
    for (int j = 0; j < 8; ++j) {
      sum8[expert] += T(d.ToFloat() * .125f * static_cast<float>((grid >> (8 * j)) & 255u)).ToFloat();
    }
    for (int stage = 0; stage < 2; ++stage) {
      const int n = stage == 0 ? inter_size : hidden_size;
      const int blocks = stage == 0 ? hidden_size / 256 : inter_size / 256;
      auto& bank = stage == 0 ? gate : down;
      for (int col = 0; col < n; ++col) {
        for (int block = 0; block < blocks; ++block) {
          auto* bytes = bank.data() + ((expert * n + col) * blocks + block) * 66;
          bytes[0] = static_cast<uint8_t>(d.val);
          bytes[1] = static_cast<uint8_t>(d.val >> 8);
          for (int record = 0; record < 8; ++record) {
            for (int group = 0; group < 4; ++group) bytes[2 + record * 8 + group] = static_cast<uint8_t>(code);
            if (stage == 0 && grid_offset < 0 && expert % 5 == 4) {
              for (int byte = 0; byte < 4; ++byte) {
                bytes[2 + record * 8 + 4 + byte] = static_cast<uint8_t>(0x0fffffffu >> (8 * byte));
              }
            }
          }
        }
      }
    }
  }
  const auto raw_mixing = [malformed, selection_case](int rank) {
    if (selection_case == 8) return 1.f;
    if (selection_case == 6 || selection_case == 7) return .5f;
    if (malformed == 5) return 0.f;
    if (malformed == 6) return T::FromBits(std::is_same_v<T, BFloat16> ? 0x7f7f : 0x7bff).ToFloat();
    if (malformed == 7) return rank == 0 ? .5f : -.5f;
    if (malformed == 8) return T(std::is_same_v<T, BFloat16> ? 1e-25f : 1e-4f).ToFloat();
    return static_cast<float>(TR(rank == 0 ? .3116f : .5098f));
  };
  float routing_sum = 0.f;
  for (int rank = 0; rank < topk; ++rank) routing_sum += raw_mixing(rank);
  for (int row = 0; row < rows; ++row) {
    const float factor = static_cast<float>(1 + (row & 1));
    for (int col = 0; col < hidden_size; ++col) input[row * hidden_size + col] = T(factor / 32.f);
    float mixed = 0.f;
    for (int rank = 0; rank < topk; ++rank) {
      const int expert = selection_case == 2                                                 ? std::array<int, 4>{1, 2, 0, 3}[rank]
                         : selection_case == 3 || selection_case == 7 || selection_case == 8 ? 31
                         : selection_case == 6                                               ? 16
                         : selection_case                                                    ? rank
                         : row & 1                                                           ? (experts - 1 - rank)
                                                                                             : rank;
      indices[row * topk + rank] = expert;
      const float raw_weight = raw_mixing(rank);
      weights[row * topk + rank] = raw_weight;
      const float inverse_sum = routing_sum == 0.f ? 0.f : 1.f / routing_sum;
      const float weight = normalize ? raw_weight * inverse_sum : raw_weight;
      const float fc1 = grid_offset < 0 && expert % 5 == 4
                            ? 0.f
                            : T(static_cast<float>(hidden_size / 256) * factor * sum8[expert]).ToFloat();
      const float activation = T(fc1 * fc1).ToFloat();
      const float fc2 = T(static_cast<float>(inter_size / 8) * activation * sum8[expert]).ToFloat();
      mixed = std::fma(weight, fc2, mixed);
    }
    for (int col = 0; col < hidden_size; ++col) expected[row * hidden_size + col] = T(mixed);
  }
  if (malformed >= 1 && malformed <= 4) {
    weights[0] = malformed == 1   ? std::numeric_limits<float>::infinity()
                 : malformed == 3 ? -std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    if (malformed == 4) {
      for (int row = 0; row < rows; ++row) weights[row * topk] = std::numeric_limits<float>::quiet_NaN();
    }
    for (int row = 0; row < (malformed == 4 ? rows : 1); ++row) {
      for (int col = 0; col < hidden_size; ++col) {
        expected[row * hidden_size + col] = T(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  auto routing = MakeFullRoutingInputs<TR>(rows, experts, topk, indices, weights);
  if (selection_case) {
    std::fill(routing.scores.begin(), routing.scores.end(),
              TR(selection_case == 2 || selection_case == 5 ? std::numeric_limits<float>::quiet_NaN() : 0.f));
    for (int row = 0; row < rows; ++row) {
      if (selection_case == 2) {
        routing.scores[row * experts] = TR(-std::numeric_limits<float>::infinity());
        routing.scores[row * experts + 1] = TR(std::numeric_limits<float>::infinity());
        routing.scores[row * experts + 2] = TR(.5f);
      } else if (selection_case == 3 || selection_case == 6 || selection_case == 7 || selection_case == 8) {
        routing.scores[row * experts + 16] = TR(1.f);
        routing.scores[row * experts + 31] = selection_case == 3
                                                 ? TR(T::FromBits(T(1.f).val + 1).ToFloat())
                                                 : TR(std::nextafter(1.f, 2.f));
        if (selection_case == 6 || selection_case == 7) {
          std::fill_n(routing.weights.begin() + row * experts, experts, TR(.5f));
        }
      } else if (selection_case == 4) {
        routing.scores[row * experts] = TR(-0.f);
      }
    }
  }
  OpTester test("MoEPacked", 1, kMSDomain);
  test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("normalize_routing_weights", static_cast<int64_t>(normalize));
  test.AddAttribute("activation_type", std::string(activation_type));
  test.AddAttribute("gate_type", int64_t{16});
  test.AddAttribute("down_type", int64_t{16});
  test.AddInput<T>("input", {1, rows, hidden_size}, input);
  test.AddInput<TR>("router_probs", {rows, experts}, routing.scores);
  test.AddInput<uint8_t>("gate", {experts, inter_size, gate_row_bytes}, gate, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddInput<uint8_t>("down", {experts, hidden_size, down_row_bytes}, down, true);
  if (selection_case != 8) test.AddInput<TR>("router_weights", {rows, experts}, routing.weights);
  test.AddOutput<T>("output", {1, rows, hidden_size}, expected);
  if (malformed || selection_case == 6) {
    // TensorCheck<BFloat16> checks expected NaNs, not actual NaNs.
    test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
      ASSERT_EQ(fetches.size(), 1u);
      const auto actual = fetches[0].Get<Tensor>().DataAsSpan<T>();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < expected.size(); ++i) {
        SCOPED_TRACE(testing::Message() << "token " << i / hidden_size << " col " << i % hidden_size);
        const float actual_value = actual[i].ToFloat();
        const float expected_value = expected[i].ToFloat();
        if (std::isnan(expected_value)) {
          EXPECT_TRUE(std::isnan(actual_value));
        } else {
          EXPECT_TRUE(std::isfinite(actual_value));
          EXPECT_EQ(actual_value, expected_value);
        }
        if (selection_case == 6 && i == 0) {
          const float fc1 = T(static_cast<float>(hidden_size / 256) * sum8[31]).ToFloat();
          const float activation = T(fc1 * fc1).ToFloat();
          const float fc2 = T(static_cast<float>(inter_size / 8) * activation * sum8[31]).ToFloat();
          const float old_value = T(.5f * fc2).ToFloat();
          const float new_value = actual[0].ToFloat();
          std::cout << "FP32_BASELINE_COUNTEREXAMPLE dtype=" << (std::is_same_v<T, BFloat16> ? "bf16" : "fp16")
                    << " raw_scores=[1,nextafter(1,2)] old_expert=31 typed_expert=16"
                    << " baseline_output=" << old_value << " typed_output=" << new_value
                    << " relative_error=" << std::abs(new_value - old_value) / std::abs(old_value) << "\n";
          EXPECT_NE(old_value, new_value);
        }
      }
    });
  }
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, RoundedScoresDemonstrateFp32BaselineCompatibilityBoundary) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestFullScoreReluSquared(1, 1, 0, 256, "relu2", false, 6);
  TestFullScoreReluSquared<MLFloat16>(1, 1, 0, 256, "relu2", false, 6);
}

TEST(MoEPackedRoutingTest, IndependentFp32RoutingPreservesNearTieSelection) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 9}) {
    TestFullScoreReluSquared<BFloat16, float>(rows, 1, 0, 256, "relu2", false, 7);
    TestFullScoreReluSquared<MLFloat16, float>(rows, 1, 0, 256, "relu2", false, 7);
  }
}

TEST(MoEPackedRoutingTest, IndependentFp32MixingPreservesGatherAndNormalization) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (bool normalize : {false, true}) {
    TestFullScoreReluSquared<BFloat16, float>(9, 2, 0, 256, "relu2", normalize);
    TestFullScoreReluSquared<MLFloat16, float>(9, 2, 0, 256, "relu2", normalize);
  }
}

TEST(MoEPackedRoutingTest, IndependentFp32LegacyLogitsPreserveNearTieSelection) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int experts : {32, 512}) {
    TestFullScoreReluSquared<BFloat16, float>(9, 1, 0, 256, "relu2", true, 8, experts);
    TestFullScoreReluSquared<MLFloat16, float>(9, 1, 0, 256, "relu2", true, 8, experts);
  }
}

TEST(MoEPackedRoutingTest, ReluSquaredSelectsInternallyFromTypedRouting) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8, 9, 17}) {
    for (int topk : {2, 32}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " k=" << topk);
      TestFullScoreReluSquared(rows, topk);
    }
  }
}

TEST(MoEPackedRoutingTest, NonfiniteSelectedMixingPoisonsOnlyItsToken) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int malformed : {1, 2, 3, 4}) {
    SCOPED_TRACE(testing::Message() << "malformed=" << malformed);
    TestFullScoreReluSquared(2, 2, malformed);
  }
}

TEST(MoEPackedRoutingTest, FullScoresMatchQMoENormalization) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (bool normalize : {false, true}) {
    TestFullScoreReluSquared(3, 9, 0, 256, "relu2", normalize);
    TestFullScoreReluSquared<MLFloat16>(3, 9, 0, 256, "relu2", normalize);
  }
  for (int weight_case : {5, 6, 7, 8}) {
    TestFullScoreReluSquared(2, 2, weight_case, 256, "relu2", true);
  }
}

TEST(MoEPackedRoutingTest, FullScoresPreserveExactOrderingAndExceptionalValues) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int selection_case : {1, 2, 3, 4, 5}) {
    const int k = selection_case == 2 ? 4 : selection_case == 3 ? 1
                                                                : 2;
    TestFullScoreReluSquared(2, k, 0, 256, "relu2", false, selection_case);
  }
}

TEST(MoEPackedRoutingTest, FullScoresHandleEmptyRowsMaximumExpertsAndChunkBoundaries) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestFullScoreReluSquared(0, 32, 0, 256);
  TestFullScoreReluSquared(2, 32, 0, 256, "relu2", true, 0, 512);
  for (int rows : {2047, 2048, 2049}) {
    TestFullScoreReluSquared(rows, 2, 0, 256, "relu2", true);
  }
}

TEST(MoEPackedRoutingTest, Iq2ProductionWidthsPreserveOperandRoundingAndReduction) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int inter_size : {512, 1536, 2048, 2304, 2560}) {
    for (int rows : {1, 8, 9}) {
      SCOPED_TRACE(testing::Message() << "inter_size=" << inter_size << " rows=" << rows);
      TestFullScoreReluSquared(rows, 9, 0, 1024, "relu2", false, 0, 32, inter_size);
    }
  }
}

TEST(MoEPackedRoutingTest, Iq2EightRowFc1TilesPreserveWidthAndRoutingBoundaries) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int hidden_size : {1024, 4096}) {
    for (int rows : {7, 8, 9}) {
      for (int topk : {9, 17}) {
        SCOPED_TRACE(testing::Message() << "hidden_size=" << hidden_size << " rows=" << rows << " k=" << topk);
        TestFullScoreReluSquared(rows, topk, 0, hidden_size, "relu2", false, 0, 32, 1536);
      }
    }
  }
}

TEST(MoEPackedRoutingTest, ReluSquaredFullScoresPreserveFp16Experts) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 9}) {
    for (int topk : {2, 32}) TestFullScoreReluSquared<MLFloat16>(rows, topk);
  }
}

TEST(MoEPackedRoutingTest, ReluSquaredAliasRemainsSupported) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestFullScoreReluSquared(1, 2, 0, 1024, "relu_squared");
}

TEST(MoEPackedRoutingTest, Iq2CodebookSweepPreservesFp32ScaleReconstruction) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int grid_offset = 0; grid_offset < 8; ++grid_offset) {
    SCOPED_TRACE(testing::Message() << "grid_offset=" << grid_offset);
    TestFullScoreReluSquared(2, 32, 0, 1024, "relu2", false, 0, 32, 256, grid_offset);
  }
}

void TestSwigluRouting(int rows, int topk = 1, bool full_scores = false, bool defaults = false) {
  constexpr int size = 32;
  const int experts = std::max(2, topk);
  const BFloat16 hidden(.125f * (.25f / (1.f + std::exp(-.25f))));
  const BFloat16 expert_output(16.f * hidden.ToFloat());
  float mixed = 0.f;
  std::vector<float> weights(rows * topk);
  std::vector<int32_t> indices(rows * topk);
  if (full_scores) {
    for (int row = 0; row < rows; ++row) {
      mixed = 0.f;
      for (int rank = 0; rank < topk; ++rank) {
        indices[row * topk + rank] = experts - 1 - rank;
        weights[row * topk + rank] = BFloat16(rank == 0 ? .3301f : .501f).ToFloat();
        mixed = std::fma(weights[row * topk + rank], expert_output.ToFloat(), mixed);
      }
    }
  }
  const BFloat16 output = full_scores ? BFloat16(mixed) : expert_output;
  const auto routing = MakeFullRoutingInputs<BFloat16>(rows, experts, topk, indices, weights);
  const auto bank = [experts](float value) {
    std::vector<uint8_t> bytes(experts * size * size * 2);
    const BFloat16 element(value);
    for (size_t index = 0; index < bytes.size(); index += 2) {
      bytes[index] = static_cast<uint8_t>(element.val);
      bytes[index + 1] = static_cast<uint8_t>(element.val >> 8);
    }
    return bytes;
  };
  OpTester test("MoEPacked", 1, kMSDomain);
  if (!defaults) test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("gate_type", int64_t{30});
  test.AddAttribute("up_type", int64_t{30});
  test.AddAttribute("down_type", int64_t{30});
  test.AddInput<BFloat16>("input", {rows, size}, std::vector<BFloat16>(rows * size, BFloat16(.03125f)));
  if (full_scores) {
    test.AddAttribute("normalize_routing_weights", int64_t{0});
    test.AddInput<BFloat16>("router_probs", {rows, experts}, routing.scores);
  } else {
    if (!defaults) test.AddAttribute("normalize_routing_weights", int64_t{1});
    test.AddInput<BFloat16>("router_probs", {rows, experts}, std::vector<BFloat16>(rows * experts, BFloat16(0.f)));
  }
  test.AddInput<uint8_t>("gate", {experts, size, size * 2}, bank(.25f), true);
  test.AddInput<uint8_t>("up", {experts, size, size * 2}, bank(.125f), true);
  test.AddInput<uint8_t>("down", {experts, size, size * 2}, bank(.5f), true);
  if (full_scores) {
    test.AddInput<BFloat16>("router_weights", {rows, experts}, routing.weights);
  }
  test.AddOutput<BFloat16>("output", {rows, size}, std::vector<BFloat16>(rows * size, output));
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, ExplicitlyNormalizedLegacySwigluRemainsUnchanged) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {4, 48, 129}) TestSwigluRouting(rows);
}

TEST(MoEPackedRoutingTest, FullScoresExplicitlySkipRenormalizingMixingValues) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSwigluRouting(3, 1, true, true);
}

TEST(MoEPackedRoutingTest, LegacyDefaultStillRenormalizesSoftmaxProbabilities) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestSwigluRouting(1, 1, false, true);
}

TEST(MoEPackedRoutingTest, SwigluAcceptsFullScoresAndMixingValues) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int topk : {2, 32}) TestSwigluRouting(5, topk, true);
}

TEST(MoEPackedRoutingTest, ReluIq2PreservesContiguousKCancellation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int hidden = 1024;
  constexpr int intermediate = 256;
  std::vector<uint8_t> gate(intermediate * 264, 0), down(hidden * 66, 0);
  for (size_t block = 0; block < gate.size() / 66; ++block) gate[block * 66 + 1] = 0x3c;
  for (size_t block = 0; block < down.size() / 66; ++block) down[block * 66 + 1] = 0x3c;
  for (int rows : {1, 8, 17, 129}) {
    SCOPED_TRACE(testing::Message() << "rows=" << rows);
    std::vector<BFloat16> input(rows * hidden, BFloat16(0.f));
    for (int row = 0; row < rows; ++row) {
      input[row * hidden] = BFloat16(0x1p25f);
      input[row * hidden + 64] = BFloat16(-0x1p25f);
      input[row * hidden + 128] = BFloat16(1.f);
      input[row * hidden + 160] = BFloat16(1.f);
    }
    // Separate small terms from large MMA partials on both Ampere and Hopper.
    // FC1 retains both unit terms, then BF16 ReLU-square and FC2 produce 1024.
    OpTester test("MoEPacked", 1, kMSDomain);
    test.AddAttribute("k", int64_t{1});
    test.AddAttribute("activation_type", std::string("relu_squared"));
    test.AddAttribute("gate_type", int64_t{16});
    test.AddAttribute("down_type", int64_t{16});
    test.AddAttribute("normalize_routing_weights", int64_t{0});
    test.AddInput<BFloat16>("input", {rows, hidden}, input);
    test.AddInput<BFloat16>("router_probs", {rows, 1}, std::vector<BFloat16>(rows, BFloat16(0.f)));
    test.AddInput<uint8_t>("gate", {1, intermediate, 264}, gate, true);
    test.AddOptionalInputEdge<uint8_t>();
    test.AddInput<uint8_t>("down", {1, hidden, 66}, down, true);
    test.AddInput<BFloat16>("router_weights", {rows, 1}, std::vector<BFloat16>(rows, BFloat16(1.f)));
    test.AddOutput<BFloat16>("output", {rows, hidden},
                             std::vector<BFloat16>(rows * hidden, BFloat16(1024.f)));
    RunMoeCuda(test);
  }
}

void TestRandomIq2Relu2(int rows, int hidden, int intermediate, int topk) {
  const int experts = topk;
  const int gate_row_bytes = hidden / 256 * 66;
  const int down_row_bytes = intermediate / 256 * 66;
  std::mt19937 random(33963);
  std::vector<uint8_t> gate(experts * intermediate * gate_row_bytes);
  std::vector<uint8_t> down(experts * hidden * down_row_bytes);
  const auto fill_bank = [&](std::vector<uint8_t>& bank) {
    for (auto& byte : bank) byte = static_cast<uint8_t>(random());
    for (size_t block = 0; block < bank.size(); block += 66) {
      const MLFloat16 scale(static_cast<float>(256 + random() % 1024) / 262144.f);
      bank[block] = static_cast<uint8_t>(scale.val);
      bank[block + 1] = static_cast<uint8_t>(scale.val >> 8);
    }
  };
  fill_bank(gate);
  fill_bank(down);
  std::vector<BFloat16> input(rows * hidden), expected(rows * hidden);
  for (auto& value : input) value = BFloat16(static_cast<float>(static_cast<int>(random() % 129) - 64) / 64.f);
  std::vector<int32_t> indices(rows * topk);
  std::vector<float> weights(rows * topk);
  std::vector<float> gate_values(experts * intermediate * hidden);
  std::vector<float> down_values(experts * hidden * intermediate);
  // Use the CPU decoder, not a reconstruction of the CUDA fragment mapping.
  for (int row = 0; row < experts * intermediate; ++row) {
    ASSERT_TRUE(contrib::gguf_quant::DequantizeGgufRowCpu(
        gate.data() + row * gate_row_bytes, 16, hidden, gate_values.data() + row * hidden));
  }
  for (int row = 0; row < experts * hidden; ++row) {
    ASSERT_TRUE(contrib::gguf_quant::DequantizeGgufRowCpu(
        down.data() + row * down_row_bytes, 16, intermediate, down_values.data() + row * intermediate));
  }
  for (auto& value : gate_values) value = BFloat16(value).ToFloat();
  for (auto& value : down_values) value = BFloat16(value).ToFloat();
  std::vector<float> activation(intermediate), mixed(hidden);
  for (int row = 0; row < rows; ++row) {
    std::fill(mixed.begin(), mixed.end(), 0.f);
    for (int rank = 0; rank < topk; ++rank) {
      const int expert = (row & 1) ? experts - 1 - rank : rank;
      const float weight = BFloat16(static_cast<float>(17 + random() % 97) / 113.f).ToFloat();
      indices[row * topk + rank] = expert;
      weights[row * topk + rank] = weight;
      for (int channel = 0; channel < intermediate; ++channel) {
        const auto* w = gate_values.data() + (expert * intermediate + channel) * hidden;
        double sum = 0;
        for (int k = 0; k < hidden; ++k) sum += static_cast<double>(input[row * hidden + k].ToFloat()) * w[k];
        const float fc1 = BFloat16(static_cast<float>(sum)).ToFloat();
        const float positive = std::max(fc1, 0.f);
        activation[channel] = BFloat16(positive * positive).ToFloat();
      }
      for (int channel = 0; channel < hidden; ++channel) {
        const auto* w = down_values.data() + (expert * hidden + channel) * intermediate;
        double sum = 0;
        for (int k = 0; k < intermediate; ++k) sum += static_cast<double>(activation[k]) * w[k];
        const float fc2 = BFloat16(static_cast<float>(sum)).ToFloat();
        mixed[channel] = std::fma(weight, fc2, mixed[channel]);
      }
    }
    for (int channel = 0; channel < hidden; ++channel) expected[row * hidden + channel] = BFloat16(mixed[channel]);
  }
  OpTester test("MoEPacked", 1, kMSDomain);
  test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("activation_type", std::string("relu2"));
  test.AddAttribute("normalize_routing_weights", int64_t{0});
  test.AddAttribute("gate_type", int64_t{16});
  test.AddAttribute("down_type", int64_t{16});
  test.AddInput<BFloat16>("input", {rows, hidden}, input);
  const auto routing = MakeFullRoutingInputs<BFloat16>(rows, experts, topk, indices, weights);
  test.AddInput<BFloat16>("router_probs", {rows, experts}, routing.scores);
  test.AddInput<uint8_t>("gate", {experts, intermediate, gate_row_bytes}, gate, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddInput<uint8_t>("down", {experts, hidden, down_row_bytes}, down, true);
  test.AddInput<BFloat16>("router_weights", {rows, experts}, routing.weights);
  test.AddOutput<BFloat16>("output", {rows, hidden}, expected);
  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), 1u);
    const auto actual = fetches[0].Get<Tensor>().DataAsSpan<BFloat16>();
    ASSERT_EQ(actual.size(), expected.size());
    const auto ordered_bits = [](BFloat16 value) {
      return (value.val & 0x8000) ? 0x8000 - (value.val & 0x7fff) : 0x8000 + value.val;
    };
    size_t mismatches = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_TRUE(std::isfinite(actual[i].ToFloat())) << "element " << i;
      const int ulps = std::abs(ordered_bits(actual[i]) - ordered_bits(expected[i]));
      ASSERT_LE(ulps, 2) << "element " << i;
      mismatches += ulps != 0;
    }
    // FP32 MMA reduction may flip isolated BF16 rounding boundaries, not whole channels.
    EXPECT_LE(mismatches, std::max<size_t>(1, actual.size() / 1000));
  });
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, RandomIq2Relu2MatchesIndependentCpuReference) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int hidden : {256, 1024}) {
    const int intermediate = hidden == 256 ? 512 : 256;
    for (int rows : {1, 9, 65}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " hidden=" << hidden << " inter=" << intermediate);
      TestRandomIq2Relu2(rows, hidden, intermediate, 2);
    }
  }
  TestRandomIq2Relu2(5, 256, 512, 32);
}

TEST(MoEPackedRoutingTest, RejectsInvalidRoutingContracts) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  enum class Invalid {
    MissingScores,
    ScoresType,
    MixingType,
    WeightShape,
    ProbsShape,
    ReluUpWeights,
    ReluGateFormat,
    ReluDownFormat,
    SoftmaxK,
    ExcessiveK,
    SwigluMissingUpType,
    IntegratedRoutingFunction,
    RemovedRouterScale,
    LegacyCompactInputs,
    NonfiniteRoutedScalingFactor,
    NegativeRoutingEpsilon,
    NonfiniteRoutingEpsilon,
    LegacyRoutedScalingFactor,
    LegacyRoutingEpsilon,
  };
  const std::pair<Invalid, const char*> cases[] = {
      {Invalid::MissingScores, ""},
      {Invalid::ScoresType, "bound to different types"},
      {Invalid::MixingType, "bound to different types"},
      {Invalid::WeightShape, "router_weights must be [num_rows, num_experts]"},
      {Invalid::ProbsShape, "router_probs must be [num_rows, num_experts]"},
      {Invalid::ReluUpWeights, "relu2 requires omitted up weights"},
      {Invalid::ReluGateFormat, "relu2 currently requires IQ2_XXS gate and down weights"},
      {Invalid::ReluDownFormat, "relu2 currently requires IQ2_XXS gate and down weights"},
      {Invalid::SoftmaxK, "softmax routing requires k <= 16"},
      {Invalid::ExcessiveK, "requires k in 1..32"},
      {Invalid::SwigluMissingUpType, "swiglu requires up_type"},
      {Invalid::IntegratedRoutingFunction, "Unrecognized attribute: routing_function"},
      {Invalid::RemovedRouterScale, "Unrecognized attribute: router_scale"},
      {Invalid::LegacyCompactInputs, "input size"},
      {Invalid::NonfiniteRoutedScalingFactor, "routed_scaling_factor must be finite"},
      {Invalid::NegativeRoutingEpsilon, "routing_epsilon must be finite and >= 0"},
      {Invalid::NonfiniteRoutingEpsilon, "routing_epsilon must be finite and >= 0"},
      {Invalid::LegacyRoutedScalingFactor, "routed_scaling_factor and routing_epsilon require router_weights"},
      {Invalid::LegacyRoutingEpsilon, "routed_scaling_factor and routing_epsilon require router_weights"},
  };
  for (const auto& [invalid, message] : cases) {
    SCOPED_TRACE(testing::Message() << "invalid=" << static_cast<int>(invalid));
    constexpr int experts = 32;
    constexpr int size = 256;
    bool scores = true, mixing = true, up = false;
    int topk = 2;
    int weights_width = experts, probs_width = experts;
    int gate_type = 16, down_type = 16;
    std::string activation = "relu2";
    switch (invalid) {
      case Invalid::MissingScores:
        scores = false;
        break;
      case Invalid::WeightShape:
        weights_width = 1;
        break;
      case Invalid::ProbsShape:
        probs_width = experts - 1;
        break;
      case Invalid::ReluUpWeights:
        up = true;
        break;
      case Invalid::ReluGateFormat:
        gate_type = 30;
        break;
      case Invalid::ReluDownFormat:
        down_type = 30;
        break;
      case Invalid::SoftmaxK:
        mixing = false;
        topk = 17;
        break;
      case Invalid::ExcessiveK:
        topk = 33;
        break;
      case Invalid::SwigluMissingUpType:
        activation = "swiglu";
        up = true;
        break;
      case Invalid::LegacyRoutedScalingFactor:
      case Invalid::LegacyRoutingEpsilon:
        mixing = false;
        break;
      case Invalid::IntegratedRoutingFunction:
      case Invalid::ScoresType:
      case Invalid::MixingType:
      case Invalid::RemovedRouterScale:
      case Invalid::LegacyCompactInputs:
      case Invalid::NonfiniteRoutedScalingFactor:
      case Invalid::NegativeRoutingEpsilon:
      case Invalid::NonfiniteRoutingEpsilon:
        break;
    }
    OpTester test("MoEPacked", 1, kMSDomain);
    test.AddAttribute("k", static_cast<int64_t>(topk));
    test.AddAttribute("activation_type", activation);
    test.AddAttribute("normalize_routing_weights", int64_t{1});
    if (invalid == Invalid::IntegratedRoutingFunction) {
      test.AddAttribute("routing_function", std::string("sigmoid"));
    }
    if (invalid == Invalid::RemovedRouterScale) test.AddAttribute("router_scale", 2.5f);
    if (invalid == Invalid::NonfiniteRoutedScalingFactor) {
      test.AddAttribute("routed_scaling_factor", std::numeric_limits<float>::infinity());
    }
    if (invalid == Invalid::NegativeRoutingEpsilon) {
      test.AddAttribute("routing_epsilon", -1e-6f);
    }
    if (invalid == Invalid::NonfiniteRoutingEpsilon) {
      test.AddAttribute("routing_epsilon", std::numeric_limits<float>::quiet_NaN());
    }
    if (invalid == Invalid::LegacyRoutedScalingFactor) test.AddAttribute("routed_scaling_factor", 2.5f);
    if (invalid == Invalid::LegacyRoutingEpsilon) test.AddAttribute("routing_epsilon", 0.25f);
    test.AddAttribute("gate_type", static_cast<int64_t>(gate_type));
    test.AddAttribute("down_type", static_cast<int64_t>(down_type));
    if (up && invalid != Invalid::SwigluMissingUpType) test.AddAttribute("up_type", int64_t{16});
    test.AddInput<BFloat16>("input", {1, size}, std::vector<BFloat16>(size));
    if (!scores) {
      test.AddOptionalInputEdge<float>();
    } else if (invalid == Invalid::ScoresType) {
      test.AddInput<float>("probs", {1, probs_width}, std::vector<float>(probs_width));
    } else {
      test.AddInput<BFloat16>("probs", {1, probs_width}, std::vector<BFloat16>(probs_width));
    }
    const std::vector<uint8_t> bank(experts * size * 66);
    test.AddInput<uint8_t>("gate", {experts, size, 66}, bank, true);
    if (up) {
      test.AddInput<uint8_t>("up", {experts, size, 66}, bank, true);
    } else {
      test.AddOptionalInputEdge<uint8_t>();
    }
    test.AddInput<uint8_t>("down", {experts, size, 66}, bank, true);
    if (invalid == Invalid::LegacyCompactInputs) {
      test.AddInput<int32_t>("indices", {1, topk}, std::vector<int32_t>(topk));
      test.AddInput<float>("weights", {1, topk}, std::vector<float>(topk, 1.f));
    } else if (mixing) {
      if (invalid == Invalid::MixingType) {
        test.AddInput<float>("weights", {1, weights_width}, std::vector<float>(weights_width, 1.f));
      } else {
        test.AddInput<BFloat16>("weights", {1, weights_width}, std::vector<BFloat16>(weights_width, BFloat16(1.f)));
      }
    }
    test.AddOutput<BFloat16>("output", {1, size}, std::vector<BFloat16>(size));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectFailure, message, {}, nullptr, &providers);
  }
}

float DecodeNonuniformMoeIq2Operand(const uint8_t* row, int index) {
  const auto* block = row + (index / 256) * 66;
  const auto* record = block + 2 + ((index % 256) / 32) * 8;
  const float d = MLFloat16::FromBits(
                      static_cast<uint16_t>(block[0] | (static_cast<uint16_t>(block[1]) << 8)))
                      .ToFloat();
  uint32_t signs_scale = 0;
  for (int byte = 0; byte < 4; ++byte) signs_scale |= static_cast<uint32_t>(record[4 + byte]) << (8 * byte);
  const int group = (index % 32) / 8, position = index % 8;
  const uint32_t seven = (signs_scale >> (7 * group)) & 127u;
  int sign = position == 7 ? 0 : (seven >> position) & 1u;
  if (position == 7) {
    for (int bit = 0; bit < 7; ++bit) sign ^= (seven >> bit) & 1u;
  }
  const uint64_t grid = contrib::gguf_quant::iq2xxs_grid[record[group]];
  const float magnitude = static_cast<float>((grid >> (8 * position)) & 255u);
  return BFloat16(d * (.5f + static_cast<float>(signs_scale >> 28)) * .25f *
                  magnitude * (sign ? -1.f : 1.f))
      .ToFloat();
}

std::vector<uint8_t> MakeNonuniformMoeIq2Bank(int outputs, int k, int active_channel, uint32_t seed) {
  constexpr int experts = 2;
  const int blocks = k / 256;
  std::mt19937 rng(seed);
  std::vector<uint8_t> bank(experts * outputs * blocks * 66);
  for (int expert = 0; expert < experts; ++expert) {
    for (int col = 0; col < outputs; ++col) {
      for (int block = 0; block < blocks; ++block) {
        auto* bytes = bank.data() + ((expert * outputs + col) * blocks + block) * 66;
        // Codes, seven-bit sign fields and four-bit scales vary independently
        // for every output/record/block/expert, including sign parity at lane 7.
        for (int byte = 2; byte < 66; ++byte) bytes[byte] = static_cast<uint8_t>(rng());
        const float d = std::ldexp(rng() & 1u ? -1.f : 1.f, -static_cast<int>(3 + rng() % 3));
        const MLFloat16 scale(active_channel < 0 || col == active_channel ? d : 0.f);
        bytes[0] = static_cast<uint8_t>(scale.val);
        bytes[1] = static_cast<uint8_t>(scale.val >> 8);
      }
    }
  }
  return bank;
}

TEST(MoEPackedRoutingTest, Iq2NonuniformSingleProductOracleAndEightNineRowEquivalence) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int experts = 2, hidden = 1024, intermediate = 512;
  constexpr int gate_row_bytes = (hidden / 256) * 66, down_row_bytes = (intermediate / 256) * 66;
  // Cover every pair in both grid halves, all eight records, and both blocks.
  for (int channel : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                      32, 65, 98, 131, 164, 197, 230, 255, 256, 511}) {
    SCOPED_TRACE(testing::Message() << "channel=" << channel);
    const auto gate = MakeNonuniformMoeIq2Bank(intermediate, hidden, channel, 34002 + channel);
    const auto down = MakeNonuniformMoeIq2Bank(hidden, intermediate, -1, 134002 + channel);
    std::vector<BFloat16> decode_output;
    for (int rows : {8, 9}) {
      std::vector<BFloat16> input(rows * hidden, BFloat16(0.f)), expected(rows * hidden);
      std::vector<int32_t> indices(rows * experts);
      std::vector<float> weights(rows * experts);
      for (int row = 0; row < rows; ++row) {
        const int position = (channel * 37 + row * 53) % hidden;
        const float x = static_cast<float>(row % 3 + 1);
        input[row * hidden + position] = BFloat16(x);
        float activation[experts];
        for (int expert = 0; expert < experts; ++expert) {
          indices[row * experts + expert] = expert;
          weights[row * experts + expert] = expert ? .5f : .25f;
          const auto* gate_row = gate.data() + (expert * intermediate + channel) * gate_row_bytes;
          const float fc1 = BFloat16(x * DecodeNonuniformMoeIq2Operand(gate_row, position)).ToFloat();
          const float relu = std::max(fc1, 0.f);
          activation[expert] = BFloat16(relu * relu).ToFloat();
        }
        for (int col = 0; col < hidden; ++col) {
          float mixed = 0.f;
          for (int expert = 0; expert < experts; ++expert) {
            const auto* down_row = down.data() + (expert * hidden + col) * down_row_bytes;
            // Only one FC1 channel survives. Both projections therefore have
            // one exact product, independently of the MMA accumulation order.
            const float fc2 =
                BFloat16(activation[expert] * DecodeNonuniformMoeIq2Operand(down_row, channel)).ToFloat();
            mixed = std::fma(weights[row * experts + expert], fc2, mixed);
          }
          expected[row * hidden + col] = BFloat16(mixed);
        }
      }
      ASSERT_TRUE(std::any_of(expected.begin(), expected.end(), [](BFloat16 value) { return value.ToFloat() != 0.f; }));
      OpTester test("MoEPacked", 1, kMSDomain);
      test.AddAttribute("k", int64_t{experts});
      test.AddAttribute("normalize_routing_weights", int64_t{0});
      test.AddAttribute("activation_type", std::string("relu_squared"));
      test.AddAttribute("gate_type", int64_t{16});
      test.AddAttribute("down_type", int64_t{16});
      test.AddInput<BFloat16>("input", {rows, hidden}, input);
      const auto routing = MakeFullRoutingInputs<BFloat16>(rows, experts, experts, indices, weights);
      test.AddInput<BFloat16>("router_probs", {rows, experts}, routing.scores);
      test.AddInput<uint8_t>("gate", {experts, intermediate, gate_row_bytes}, gate, true);
      test.AddOptionalInputEdge<uint8_t>();
      test.AddInput<uint8_t>("down", {experts, hidden, down_row_bytes}, down, true);
      test.AddInput<BFloat16>("router_weights", {rows, experts}, routing.weights);
      test.AddOutput<BFloat16>("output", {rows, hidden}, expected);
      RunMoeCuda(test);
      ASSERT_EQ(test.GetFetches().size(), 1u);
      const auto* output = test.GetFetches()[0].Get<Tensor>().Data<BFloat16>();
      if (rows == 8) {
        decode_output.assign(output, output + rows * hidden);
      } else {
        for (size_t index = 0; index < decode_output.size(); ++index) {
          ASSERT_EQ(output[index].val, decode_output[index].val) << "index=" << index;
        }
      }
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
