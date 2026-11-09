// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
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

void RunMoeCuda(OpTester& test) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.SetOutputTolerance(0, 0);
  test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

template <typename T = BFloat16>
void TestPreselectedReluSquared(int rows, int topk, int malformed = 0, bool sigmoid = false,
                                const char* activation_type = "relu2") {
  constexpr int experts = 32;
  constexpr int hidden_size = 1024;
  constexpr int inter_size = 256;
  std::vector<uint8_t> gate(experts * inter_size * 264);
  std::vector<uint8_t> down(experts * hidden_size * 66);
  std::vector<T> input(rows * hidden_size), expected(rows * hidden_size);
  std::vector<int32_t> indices(rows * topk);
  std::vector<float> weights(rows * topk);
  std::vector<float> sum8(experts);
  const float sigmoid_scale = topk == 2 ? 5.03f : 5.01f;
  const auto probability = [](int expert) {
    return 1.f / (1.f + std::exp(expert & 1 ? 1.f : -1.f));
  };
  float sigmoid_sum = 0.f;
  if (sigmoid) {
    for (int rank = 0; rank < topk; ++rank) sigmoid_sum += probability(experts - 1 - rank);
  }
  for (int expert = 0; expert < experts; ++expert) {
    const MLFloat16 d(expert & 1 ? .023f : .019f);
    sum8[expert] = T(d.ToFloat() * .125f * 43.f).ToFloat() +
                   7.f * T(d.ToFloat()).ToFloat();
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
            for (int group = 0; group < 4; ++group) bytes[2 + record * 8 + group] = 1;
            if (stage == 0 && expert % 5 == 4) {
              for (int byte = 0; byte < 4; ++byte) {
                bytes[2 + record * 8 + 4 + byte] = static_cast<uint8_t>(0x0fffffffu >> (8 * byte));
              }
            }
          }
        }
      }
    }
  }
  bool distinguishes_narrowed_weights = false;
  for (int row = 0; row < rows; ++row) {
    const float factor = static_cast<float>(1 + (row & 1));
    for (int col = 0; col < hidden_size; ++col) input[row * hidden_size + col] = T(factor / 32.f);
    float mixed = 0.f;
    float narrowed_mixed = 0.f;
    for (int rank = 0; rank < topk; ++rank) {
      const int expert = sigmoid || (row & 1) ? (experts - 1 - rank) : rank;
      indices[row * topk + rank] = expert;
      const float weight = sigmoid ? (probability(expert) / (sigmoid_sum + 1e-20f) * sigmoid_scale)
                                   : (rank == 0 ? .3301f : .501f);
      weights[row * topk + rank] = weight;
      const float fc1 = expert % 5 == 4 ? 0.f : T(4.f * factor * sum8[expert]).ToFloat();
      const float activation = T(fc1 * fc1).ToFloat();
      const float fc2 = T(32.f * activation * sum8[expert]).ToFloat();
      mixed = std::fma(weight, fc2, mixed);
      narrowed_mixed = std::fma(T(weight).ToFloat(), fc2, narrowed_mixed);
    }
    distinguishes_narrowed_weights |= T(mixed).val != T(narrowed_mixed).val;
    for (int col = 0; col < hidden_size; ++col) expected[row * hidden_size + col] = T(mixed);
  }
  if constexpr (std::is_same_v<T, BFloat16>) {
    ASSERT_TRUE(distinguishes_narrowed_weights) << "Fixture must detect BF16-rounded routing weights";
  }
  if (malformed) {
    if (malformed == 1) indices[0] = -1;
    if (malformed == 2) indices[1] = indices[0];
    if (malformed == 3) weights[0] = std::numeric_limits<float>::infinity();
    for (int col = 0; col < hidden_size; ++col) {
      expected[col] = T(std::numeric_limits<float>::quiet_NaN());
    }
  }
  OpTester test("MoEPacked", 1, kMSDomain);
  test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("normalize_routing_weights", int64_t{sigmoid ? 1 : 0});
  test.AddAttribute("activation_type", std::string(activation_type));
  test.AddAttribute("gate_type", int64_t{16});
  test.AddAttribute("down_type", int64_t{16});
  test.AddInput<T>("input", {1, rows, hidden_size}, input);
  test.AddOptionalInputEdge<T>();
  test.AddInput<uint8_t>("gate", {experts, inter_size, 264}, gate, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddInput<uint8_t>("down", {experts, hidden_size, 66}, down, true);
  if (sigmoid) {
    constexpr int router_width = 256;
    test.AddAttribute("routing_function", std::string("sigmoid"));
    test.AddAttribute("router_scale", sigmoid_scale);
    test.AddOptionalInputEdge<int32_t>();
    test.AddOptionalInputEdge<float>();
    std::vector<T> router_input(rows * router_width, T(0.f));
    std::vector<T> router_weight(experts * router_width, T(0.f));
    for (int row = 0; row < rows; ++row) router_input[row * router_width] = T(1.f);
    for (int expert = 0; expert < experts; ++expert) {
      router_weight[expert * router_width] = T(expert & 1 ? -1.f : 1.f);
    }
    test.AddInput<T>("router_input", {rows, router_width}, router_input);
    test.AddInput<T>("router_weight", {experts, router_width}, router_weight, true);
    std::vector<float> bias(experts);
    for (int expert = 0; expert < experts; ++expert) bias[expert] = static_cast<float>(expert) * 2.f;
    test.AddInput<float>("router_bias", {experts}, bias, true);
  } else {
    test.AddInput<int32_t>("expert_indices", {rows, topk}, indices);
    test.AddInput<float>("expert_weights", {rows, topk}, weights);
  }
  test.AddOutput<T>("output", {1, rows, hidden_size}, expected);
  if (malformed) {
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
      }
    });
  }
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, ReluSquaredPreservesPreselectedFp32Routing) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8, 9, 17}) {
    for (int topk : {2, 32}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " k=" << topk);
      TestPreselectedReluSquared(rows, topk);
    }
  }
}

TEST(MoEPackedRoutingTest, MalformedRoutesPoisonOnlyTheirToken) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int malformed : {1, 2, 3}) {
    SCOPED_TRACE(testing::Message() << "malformed=" << malformed);
    TestPreselectedReluSquared(2, 2, malformed);
  }
}

TEST(MoEPackedRoutingTest, ReluSquaredPreservesPreselectedFp16Routing) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 9}) {
    for (int topk : {2, 32}) TestPreselectedReluSquared<MLFloat16>(rows, topk);
  }
}

TEST(MoEPackedRoutingTest, ReluSquaredAliasRemainsSupported) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  TestPreselectedReluSquared(1, 2, 0, false, "relu_squared");
}

TEST(MoEPackedRoutingTest, SigmoidCorrectionBiasUsesUnbiasedWeights) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 8, 9, 17}) {
    for (int topk : {2, 32}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " k=" << topk);
      TestPreselectedReluSquared(rows, topk, 0, true);
      TestPreselectedReluSquared<MLFloat16>(rows, topk, 0, true);
    }
  }
}

void TestSwigluRouting(int rows, int topk = 1, bool preselected = false) {
  constexpr int size = 32;
  const int experts = std::max(2, topk);
  const BFloat16 hidden(.125f * (.25f / (1.f + std::exp(-.25f))));
  const BFloat16 expert_output(16.f * hidden.ToFloat());
  float mixed = 0.f;
  std::vector<float> weights(rows * topk);
  std::vector<int32_t> indices(rows * topk);
  if (preselected) {
    for (int row = 0; row < rows; ++row) {
      mixed = 0.f;
      for (int rank = 0; rank < topk; ++rank) {
        indices[row * topk + rank] = experts - 1 - rank;
        weights[row * topk + rank] = rank == 0 ? .3301f : .501f;
        mixed = std::fma(weights[row * topk + rank], expert_output.ToFloat(), mixed);
      }
    }
  }
  const BFloat16 output = preselected ? BFloat16(mixed) : expert_output;
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
  test.AddAttribute("k", static_cast<int64_t>(topk));
  test.AddAttribute("gate_type", int64_t{30});
  test.AddAttribute("up_type", int64_t{30});
  test.AddAttribute("down_type", int64_t{30});
  test.AddInput<BFloat16>("input", {rows, size}, std::vector<BFloat16>(rows * size, BFloat16(.03125f)));
  if (preselected) {
    test.AddAttribute("normalize_routing_weights", int64_t{0});
    test.AddOptionalInputEdge<BFloat16>();
  } else {
    test.AddInput<BFloat16>("router_probs", {rows, experts}, std::vector<BFloat16>(rows * experts, BFloat16(0.f)));
  }
  test.AddInput<uint8_t>("gate", {experts, size, size * 2}, bank(.25f), true);
  test.AddInput<uint8_t>("up", {experts, size, size * 2}, bank(.125f), true);
  test.AddInput<uint8_t>("down", {experts, size, size * 2}, bank(.5f), true);
  if (preselected) {
    test.AddInput<int32_t>("indices", {rows, topk}, indices);
    test.AddInput<float>("weights", {rows, topk}, weights);
  }
  test.AddOutput<BFloat16>("output", {rows, size}, std::vector<BFloat16>(rows * size, output));
  RunMoeCuda(test);
}

TEST(MoEPackedRoutingTest, ExistingSwigluRoutingDefaultsRemainUnchanged) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {4, 48, 129}) TestSwigluRouting(rows);
}

TEST(MoEPackedRoutingTest, SwigluAcceptsPreselectedRoutes) {
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
    test.AddOptionalInputEdge<BFloat16>();
    test.AddInput<uint8_t>("gate", {1, intermediate, 264}, gate, true);
    test.AddOptionalInputEdge<uint8_t>();
    test.AddInput<uint8_t>("down", {1, hidden, 66}, down, true);
    test.AddInput<int32_t>("indices", {rows, 1}, std::vector<int32_t>(rows, 0));
    test.AddInput<float>("weights", {rows, 1}, std::vector<float>(rows, 1.f));
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
      const float weight = static_cast<float>(17 + random() % 97) / 113.f;
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
  test.AddOptionalInputEdge<BFloat16>();
  test.AddInput<uint8_t>("gate", {experts, intermediate, gate_row_bytes}, gate, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddInput<uint8_t>("down", {experts, hidden, down_row_bytes}, down, true);
  test.AddInput<int32_t>("indices", {rows, topk}, indices);
  test.AddInput<float>("weights", {rows, topk}, weights);
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
    ConflictingProbs,
    NormalizedPreselected,
    MissingWeights,
    MissingIndices,
    IndexShape,
    WeightShape,
    ProbsShape,
    SigmoidSwiglu,
    SigmoidNormalization,
    SigmoidProbs,
    SigmoidPreselected,
    RouterWidth,
    EmptyRouterWidth,
    RouterWeightShape,
    RouterBiasShape,
    ReluUpWeights,
    ReluGateFormat,
    ReluDownFormat,
    SoftmaxK,
    SwigluMissingUpType,
    SoftmaxScale,
    SigmoidNonfiniteScale,
  };
  const std::pair<Invalid, const char*> cases[] = {
      {Invalid::ConflictingProbs, "preselected routing requires both"},
      {Invalid::NormalizedPreselected, "normalize_routing_weights=0"},
      {Invalid::MissingWeights, "preselected routing requires both"},
      {Invalid::MissingIndices, "preselected routing requires both"},
      {Invalid::IndexShape, "indices and weights [num_rows, k]"},
      {Invalid::WeightShape, "indices and weights [num_rows, k]"},
      {Invalid::ProbsShape, "router_probs must be [num_rows, num_experts]"},
      {Invalid::SigmoidSwiglu, "sigmoid routing requires relu2"},
      {Invalid::SigmoidNormalization, "sigmoid routing requires relu2 and normalize_routing_weights=1"},
      {Invalid::SigmoidProbs, "with no other routing inputs"},
      {Invalid::SigmoidPreselected, "with no other routing inputs"},
      {Invalid::RouterWidth, "width divisible by 256"},
      {Invalid::EmptyRouterWidth, "width divisible by 256"},
      {Invalid::RouterWeightShape, "router_weight[num_experts,width]"},
      {Invalid::RouterBiasShape, "router_bias[num_experts]"},
      {Invalid::ReluUpWeights, "relu2 requires omitted up weights"},
      {Invalid::ReluGateFormat, "relu2 currently requires IQ2_XXS gate and down weights"},
      {Invalid::ReluDownFormat, "relu2 currently requires IQ2_XXS gate and down weights"},
      {Invalid::SoftmaxK, "internal routing requires router_probs and k <= 16"},
      {Invalid::SwigluMissingUpType, "swiglu requires up_type"},
      {Invalid::SoftmaxScale, "non-default scaling requires sigmoid"},
      {Invalid::SigmoidNonfiniteScale, "router_scale must be finite"},
  };
  for (const auto& [invalid, message] : cases) {
    SCOPED_TRACE(testing::Message() << "invalid=" << static_cast<int>(invalid));
    constexpr int experts = 32;
    constexpr int size = 256;
    bool sigmoid = false, probs = false, indices = true, weights = true, up = false;
    int topk = 2, normalize = 0, width = size, weight_width = size, bias_count = experts;
    int index_width = topk, weights_width = topk, probs_width = experts;
    int gate_type = 16, down_type = 16;
    float scale = 1.f;
    std::string activation = "relu2";
    switch (invalid) {
      case Invalid::SigmoidSwiglu:
      case Invalid::SigmoidNormalization:
      case Invalid::SigmoidProbs:
      case Invalid::SigmoidPreselected:
      case Invalid::RouterWidth:
      case Invalid::EmptyRouterWidth:
      case Invalid::RouterWeightShape:
      case Invalid::RouterBiasShape:
      case Invalid::SigmoidNonfiniteScale:
        sigmoid = true;
        normalize = 1;
        indices = weights = false;
        break;
      default:
        break;
    }
    switch (invalid) {
      case Invalid::ConflictingProbs:
        probs = true;
        break;
      case Invalid::NormalizedPreselected:
        normalize = 1;
        break;
      case Invalid::MissingWeights:
        weights = false;
        break;
      case Invalid::MissingIndices:
        indices = false;
        break;
      case Invalid::IndexShape:
        index_width = 1;
        break;
      case Invalid::WeightShape:
        weights_width = 1;
        break;
      case Invalid::ProbsShape:
        indices = weights = false;
        probs = true;
        probs_width = experts - 1;
        break;
      case Invalid::SigmoidSwiglu:
        activation = "swiglu";
        up = true;
        break;
      case Invalid::SigmoidNormalization:
        normalize = 0;
        break;
      case Invalid::SigmoidProbs:
        probs = true;
        break;
      case Invalid::SigmoidPreselected:
        indices = weights = true;
        break;
      case Invalid::RouterWidth:
        width = size - 1;
        break;
      case Invalid::EmptyRouterWidth:
        width = 0;
        break;
      case Invalid::RouterWeightShape:
        weight_width = size - 1;
        break;
      case Invalid::RouterBiasShape:
        bias_count = experts - 1;
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
        indices = weights = false;
        probs = true;
        topk = 17;
        break;
      case Invalid::SwigluMissingUpType:
        activation = "swiglu";
        up = true;
        break;
      case Invalid::SoftmaxScale:
        scale = 2.f;
        break;
      case Invalid::SigmoidNonfiniteScale:
        scale = std::numeric_limits<float>::infinity();
        break;
    }
    OpTester test("MoEPacked", 1, kMSDomain);
    test.AddAttribute("k", static_cast<int64_t>(topk));
    test.AddAttribute("activation_type", activation);
    test.AddAttribute("routing_function", std::string(sigmoid ? "sigmoid" : "softmax"));
    test.AddAttribute("normalize_routing_weights", static_cast<int64_t>(normalize));
    test.AddAttribute("router_scale", scale);
    test.AddAttribute("gate_type", static_cast<int64_t>(gate_type));
    test.AddAttribute("down_type", static_cast<int64_t>(down_type));
    if (up && invalid != Invalid::SwigluMissingUpType) test.AddAttribute("up_type", int64_t{16});
    test.AddInput<BFloat16>("input", {1, size}, std::vector<BFloat16>(size));
    if (probs) {
      test.AddInput<BFloat16>("probs", {1, probs_width}, std::vector<BFloat16>(probs_width));
    } else {
      test.AddOptionalInputEdge<BFloat16>();
    }
    const std::vector<uint8_t> bank(experts * size * 66);
    test.AddInput<uint8_t>("gate", {experts, size, 66}, bank, true);
    if (up) {
      test.AddInput<uint8_t>("up", {experts, size, 66}, bank, true);
    } else {
      test.AddOptionalInputEdge<uint8_t>();
    }
    test.AddInput<uint8_t>("down", {experts, size, 66}, bank, true);
    if (indices) {
      test.AddInput<int32_t>("indices", {1, index_width}, std::vector<int32_t>(index_width));
    } else {
      test.AddOptionalInputEdge<int32_t>();
    }
    if (weights) {
      test.AddInput<float>("weights", {1, weights_width}, std::vector<float>(weights_width, 1.f));
    } else {
      test.AddOptionalInputEdge<float>();
    }
    if (sigmoid) {
      test.AddInput<BFloat16>("router_input", {1, width}, std::vector<BFloat16>(width));
      test.AddInput<BFloat16>("router_weight", {experts, weight_width}, std::vector<BFloat16>(experts * weight_width), true);
      test.AddInput<float>("router_bias", {bias_count}, std::vector<float>(bias_count), true);
    }
    test.AddOutput<BFloat16>("output", {1, size}, std::vector<BFloat16>(size));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(SessionOptions{}, OpTester::ExpectResult::kExpectFailure, message, {}, nullptr, &providers);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
