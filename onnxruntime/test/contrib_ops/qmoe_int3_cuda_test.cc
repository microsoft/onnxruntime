#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_CUDA
namespace onnxruntime::test {
namespace {

void PackInt3TestCode(std::vector<uint8_t>& packed, int row, int count, int index, int bits, int code) {
  const int bytes = (count * bits + 7) / 8;
  for (int bit = 0; bit < bits; ++bit) {
    const int position = index * bits + bit;
    packed[row * bytes + position / 8] |= ((code >> bit) & 1) << (position % 8);
  }
}

template <typename T>
void RunInt3CudaCase(int block, int fc1_bits, int fc2_bits, bool separate_routing,
                     bool initializers, int malformed = 0) {
  auto provider = DefaultCudaExecutionProvider();
  if (!provider) GTEST_SKIP() << "CUDA execution provider unavailable";
  constexpr int64_t rows = 5;
  constexpr int64_t experts = 3;
  constexpr int64_t top_k = 2;
  const int64_t hidden = block * 2;
  const int64_t inter = block * 3;
  const float scale = 0.015625f;
  std::vector<T> input(rows * hidden);
  for (int64_t index = 0; index < rows * hidden; ++index) input[index] = T((index % 11 - 5) * 0.0625f);
  std::vector<T> router(rows * experts, T(0.0f));
  std::vector<T> external_routing(rows * experts);
  for (int64_t index = 0; index < rows * experts; ++index) external_routing[index] = T(0.125f * (1 + index % experts));
  std::vector<uint8_t> first(experts * inter * 2 * (hidden * fc1_bits / 8));
  std::vector<uint8_t> second(experts * hidden * (inter * fc2_bits / 8));
  std::vector<T> first_scales(experts * inter * 2 * (hidden / block), T(scale));
  std::vector<T> second_scales(experts * hidden * (inter / block), T(scale));
  std::vector<T> first_bias(experts * inter * 2, T(0.015625f));
  std::vector<T> second_bias(experts * hidden, T(0.0078125f));
  for (int64_t row = 0; row < experts * inter * 2; ++row) {
    for (int64_t feature = 0; feature < hidden; ++feature) {
      PackInt3TestCode(first, row, hidden, feature, fc1_bits, (row * 5 + feature * 3) % (1 << fc1_bits));
    }
  }
  for (int64_t row = 0; row < experts * hidden; ++row) {
    for (int64_t feature = 0; feature < inter; ++feature) {
      PackInt3TestCode(second, row, inter, feature, fc2_bits, (row * 3 + feature * 5) % (1 << fc2_bits));
    }
  }
  std::vector<T> expected(rows * hidden);
  for (int64_t token = 0; token < rows; ++token) {
    std::vector<float> accumulated(hidden, 0);
    for (int64_t expert = 0; expert < top_k; ++expert) {
      std::vector<T> activation(inter);
      for (int64_t column = 0; column < inter; ++column) {
        const int64_t weight_row = expert * inter * 2 + column * 2;
        float gate = first_bias[weight_row].ToFloat();
        float up = first_bias[weight_row + 1].ToFloat();
        for (int64_t feature = 0; feature < hidden; ++feature) {
          const float value = input[token * hidden + feature].ToFloat();
          gate += value * (((weight_row * 5 + feature * 3) % (1 << fc1_bits)) - (1 << (fc1_bits - 1))) * scale;
          up += value * ((((weight_row + 1) * 5 + feature * 3) % (1 << fc1_bits)) - (1 << (fc1_bits - 1))) * scale;
        }
        activation[column] = T(gate / (1 + std::exp(-gate)) * up);
      }
      for (int64_t column = 0; column < hidden; ++column) {
        const int64_t weight_row = expert * hidden + column;
        float value = 0;
        for (int64_t feature = 0; feature < inter; ++feature) {
          value += activation[feature].ToFloat() *
                   (((weight_row * 3 + feature * 5) % (1 << fc2_bits)) - (1 << (fc2_bits - 1))) * scale;
        }
        value += second_bias[weight_row].ToFloat();
        const float routing_weight = separate_routing ? external_routing[token * experts + expert].ToFloat() : 0.5f;
        accumulated[column] += T(value).ToFloat() * routing_weight;
      }
    }
    for (int64_t column = 0; column < hidden; ++column) expected[token * hidden + column] = T(accumulated[column]);
  }
  const int64_t first_zero_bytes = ((hidden / block) * fc1_bits + 7) / 8;
  const int64_t second_zero_bytes = ((inter / block) * fc2_bits + 7) / 8;
  std::vector<uint8_t> first_zeros(experts * inter * 2 * first_zero_bytes);
  std::vector<uint8_t> second_zeros(experts * hidden * second_zero_bytes);
  for (int64_t row = 0; row < experts * inter * 2; ++row) {
    for (int64_t group = 0; group < hidden / block; ++group) {
      PackInt3TestCode(first_zeros, row, hidden / block, group, fc1_bits, 1 << (fc1_bits - 1));
    }
  }
  for (int64_t row = 0; row < experts * hidden; ++row) {
    for (int64_t group = 0; group < inter / block; ++group) {
      PackInt3TestCode(second_zeros, row, inter / block, group, fc2_bits, 1 << (fc2_bits - 1));
    }
  }
  if (malformed == 1) first_scales[0] = T(-1.0f);
  if (malformed == 2) second_scales[0] = T(std::numeric_limits<float>::infinity());
  if (malformed == 3) first_zeros[0] ^= 1;
  if (malformed == 4) first_zeros[0] |= 0x80;
  OpTester tester("QMoE", 1, kMSDomain);
  tester.AddAttribute<int64_t>("k", top_k);
  tester.AddAttribute<std::string>("activation_type", "swiglu");
  tester.AddAttribute<int64_t>("swiglu_fusion", 1);
  tester.AddAttribute<int64_t>("normalize_routing_weights", 1);
  tester.AddAttribute<int64_t>("expert_weight_bits", fc1_bits);
  tester.AddAttribute<int64_t>("fc2_expert_weight_bits", fc2_bits);
  tester.AddAttribute<int64_t>("block_size", block);
  tester.AddAttribute<int64_t>("weights_prepacked", 0);
  tester.AddInput<T>("input", {1, rows, hidden}, input);
  tester.AddInput<T>("router_probs", {rows, experts}, router);
  tester.AddInput<uint8_t>("fc1_experts_weights", {experts, 2 * inter, hidden * fc1_bits / 8}, first, initializers);
  tester.AddInput<T>("fc1_scales", {experts, 2 * inter, hidden / block}, first_scales, initializers);
  tester.AddInput<T>("fc1_experts_bias", {experts, 2 * inter}, first_bias, initializers);
  tester.AddInput<uint8_t>("fc2_experts_weights", {experts, hidden, inter * fc2_bits / 8}, second, initializers);
  tester.AddInput<T>("fc2_scales", {experts, hidden, inter / block}, second_scales, initializers);
  tester.AddInput<T>("fc2_experts_bias", {experts, hidden}, second_bias, initializers);
  tester.AddOptionalInputEdge<uint8_t>();
  tester.AddOptionalInputEdge<T>();
  tester.AddOptionalInputEdge<T>();
  tester.AddInput<uint8_t>("fc1_zero_points", {experts, 2 * inter, first_zero_bytes}, first_zeros, initializers);
  tester.AddInput<uint8_t>("fc2_zero_points", {experts, hidden, second_zero_bytes}, second_zeros, initializers);
  tester.AddOptionalInputEdge<uint8_t>();
  if (separate_routing) tester.AddInput<T>("router_weights", {rows, experts}, external_routing);
  tester.AddOutput<T>("output", {1, rows, hidden}, expected);
  tester.SetOutputAbsErr("output", 0.001f);
  tester.SetOutputRelErr("output", 0.02f);
  SessionOptions options;
  options.config_options.configurations["ep.cuda.qmoe_row_tile_size"] = "2";
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(std::move(provider));
  tester.Run(options, malformed ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
             malformed ? "finite nonnegative scales and INT3 zero points equal to 4 with zero padding" : "",
             {}, nullptr, &providers);
}

TEST(QMoEInt3CudaTest, MixedWidthsAndTiledRank3Input) {
  for (int block : {32, 64, 128}) {
    for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
      SCOPED_TRACE(::testing::Message() << block << ":" << bits.first << "/" << bits.second);
      RunInt3CudaCase<MLFloat16>(block, bits.first, bits.second, false, true);
      RunInt3CudaCase<BFloat16>(block, bits.first, bits.second, true, false);
    }
  }
}

TEST(QMoEInt3CudaTest, RejectsMalformedDeviceContent) {
  for (int malformed : {1, 2, 3, 4}) {
    RunInt3CudaCase<MLFloat16>(32, 3, 3, false, malformed % 2 == 0, malformed);
    RunInt3CudaCase<BFloat16>(32, 3, 3, false, malformed % 2 == 1, malformed);
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif