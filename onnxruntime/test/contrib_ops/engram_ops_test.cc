// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <vector>

#include <memory>

#include "gtest/gtest.h"
#include "core/framework/execution_provider.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr float kEpsilon = 1.0e-5f;

float Sigmoid(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// Returns false when the execution provider required by T is unavailable. This must be checked before
// an OpTester is constructed: BaseTester's destructor traps when a tester is destroyed without running.
template <typename T>
bool IsTypeSupported() {
  if constexpr (std::is_same_v<T, BFloat16>) {
    return DefaultCudaExecutionProvider() != nullptr;
  } else {
    return true;
  }
}

// Runs the tester on CUDA only for BFloat16, otherwise on the default set of execution providers.
template <typename T>
void RunOnSupportedProviders(OpTester& test) {
  if constexpr (std::is_same_v<T, BFloat16>) {
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
  } else {
    test.Run();
  }
}

// Reference for the gate pre-activation, sign(dot) * sqrt(max(abs(dot), 1e-6)).
// std::copysign is deliberately avoided: it maps a zero dot product to +sqrt(1e-6) instead of zero.
float GateArg(float dot) {
  if (dot == 0.0f) {
    return 0.0f;
  }
  const float magnitude = std::sqrt(std::max(std::abs(dot), 1.0e-6f));
  return dot < 0.0f ? -magnitude : magnitude;
}

// ---------------------------------------------------------------------------------------------
// ShortConv
// ---------------------------------------------------------------------------------------------

// Shape and attribute description shared by the ShortConv reference helpers and the tests.
struct ShortConvConfig {
  int64_t batch_size = 1;
  int64_t sequence_length = 1;
  int64_t hc_mult = 1;
  int64_t hidden_size = 1;
  int64_t kernel_size = 1;
  int64_t dilation = 1;
  bool apply_silu = true;

  int64_t channels() const { return hc_mult * hidden_size; }
  int64_t state_length() const { return (kernel_size - 1) * dilation; }
  // Flat index of element `c` of the (batch, position, hyper-connection) row, for a tensor whose
  // sequence extent is `length`.
  int64_t Offset(int64_t b, int64_t t, int64_t g, int64_t c, int64_t length) const {
    return ((b * length + t) * hc_mult + g) * hidden_size + c;
  }
};

// Inverse RMS of one (batch, position, hyper-connection) row of `input`.
float ShortConvInvRms(const std::vector<float>& input, const ShortConvConfig& config,
                      int64_t b, int64_t t, int64_t g) {
  float sum_sq = 0.0f;
  for (int64_t c = 0; c < config.hidden_size; ++c) {
    const float value = input[static_cast<size_t>(config.Offset(b, t, g, c, config.sequence_length))];
    sum_sq += value * value;
  }
  return 1.0f / std::sqrt(sum_sq / static_cast<float>(config.hidden_size) + kEpsilon);
}

// Reference ShortConv over the whole sequence, so a chunked run can be compared against the same
// values. `bias` may be empty.
std::vector<float> ShortConvReference(const std::vector<float>& input,
                                      const std::vector<float>& scale,
                                      const std::vector<float>& weight,
                                      const std::vector<float>& bias,
                                      const ShortConvConfig& config) {
  std::vector<float> output(input.size(), 0.0f);
  for (int64_t b = 0; b < config.batch_size; ++b) {
    for (int64_t t = 0; t < config.sequence_length; ++t) {
      for (int64_t g = 0; g < config.hc_mult; ++g) {
        for (int64_t c = 0; c < config.hidden_size; ++c) {
          const int64_t flat_channel = g * config.hidden_size + c;
          float sum = bias.empty() ? 0.0f : bias[static_cast<size_t>(flat_channel)];
          for (int64_t k = 0; k < config.kernel_size; ++k) {
            const int64_t source_t = t - (config.kernel_size - 1 - k) * config.dilation;
            if (source_t < 0) {
              continue;
            }
            const float normed =
                input[static_cast<size_t>(config.Offset(b, source_t, g, c, config.sequence_length))] *
                ShortConvInvRms(input, config, b, source_t, g) * scale[static_cast<size_t>(flat_channel)];
            sum += normed * weight[static_cast<size_t>(flat_channel * config.kernel_size + k)];
          }
          output[static_cast<size_t>(config.Offset(b, t, g, c, config.sequence_length))] =
              config.apply_silu ? sum * Sigmoid(sum) : sum;
        }
      }
    }
  }
  return output;
}

// Trailing `state_length` raw (un-normalized) rows of the whole sequence starting at
// `first_position`, i.e. the present_state a full run emits. Positions before the sequence are zero.
std::vector<float> ShortConvRawWindow(const std::vector<float>& input, const ShortConvConfig& config,
                                      int64_t first_position) {
  const int64_t state_length = config.state_length();
  std::vector<float> window(
      static_cast<size_t>(config.batch_size * state_length * config.channels()), 0.0f);
  for (int64_t b = 0; b < config.batch_size; ++b) {
    for (int64_t slot = 0; slot < state_length; ++slot) {
      const int64_t t = first_position + slot;
      if (t < 0) {
        continue;
      }
      for (int64_t g = 0; g < config.hc_mult; ++g) {
        for (int64_t c = 0; c < config.hidden_size; ++c) {
          window[static_cast<size_t>(config.Offset(b, slot, g, c, state_length))] =
              input[static_cast<size_t>(config.Offset(b, t, g, c, config.sequence_length))];
        }
      }
    }
  }
  return window;
}

// Deterministic spread of values that is not symmetric in any axis, so a transposed index would
// change the result.
std::vector<float> MakeRamp(size_t count, float start, float step) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = start + step * static_cast<float>(i) + ((i % 3 == 0) ? 0.5f : -0.25f);
  }
  return values;
}

template <typename T>
void RunShortConvTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  ShortConvConfig config;
  config.sequence_length = 2;
  config.hidden_size = 2;
  config.kernel_size = 2;
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> scale{1.0f, 2.0f};
  const std::vector<float> weight{0.25f, 0.5f, 0.75f, -0.5f};
  const std::vector<float> expected = ShortConvReference(input, scale, weight, {}, config);

  OpTester test("ShortConv", 1, kMSDomain);
  test.AddAttribute<int64_t>("dilation", 1);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddAttribute<std::string>("activation", "silu");
  test.AddInput<T>("input", {1, 2, 1, 2}, ToTensorType<T>(input));
  test.AddInput<T>("weight", {2, 1, 2}, ToTensorType<T>(weight));
  test.AddInput<T>("norm_scale", {1, 2}, ToTensorType<T>(scale));
  test.AddOptionalInputEdge<T>();
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("output", {1, 2, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);
  test.AddOptionalOutputEdge<T>();
  RunOnSupportedProviders<T>(test);
}

// Runs `config` over the given chunk boundaries, threading present_state into past_state, and checks
// every chunk against the corresponding slice of a full-sequence run.
template <typename T>
void RunShortConvChunkedTest(const ShortConvConfig& config, const std::vector<int64_t>& chunk_lengths,
                             bool with_bias, float tolerance) {
  const auto element_count = static_cast<size_t>(config.batch_size * config.sequence_length * config.channels());
  const std::vector<float> input = MakeRamp(element_count, -1.5f, 0.35f);
  const std::vector<float> scale = MakeRamp(static_cast<size_t>(config.channels()), 0.75f, 0.2f);
  const std::vector<float> weight =
      MakeRamp(static_cast<size_t>(config.channels() * config.kernel_size), -0.4f, 0.15f);
  const std::vector<float> bias =
      with_bias ? MakeRamp(static_cast<size_t>(config.channels()), 0.1f, -0.05f) : std::vector<float>{};
  const std::vector<float> full = ShortConvReference(input, scale, weight, bias, config);

  const int64_t state_length = config.state_length();
  int64_t first_position = 0;
  for (int64_t chunk_length : chunk_lengths) {
    ShortConvConfig chunk_config = config;
    chunk_config.sequence_length = chunk_length;

    std::vector<float> chunk(static_cast<size_t>(config.batch_size * chunk_length * config.channels()));
    std::vector<float> expected(chunk.size());
    for (int64_t b = 0; b < config.batch_size; ++b) {
      for (int64_t t = 0; t < chunk_length; ++t) {
        for (int64_t i = 0; i < config.channels(); ++i) {
          const auto dst = static_cast<size_t>((b * chunk_length + t) * config.channels() + i);
          const auto src =
              static_cast<size_t>((b * config.sequence_length + first_position + t) * config.channels() + i);
          chunk[dst] = input[src];
          expected[dst] = full[src];
        }
      }
    }
    const std::vector<float> past = ShortConvRawWindow(input, config, first_position - state_length);
    const std::vector<float> present =
        ShortConvRawWindow(input, config, first_position + chunk_length - state_length);

    OpTester test("ShortConv", 1, kMSDomain);
    test.AddAttribute<int64_t>("dilation", config.dilation);
    test.AddAttribute<float>("epsilon", kEpsilon);
    test.AddAttribute<std::string>("activation", config.apply_silu ? "silu" : "none");
    test.AddInput<T>("input", {config.batch_size, chunk_length, config.hc_mult, config.hidden_size},
                     ToTensorType<T>(chunk));
    test.AddInput<T>("weight", {config.channels(), 1, config.kernel_size}, ToTensorType<T>(weight));
    test.AddInput<T>("norm_scale", {config.hc_mult, config.hidden_size}, ToTensorType<T>(scale));
    if (with_bias) {
      test.AddInput<T>("bias", {config.channels()}, ToTensorType<T>(bias));
    } else {
      test.AddOptionalInputEdge<T>();
    }
    if (first_position == 0) {
      test.AddOptionalInputEdge<T>();
    } else {
      test.AddInput<T>("past_state", {config.batch_size, state_length, config.hc_mult, config.hidden_size},
                       ToTensorType<T>(past));
    }
    test.AddOutput<T>("output", {config.batch_size, chunk_length, config.hc_mult, config.hidden_size},
                      ToTensorType<T>(expected), false, tolerance, tolerance);
    test.AddOutput<T>("present_state",
                      {config.batch_size, state_length, config.hc_mult, config.hidden_size},
                      ToTensorType<T>(present), false, tolerance, tolerance);
    RunOnSupportedProviders<T>(test);

    first_position += chunk_length;
  }
}

// ---------------------------------------------------------------------------------------------
// EngramGate
// ---------------------------------------------------------------------------------------------

// Reference EngramGate for a single (token, hyper-connection) row with unit norm scales.
float EngramGateReference(const std::vector<float>& key, const std::vector<float>& query) {
  const auto hidden_size = static_cast<float>(key.size());
  float key_sum_sq = 0.0f;
  float query_sum_sq = 0.0f;
  for (size_t c = 0; c < key.size(); ++c) {
    key_sum_sq += key[c] * key[c];
    query_sum_sq += query[c] * query[c];
  }
  const float key_inv = 1.0f / std::sqrt(key_sum_sq / hidden_size + kEpsilon);
  const float query_inv = 1.0f / std::sqrt(query_sum_sq / hidden_size + kEpsilon);
  float dot = 0.0f;
  for (size_t c = 0; c < key.size(); ++c) {
    dot += key[c] * key_inv * query[c] * query_inv;
  }
  dot /= std::sqrt(hidden_size);
  return Sigmoid(GateArg(dot));
}

template <typename T>
void RunEngramGateTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  const std::vector<float> key{0.0f, 2.5f};
  const std::vector<float> query{3.0f, 4.0f};
  const std::vector<float> value{2.0f, -1.5f};
  const std::vector<float> unit_scale{1.0f, 1.0f};

  const float gate = EngramGateReference(key, query);
  const std::vector<float> expected{gate * value[0], gate * value[1]};

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<T>("key", {1, 1, 1, 2}, ToTensorType<T>(key));
  test.AddInput<T>("query", {1, 1, 1, 2}, ToTensorType<T>(query));
  test.AddInput<T>("value", {1, 1, 2}, ToTensorType<T>(value));
  test.AddInput<T>("key_norm_scale", {1, 2}, ToTensorType<T>(unit_scale));
  test.AddInput<T>("query_norm_scale", {1, 2}, ToTensorType<T>(unit_scale));
  test.AddOutput<T>("output", {1, 1, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);
  RunOnSupportedProviders<T>(test);
}

// ---------------------------------------------------------------------------------------------
// NGramHashMapping
// ---------------------------------------------------------------------------------------------

constexpr int64_t kMaxNGramSize = 3;
constexpr int64_t kHeadsPerNGram = 2;
constexpr int64_t kPadId = 9;

// Reference NGramHashMapping for a single batch row. `history` holds the kMaxNGramSize - 1 ids that
// precede `ids`, right-aligned, and lets the same reference cover both the full and the chunked runs.
template <typename T>
std::vector<T> NGramHashMappingReference(const std::vector<T>& ids,
                                         const std::vector<T>& history,
                                         const std::vector<T>& multipliers,
                                         const std::vector<T>& vocab_sizes) {
  const int64_t sequence_length = static_cast<int64_t>(ids.size());
  const int64_t state_length = kMaxNGramSize - 1;
  const int64_t num_heads = state_length * kHeadsPerNGram;
  std::vector<T> output(static_cast<size_t>(sequence_length * num_heads));

  auto id_at = [&](int64_t t) -> T {
    if (t >= 0) {
      return ids[static_cast<size_t>(t)];
    }
    const int64_t slot = state_length + t;
    if (history.empty() || slot < 0) {
      return static_cast<T>(kPadId);
    }
    return history[static_cast<size_t>(slot)];
  };

  for (int64_t t = 0; t < sequence_length; ++t) {
    for (int64_t n = 2; n <= kMaxNGramSize; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        // Multiplication wraps on overflow, matching the kernel's unsigned arithmetic.
        using U = std::make_unsigned_t<T>;
        const T product = static_cast<T>(static_cast<U>(id_at(t - k)) *
                                         static_cast<U>(multipliers[static_cast<size_t>(k)]));
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }
      for (int64_t h = 0; h < kHeadsPerNGram; ++h) {
        const int64_t out_h = (n - 2) * kHeadsPerNGram + h;
        const T mod = vocab_sizes[static_cast<size_t>(out_h)];
        T value = static_cast<T>(mix % mod);
        if (value < 0) {
          value = static_cast<T>(value + mod);
        }
        output[static_cast<size_t>(t * num_heads + out_h)] = value;
      }
    }
  }
  return output;
}

template <typename T>
void RunNGramHashMappingTest() {
  const std::vector<T> ids{3, 4, 5, 6};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  const std::vector<T> expected = NGramHashMappingReference<T>(ids, {}, multipliers, vocab_sizes);
  // Guards the reference itself against silent drift.
  ASSERT_EQ(expected, (std::vector<T>{84, 84, 98, 96,
                                      11, 11, 39, 37,
                                      3, 3, 48, 48,
                                      3, 3, 71, 71}));

  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {1, 4}, ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {1, 4, 4}, expected);
  test.AddOutput<T>("present_ids", {1, 2}, {ids[2], ids[3]});
  test.Run();
}

}  // namespace

TEST(EngramOpsTest, NGramHashMappingInt64) {
  RunNGramHashMappingTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so it must be covered explicitly.
TEST(EngramOpsTest, NGramHashMappingInt32) {
  RunNGramHashMappingTest<int32_t>();
}

// A decode step must hash the same n-gram window as the corresponding position of a full-sequence
// run. Without past_ids the preceding tokens would silently fall back to pad_id.
TEST(EngramOpsTest, NGramHashMappingChunkedMatchesFullSequence) {
  using T = int64_t;
  const std::vector<T> ids{3, 4, 5, 6};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  const std::vector<T> full = NGramHashMappingReference<T>(ids, {}, multipliers, vocab_sizes);

  auto run_chunk = [&](const std::vector<T>& chunk, const std::vector<T>& past,
                       const std::vector<T>& expected_hash_ids, const std::vector<T>& expected_present) {
    OpTester test("NGramHashMapping", 1, kMSDomain);
    test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
    test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
    test.AddAttribute<int64_t>("pad_id", kPadId);
    test.AddInput<T>("input_ids", {1, static_cast<int64_t>(chunk.size())}, chunk);
    test.AddInput<T>("multipliers", {3}, multipliers);
    test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
    if (past.empty()) {
      test.AddOptionalInputEdge<T>();
    } else {
      test.AddInput<T>("past_ids", {1, 2}, past);
    }
    test.AddOutput<T>("hash_ids", {1, static_cast<int64_t>(chunk.size()), 4}, expected_hash_ids);
    test.AddOutput<T>("present_ids", {1, 2}, expected_present);
    test.Run();
  };

  // Prefill of the first two tokens, then a decode step per remaining token, threading present_ids.
  const std::vector<T> prefill{ids[0], ids[1]};
  run_chunk(prefill, {}, std::vector<T>(full.begin(), full.begin() + 8), {ids[0], ids[1]});

  // Decode token 2 with the prefill history.
  run_chunk({ids[2]}, {ids[0], ids[1]}, std::vector<T>(full.begin() + 8, full.begin() + 12),
            {ids[1], ids[2]});

  // Decode token 3 with the history returned by the previous step.
  run_chunk({ids[3]}, {ids[1], ids[2]}, std::vector<T>(full.begin() + 12, full.end()),
            {ids[2], ids[3]});
}

TEST(EngramOpsTest, ShortConvFloat) {
  RunShortConvTest<float>(1e-4f);
}

TEST(EngramOpsTest, ShortConvFloat16) {
  RunShortConvTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, EngramGateFloat) {
  RunEngramGateTest<float>(1e-4f);
}

TEST(EngramOpsTest, EngramGateFloat16) {
  RunEngramGateTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, ShortConvBFloat16) {
  RunShortConvTest<BFloat16>(2e-2f);
}

TEST(EngramOpsTest, EngramGateBFloat16) {
  RunEngramGateTest<BFloat16>(2e-2f);
}

// A decode step must see the same convolution taps as the corresponding position of a full-sequence
// run. Without past_state the earlier taps would silently be dropped.
TEST(EngramOpsTest, ShortConvChunkedMatchesFullSequence) {
  ShortConvConfig config;
  config.sequence_length = 4;
  config.hidden_size = 2;
  config.kernel_size = 3;
  // Prefill of the first two tokens, then one decode step per remaining token. The decode steps are
  // the interesting ones: two of the three taps can only come from past_state.
  RunShortConvChunkedTest<float>(config, {2, 1, 1}, /*with_bias=*/false, 1e-4f);
}

// batch_size, hc_mult and dilation are all > 1 here, so a transposed batch stride, a dropped
// hyper-connection offset in norm_scale, or a state window sized as kernel_size - 1 instead of
// (kernel_size - 1) * dilation would all change the result.
TEST(EngramOpsTest, ShortConvChunkedBatchedDilated) {
  ShortConvConfig config;
  config.batch_size = 2;
  config.sequence_length = 6;
  config.hc_mult = 2;
  config.hidden_size = 3;
  config.kernel_size = 3;
  config.dilation = 2;
  RunShortConvChunkedTest<float>(config, {4, 1, 1}, /*with_bias=*/false, 1e-4f);
}

// Bias and the "none" activation are otherwise never exercised.
TEST(EngramOpsTest, ShortConvChunkedBiasNoActivation) {
  ShortConvConfig config;
  config.batch_size = 2;
  config.sequence_length = 4;
  config.hc_mult = 2;
  config.hidden_size = 2;
  config.kernel_size = 3;
  config.apply_silu = false;
  RunShortConvChunkedTest<float>(config, {2, 1, 1}, /*with_bias=*/true, 1e-4f);
}

// present_state carries the raw input window, so the RMS reduction is recomputed from the same values
// a full-sequence run would use. That is what makes the chunked equivalence hold in float16 too.
TEST(EngramOpsTest, ShortConvChunkedFloat16) {
  if (!IsTypeSupported<MLFloat16>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  ShortConvConfig config;
  config.batch_size = 2;
  config.sequence_length = 4;
  config.hc_mult = 2;
  config.hidden_size = 2;
  config.kernel_size = 3;
  RunShortConvChunkedTest<MLFloat16>(config, {2, 1, 1}, /*with_bias=*/false, 5e-3f);
}

// A zero dot product must produce a gate of exactly 0.5 on every EP. Orthogonal key/query rows make
// the dot product vanish, which would silently become sigmoid(sqrt(1e-6)) if copysign were used.
TEST(EngramOpsTest, EngramGateZeroDotProduct) {
  const std::vector<float> key{1.0f, 0.0f};
  const std::vector<float> query{0.0f, 1.0f};
  const std::vector<float> value{1.0f, -1.0f};
  const std::vector<float> unit_scale{1.0f, 1.0f};
  const std::vector<float> expected{0.5f * value[0], 0.5f * value[1]};

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<float>("key", {1, 1, 1, 2}, key);
  test.AddInput<float>("query", {1, 1, 1, 2}, query);
  test.AddInput<float>("value", {1, 1, 2}, value);
  test.AddInput<float>("key_norm_scale", {1, 2}, unit_scale);
  test.AddInput<float>("query_norm_scale", {1, 2}, unit_scale);
  test.AddOutput<float>("output", {1, 1, 1, 2}, expected, false, 1e-5f, 1e-5f);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
