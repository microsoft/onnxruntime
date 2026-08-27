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

// Reference ShortConv for batch_size = 1 and hc_mult = 1. Returns the SiLU-activated output for
// every position of the whole sequence, so a chunked run can be compared against the same values.
std::vector<float> ShortConvReference(const std::vector<float>& input,
                                      const std::vector<float>& scale,
                                      const std::vector<float>& weight,
                                      int64_t sequence_length,
                                      int64_t hidden_size,
                                      int64_t kernel_size) {
  std::vector<float> normed(input.size());
  for (int64_t t = 0; t < sequence_length; ++t) {
    float sum_sq = 0.0f;
    for (int64_t c = 0; c < hidden_size; ++c) {
      const float value = input[t * hidden_size + c];
      sum_sq += value * value;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + kEpsilon);
    for (int64_t c = 0; c < hidden_size; ++c) {
      normed[t * hidden_size + c] = input[t * hidden_size + c] * inv_rms * scale[c];
    }
  }

  std::vector<float> output(input.size());
  for (int64_t t = 0; t < sequence_length; ++t) {
    for (int64_t c = 0; c < hidden_size; ++c) {
      float sum = 0.0f;
      for (int64_t k = 0; k < kernel_size; ++k) {
        const int64_t source_t = t - (kernel_size - 1 - k);
        if (source_t < 0) {
          continue;
        }
        sum += normed[source_t * hidden_size + c] * weight[c * kernel_size + k];
      }
      output[t * hidden_size + c] = sum * Sigmoid(sum);
    }
  }
  return output;
}

// Trailing `state_length` normed rows of the whole sequence, i.e. the present_state a full run emits.
std::vector<float> ShortConvNormedWindow(const std::vector<float>& input,
                                         const std::vector<float>& scale,
                                         int64_t first_position,
                                         int64_t state_length,
                                         int64_t hidden_size) {
  std::vector<float> window(static_cast<size_t>(state_length * hidden_size), 0.0f);
  for (int64_t slot = 0; slot < state_length; ++slot) {
    const int64_t t = first_position + slot;
    if (t < 0) {
      continue;
    }
    float sum_sq = 0.0f;
    for (int64_t c = 0; c < hidden_size; ++c) {
      const float value = input[t * hidden_size + c];
      sum_sq += value * value;
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + kEpsilon);
    for (int64_t c = 0; c < hidden_size; ++c) {
      window[slot * hidden_size + c] = input[t * hidden_size + c] * inv_rms * scale[c];
    }
  }
  return window;
}

template <typename T>
void RunShortConvTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> scale{1.0f, 2.0f};
  const std::vector<float> weight{0.25f, 0.5f, 0.75f, -0.5f};
  const std::vector<float> expected = ShortConvReference(input, scale, weight, 2, 2, 2);

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
  constexpr int64_t kSequenceLength = 4;
  constexpr int64_t kHiddenSize = 2;
  constexpr int64_t kKernelSize = 3;
  constexpr int64_t kStateLength = kKernelSize - 1;
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 2.0f, -3.0f};
  const std::vector<float> scale{1.0f, 2.0f};
  const std::vector<float> weight{0.25f, 0.5f, 0.75f, -0.5f, 0.125f, 1.0f};
  const std::vector<float> full =
      ShortConvReference(input, scale, weight, kSequenceLength, kHiddenSize, kKernelSize);

  auto run_chunk = [&](int64_t first_position, int64_t chunk_length) {
    const auto chunk_begin = static_cast<size_t>(first_position * kHiddenSize);
    const auto chunk_end = static_cast<size_t>((first_position + chunk_length) * kHiddenSize);
    const std::vector<float> chunk(input.begin() + chunk_begin, input.begin() + chunk_end);
    const std::vector<float> expected(full.begin() + chunk_begin, full.begin() + chunk_end);
    const std::vector<float> past =
        ShortConvNormedWindow(input, scale, first_position - kStateLength, kStateLength, kHiddenSize);
    const std::vector<float> present = ShortConvNormedWindow(
        input, scale, first_position + chunk_length - kStateLength, kStateLength, kHiddenSize);

    OpTester test("ShortConv", 1, kMSDomain);
    test.AddAttribute<int64_t>("dilation", 1);
    test.AddAttribute<float>("epsilon", kEpsilon);
    test.AddAttribute<std::string>("activation", "silu");
    test.AddInput<float>("input", {1, chunk_length, 1, kHiddenSize}, chunk);
    test.AddInput<float>("weight", {kHiddenSize, 1, kKernelSize}, weight);
    test.AddInput<float>("norm_scale", {1, kHiddenSize}, scale);
    test.AddOptionalInputEdge<float>();
    if (first_position == 0) {
      test.AddOptionalInputEdge<float>();
    } else {
      test.AddInput<float>("past_state", {1, kStateLength, 1, kHiddenSize}, past);
    }
    test.AddOutput<float>("output", {1, chunk_length, 1, kHiddenSize}, expected, false, 1e-4f, 1e-4f);
    test.AddOutput<float>("present_state", {1, kStateLength, 1, kHiddenSize}, present, false, 1e-4f, 1e-4f);
    test.Run();
  };

  // Prefill of the first two tokens, then one decode step per remaining token. The decode
  // steps are the interesting ones: two of the three taps can only come from past_state.
  run_chunk(0, 2);
  run_chunk(2, 1);
  run_chunk(3, 1);
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

namespace {

// ShortConvWithState: Stateful dilated depthwise conv with branchwise RMSNorm.
// Tests the key invariant: ShortConv(full_seq) == concat(ShortConvWithState(one_token_steps)).
template <typename T>
void RunShortConvWithStateTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr float epsilon = 1.0e-5f;
  constexpr int64_t kernel_size = 2;
  constexpr int64_t dilation = 1;
  constexpr int64_t state_len = dilation * (kernel_size - 1);  // 1

  // Input: [B=1, S=2, hc_mult=1, H=2]
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> scale{1.0f, 2.0f};
  const std::vector<float> weight{0.25f, 0.5f, 0.75f, -0.5f};

  // Compute the reference: normalize each row, then do causal conv + silu.
  // Row 0: [1, 2], inv_rms = 1/sqrt((1+4)/2 + eps)
  // Row 1: [3, 4], inv_rms = 1/sqrt((9+16)/2 + eps)
  std::vector<float> normed(4);
  for (int64_t t = 0; t < 2; ++t) {
    float sum_sq = input[t * 2] * input[t * 2] + input[t * 2 + 1] * input[t * 2 + 1];
    float inv_rms = 1.0f / std::sqrt(sum_sq / 2.0f + epsilon);
    for (int64_t c = 0; c < 2; ++c) {
      normed[t * 2 + c] = input[t * 2 + c] * inv_rms * scale[c];
    }
  }
  std::vector<float> expected(4);
  for (int64_t t = 0; t < 2; ++t) {
    for (int64_t c = 0; c < 2; ++c) {
      float sum = 0.0f;
      // k=0: source_t = t - (K-1-0)*dilation = t-1
      if (t > 0) {
        sum += normed[(t - 1) * 2 + c] * weight[c * kernel_size + 0];
      }
      // k=1: source_t = t
      sum += normed[t * 2 + c] * weight[c * kernel_size + 1];
      expected[t * 2 + c] = sum * Sigmoid(sum);
    }
  }

  // Full-sequence test (no past state).
  {
    OpTester test("ShortConvWithState", 1, kMSDomain);
    test.AddAttribute<int64_t>("dilation", dilation);
    test.AddAttribute<float>("epsilon", epsilon);
    test.AddAttribute<std::string>("activation", "silu");
    test.AddInput<T>("input", {1, 2, 1, 2}, ToTensorType<T>(input));
    test.AddOptionalInputEdge<T>();  // no past_state
    test.AddInput<T>("norm_scale", {1, 2}, ToTensorType<T>(scale));
    test.AddInput<T>("weight", {2, 1, 2}, ToTensorType<T>(weight));
    test.AddOptionalInputEdge<T>();  // no bias
    test.AddOutput<T>("output", {1, 2, 1, 2}, ToTensorType<T>(expected), false, tolerance, tolerance);

    // present_state: last state_len normed values per channel. state_len=1, so it's normed[t=1].
    // present_state shape: [B=1, C=2, state_len=1]
    std::vector<float> expected_state{normed[2], normed[3]};  // normed at t=1 for c=0, c=1
    test.AddOutput<T>("present_state", {1, 2, 1}, ToTensorType<T>(expected_state), false, tolerance, tolerance);
    RunOnSupportedProviders<T>(test);
  }

  // Token-by-token decode test: step 0 (no past), then step 1 (with past state from step 0).
  // Step 0: input = [1, 2], no past_state
  {
    std::vector<float> step0_input{1.0f, 2.0f};
    std::vector<float> step0_expected{expected[0], expected[1]};
    std::vector<float> step0_state{normed[0], normed[1]};

    OpTester test("ShortConvWithState", 1, kMSDomain);
    test.AddAttribute<int64_t>("dilation", dilation);
    test.AddAttribute<float>("epsilon", epsilon);
    test.AddAttribute<std::string>("activation", "silu");
    test.AddInput<T>("input", {1, 1, 1, 2}, ToTensorType<T>(step0_input));
    test.AddOptionalInputEdge<T>();
    test.AddInput<T>("norm_scale", {1, 2}, ToTensorType<T>(scale));
    test.AddInput<T>("weight", {2, 1, 2}, ToTensorType<T>(weight));
    test.AddOptionalInputEdge<T>();
    test.AddOutput<T>("output", {1, 1, 1, 2}, ToTensorType<T>(step0_expected), false, tolerance, tolerance);
    test.AddOutput<T>("present_state", {1, 2, 1}, ToTensorType<T>(step0_state), false, tolerance, tolerance);
    RunOnSupportedProviders<T>(test);
  }

  // Step 1: input = [3, 4], past_state = normed values from step 0
  {
    std::vector<float> step1_input{3.0f, 4.0f};
    std::vector<float> step1_past{normed[0], normed[1]};
    std::vector<float> step1_expected{expected[2], expected[3]};
    std::vector<float> step1_state{normed[2], normed[3]};

    OpTester test("ShortConvWithState", 1, kMSDomain);
    test.AddAttribute<int64_t>("dilation", dilation);
    test.AddAttribute<float>("epsilon", epsilon);
    test.AddAttribute<std::string>("activation", "silu");
    test.AddInput<T>("input", {1, 1, 1, 2}, ToTensorType<T>(step1_input));
    test.AddInput<T>("past_state", {1, 2, 1}, ToTensorType<T>(step1_past));
    test.AddInput<T>("norm_scale", {1, 2}, ToTensorType<T>(scale));
    test.AddInput<T>("weight", {2, 1, 2}, ToTensorType<T>(weight));
    test.AddOptionalInputEdge<T>();
    test.AddOutput<T>("output", {1, 1, 1, 2}, ToTensorType<T>(step1_expected), false, tolerance, tolerance);
    test.AddOutput<T>("present_state", {1, 2, 1}, ToTensorType<T>(step1_state), false, tolerance, tolerance);
    RunOnSupportedProviders<T>(test);
  }
}

// ShortConvWithState with dilation > 1 (matching Qwen4-Exp's typical ngram_size=3 dilation).
template <typename T>
void RunShortConvWithStateDilatedTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr float epsilon = 1.0e-5f;
  constexpr int64_t kernel_size = 2;
  constexpr int64_t dilation = 3;
  constexpr int64_t state_len = dilation * (kernel_size - 1);  // 3

  // Input: [B=1, S=4, hc_mult=1, H=1] — single channel for simplicity.
  const std::vector<float> input{1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> scale{1.0f};
  const std::vector<float> weight{0.5f, 1.0f};  // [C=1, 1, K=2]

  // Normalize: with H=1, RMSNorm(x) = x * 1/sqrt(x^2 + eps) * scale = sign(x) * scale (approx for large |x|)
  // More precisely: normed = x / sqrt(x^2 + eps) * 1.0
  std::vector<float> normed(4);
  for (int64_t t = 0; t < 4; ++t) {
    float x = input[t];
    normed[t] = x / std::sqrt(x * x + epsilon) * scale[0];
  }

  // Conv with dilation=3, K=2: for output at time t,
  // k=0: source = t - (2-1-0)*3 = t - 3
  // k=1: source = t
  std::vector<float> expected(4);
  for (int64_t t = 0; t < 4; ++t) {
    float sum = 0.0f;
    int64_t src0 = t - 3;
    if (src0 >= 0) {
      sum += normed[src0] * weight[0];
    }
    sum += normed[t] * weight[1];
    expected[t] = sum * Sigmoid(sum);  // SiLU
  }

  OpTester test("ShortConvWithState", 1, kMSDomain);
  test.AddAttribute<int64_t>("dilation", dilation);
  test.AddAttribute<float>("epsilon", epsilon);
  test.AddAttribute<std::string>("activation", "silu");
  test.AddInput<T>("input", {1, 4, 1, 1}, ToTensorType<T>(input));
  test.AddOptionalInputEdge<T>();
  test.AddInput<T>("norm_scale", {1, 1}, ToTensorType<T>(scale));
  test.AddInput<T>("weight", {1, 1, 2}, ToTensorType<T>(weight));
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("output", {1, 4, 1, 1}, ToTensorType<T>(expected), false, tolerance, tolerance);

  // present_state: last 3 normed values (state_len = 3)
  std::vector<float> expected_state{normed[1], normed[2], normed[3]};
  test.AddOutput<T>("present_state", {1, 1, 3}, ToTensorType<T>(expected_state), false, tolerance, tolerance);
  RunOnSupportedProviders<T>(test);
}

}  // namespace

TEST(EngramOpsTest, ShortConvWithStateFloat) {
  RunShortConvWithStateTest<float>(1e-4f);
}

TEST(EngramOpsTest, ShortConvWithStateFloat16) {
  RunShortConvWithStateTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, ShortConvWithStateBFloat16) {
  RunShortConvWithStateTest<BFloat16>(2e-2f);
}

TEST(EngramOpsTest, ShortConvWithStateDilatedFloat) {
  RunShortConvWithStateDilatedTest<float>(1e-4f);
}

TEST(EngramOpsTest, ShortConvWithStateDilatedFloat16) {
  RunShortConvWithStateDilatedTest<MLFloat16>(2e-3f);
}

}  // namespace test
}  // namespace onnxruntime
