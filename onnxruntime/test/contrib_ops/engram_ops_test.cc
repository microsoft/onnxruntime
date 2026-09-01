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

// Deterministic ramp of `count` values starting at `start` and advancing by `step`. Deterministic
// inputs keep the hand-computed expectations in the vectorized tests reproducible.
std::vector<float> MakeRamp(size_t count, float start, float step) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = start + step * static_cast<float>(i);
  }
  return values;
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

// Exercises hidden_size == 4 with hc_mult > 1 and non-unit norm scales. On WebGPU this is the only
// case that selects the vec4 component path through the gate reduction and the broadcast pass, and
// hc_mult > 1 makes a per-row rather than per-token scale lookup observable.
template <typename T>
void RunEngramGateVectorizedTest(float tolerance) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSequence = 2;
  constexpr int64_t kHcMult = 2;
  constexpr int64_t kHidden = 4;
  constexpr int64_t kRows = kBatch * kSequence * kHcMult;

  const std::vector<float> key = MakeRamp(static_cast<size_t>(kRows * kHidden), -0.9f, 0.3f);
  const std::vector<float> query = MakeRamp(static_cast<size_t>(kRows * kHidden), 1.2f, -0.25f);
  const std::vector<float> value = MakeRamp(static_cast<size_t>(kBatch * kSequence * kHidden), 0.4f, 0.35f);
  const std::vector<float> key_scale = MakeRamp(static_cast<size_t>(kHcMult * kHidden), 0.6f, 0.1f);
  const std::vector<float> query_scale = MakeRamp(static_cast<size_t>(kHcMult * kHidden), 1.4f, -0.15f);

  std::vector<float> expected(static_cast<size_t>(kRows * kHidden));
  for (int64_t row = 0; row < kRows; ++row) {
    const int64_t g = row % kHcMult;
    const int64_t token = row / kHcMult;
    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    for (int64_t c = 0; c < kHidden; ++c) {
      const float k = key[static_cast<size_t>(row * kHidden + c)];
      const float q = query[static_cast<size_t>(row * kHidden + c)];
      key_sum_sq += k * k;
      query_sum_sq += q * q;
    }
    const float key_inv = 1.0f / std::sqrt(key_sum_sq / static_cast<float>(kHidden) + kEpsilon);
    const float query_inv = 1.0f / std::sqrt(query_sum_sq / static_cast<float>(kHidden) + kEpsilon);
    float dot = 0.0f;
    for (int64_t c = 0; c < kHidden; ++c) {
      const auto scale_index = static_cast<size_t>(g * kHidden + c);
      const float normed_key = key[static_cast<size_t>(row * kHidden + c)] * key_inv * key_scale[scale_index];
      const float normed_query =
          query[static_cast<size_t>(row * kHidden + c)] * query_inv * query_scale[scale_index];
      dot += normed_key * normed_query;
    }
    dot /= std::sqrt(static_cast<float>(kHidden));
    const float gate = Sigmoid(GateArg(dot));
    for (int64_t c = 0; c < kHidden; ++c) {
      expected[static_cast<size_t>(row * kHidden + c)] =
          gate * value[static_cast<size_t>(token * kHidden + c)];
    }
  }

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<T>("key", {kBatch, kSequence, kHcMult, kHidden}, ToTensorType<T>(key));
  test.AddInput<T>("query", {kBatch, kSequence, kHcMult, kHidden}, ToTensorType<T>(query));
  test.AddInput<T>("value", {kBatch, kSequence, kHidden}, ToTensorType<T>(value));
  test.AddInput<T>("key_norm_scale", {kHcMult, kHidden}, ToTensorType<T>(key_scale));
  test.AddInput<T>("query_norm_scale", {kHcMult, kHidden}, ToTensorType<T>(query_scale));
  test.AddOutput<T>("output", {kBatch, kSequence, kHcMult, kHidden}, ToTensorType<T>(expected), false, tolerance,
                    tolerance);
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
                                         const std::vector<T>& vocab_sizes,
                                         int64_t pad_id = kPadId) {
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
      return static_cast<T>(pad_id);
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

// Negative ids and a negative pad_id are the only way to reach two branches that the positive-id
// tests leave dead on every EP: the `result < 0 -> result + mod` correction in PositiveMod, and the
// sign handling in WrappedMultiply. WGSL's `%` in particular follows C truncation for negative
// operands, which is worth pinning rather than assuming.
template <typename T>
void RunNGramHashMappingNegativeIdsTest() {
  constexpr int64_t kNegativePadId = -4;
  const std::vector<T> ids{-5, 7, -3, 2};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  const std::vector<T> expected =
      NGramHashMappingReference<T>(ids, {}, multipliers, vocab_sizes, kNegativePadId);
  // Pins the reference, and the values themselves: every entry is a positive residue even though
  // most of the underlying mixes are negative, which is exactly the PositiveMod correction.
  ASSERT_EQ(expected, (std::vector<T>{5, 5, 36, 38,
                                      87, 89, 78, 78,
                                      78, 82, 47, 47,
                                      52, 54, 35, 37}));
  for (const T value : expected) {
    ASSERT_GE(value, 0);
  }

  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kNegativePadId);
  test.AddInput<T>("input_ids", {1, 4}, ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {1, 4, 4}, expected);
  test.AddOutput<T>("present_ids", {1, 2}, {ids[2], ids[3]});
  test.Run();
}

// A non-positive head vocabulary size has no meaningful modulo. The CPU kernel rejects it rather
// than silently emitting a constant hash id of 0 for that head.
template <typename T>
void RunNGramHashMappingNonPositiveVocabTest() {
  const std::vector<T> ids{3, 4, 5, 6};
  const std::vector<T> multipliers{11, 13, 17};
  // Head 2 is invalid; the other three are the usual primes.
  const std::vector<T> vocab_sizes{101, 103, 0, 109};

  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {1, 4}, ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {1, 4, 4}, std::vector<T>(16, T{0}));
  test.AddOutput<T>("present_ids", {1, 2}, {ids[2], ids[3]});
  // The validation is CPU-only by design: on GPU EPs vocab_sizes lives on the device and checking it
  // would force a synchronization on every Compute call.
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "vocab_sizes must be positive", {}, nullptr,
           &execution_providers);
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

// A decode step must hash the same n-gram window as the corresponding position of a full-sequence
// run. Without past_ids the preceding tokens would silently fall back to pad_id.
template <typename T>
void RunNGramHashMappingChunkedTest() {
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

// An empty input_ids tensor must still thread history through present_ids unchanged. This is the
// only case that reaches the WebGPU kernel's sequence_length == 0 specialization, which drops the
// input_ids binding entirely because WebGPU rejects zero-sized storage bindings.
template <typename T>
void RunNGramHashMappingEmptySequenceTest() {
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  const std::vector<T> past{3, 4};

  auto run = [&](bool with_past) {
    OpTester test("NGramHashMapping", 1, kMSDomain);
    test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
    test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
    test.AddAttribute<int64_t>("pad_id", kPadId);
    test.AddInput<T>("input_ids", {1, 0}, {});
    test.AddInput<T>("multipliers", {3}, multipliers);
    test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
    if (with_past) {
      test.AddInput<T>("past_ids", {1, 2}, past);
    } else {
      test.AddOptionalInputEdge<T>();
    }
    test.AddOutput<T>("hash_ids", {1, 0, 4}, {});
    // With no new tokens the window is unchanged; without a past_ids it is all pad_id.
    test.AddOutput<T>("present_ids", {1, 2},
                      with_past ? past : std::vector<T>{static_cast<T>(kPadId), static_cast<T>(kPadId)});
    test.Run();
  };

  run(/*with_past=*/true);
  run(/*with_past=*/false);
}

}  // namespace

TEST(EngramOpsTest, NGramHashMappingEmptySequenceInt64) {
  RunNGramHashMappingEmptySequenceTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so this is what covers its zero-length
// specialization.
TEST(EngramOpsTest, NGramHashMappingEmptySequenceInt32) {
  RunNGramHashMappingEmptySequenceTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingInt64) {
  RunNGramHashMappingTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so it must be covered explicitly.
TEST(EngramOpsTest, NGramHashMappingInt32) {
  RunNGramHashMappingTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingChunkedMatchesFullSequenceInt64) {
  RunNGramHashMappingChunkedTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so this is the case that gives the WebGPU
// past_ids/present_ids shaders any execution coverage at all.
TEST(EngramOpsTest, NGramHashMappingChunkedMatchesFullSequenceInt32) {
  RunNGramHashMappingChunkedTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingNegativeIdsInt64) {
  RunNGramHashMappingNegativeIdsTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingNegativeIdsInt32) {
  RunNGramHashMappingNegativeIdsTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingRejectsNonPositiveVocabSizeInt64) {
  RunNGramHashMappingNonPositiveVocabTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingRejectsNonPositiveVocabSizeInt32) {
  RunNGramHashMappingNonPositiveVocabTest<int32_t>();
}

TEST(EngramOpsTest, EngramGateFloat) {
  RunEngramGateTest<float>(1e-4f);
}

TEST(EngramOpsTest, EngramGateFloat16) {
  RunEngramGateTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, EngramGateVectorizedFloat) {
  RunEngramGateVectorizedTest<float>(1e-4f);
}

TEST(EngramOpsTest, EngramGateVectorizedFloat16) {
  RunEngramGateVectorizedTest<MLFloat16>(3e-3f);
}

TEST(EngramOpsTest, EngramGateBFloat16) {
  RunEngramGateTest<BFloat16>(2e-2f);
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
