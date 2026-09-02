// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <memory>

#include "gtest/gtest.h"
#include "core/framework/execution_provider.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_environment.h"

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

// Deterministic values bounded to [-1, 1]. Deterministic inputs keep the expectations in the
// vectorized tests reproducible, and the bound matters because a plain ramp would grow without
// limit over the hundreds of channels the multi-iteration reduction test needs, saturating the gate
// and hiding exactly the accumulation errors that case is meant to catch.
std::vector<float> MakeWave(size_t count, float phase, float step) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::sin(phase + step * static_cast<float>(i));
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

// Exercises hc_mult > 1 and non-unit norm scales for an arbitrary hidden_size. hidden_size == 4
// selects the WebGPU vec4 component path through the gate reduction and the broadcast pass, and
// hc_mult > 1 makes a per-row rather than per-token scale lookup observable.
//
// Both GPU reductions stride over hidden_size (CUDA by blockDim.x == 256, WGSL by the workgroup size
// 64 over hidden_size / components), so only a hidden_size above those strides takes a second
// iteration and actually accumulates into the per-thread partials.
template <typename T>
void RunEngramGateVectorizedTest(float tolerance, int64_t hidden) {
  if (!IsTypeSupported<T>()) {
    GTEST_SKIP() << "No execution provider available for this type";
  }
  constexpr int64_t kBatch = 1;
  constexpr int64_t kSequence = 2;
  constexpr int64_t kHcMult = 2;
  const int64_t rows = kBatch * kSequence * kHcMult;

  const std::vector<float> key = MakeWave(static_cast<size_t>(rows * hidden), -0.9f, 0.3f);
  const std::vector<float> query = MakeWave(static_cast<size_t>(rows * hidden), 1.2f, -0.25f);
  const std::vector<float> value = MakeWave(static_cast<size_t>(kBatch * kSequence * hidden), 0.4f, 0.35f);
  const std::vector<float> key_scale = MakeWave(static_cast<size_t>(kHcMult * hidden), 0.6f, 0.1f);
  const std::vector<float> query_scale = MakeWave(static_cast<size_t>(kHcMult * hidden), 1.4f, -0.15f);

  std::vector<float> expected(static_cast<size_t>(rows * hidden));
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t g = row % kHcMult;
    const int64_t token = row / kHcMult;
    float key_sum_sq = 0.0f;
    float query_sum_sq = 0.0f;
    for (int64_t c = 0; c < hidden; ++c) {
      const float k = key[static_cast<size_t>(row * hidden + c)];
      const float q = query[static_cast<size_t>(row * hidden + c)];
      key_sum_sq += k * k;
      query_sum_sq += q * q;
    }
    const float key_inv = 1.0f / std::sqrt(key_sum_sq / static_cast<float>(hidden) + kEpsilon);
    const float query_inv = 1.0f / std::sqrt(query_sum_sq / static_cast<float>(hidden) + kEpsilon);
    float dot = 0.0f;
    for (int64_t c = 0; c < hidden; ++c) {
      const auto scale_index = static_cast<size_t>(g * hidden + c);
      const float normed_key = key[static_cast<size_t>(row * hidden + c)] * key_inv * key_scale[scale_index];
      const float normed_query =
          query[static_cast<size_t>(row * hidden + c)] * query_inv * query_scale[scale_index];
      dot += normed_key * normed_query;
    }
    dot /= std::sqrt(static_cast<float>(hidden));
    const float gate = Sigmoid(GateArg(dot));
    for (int64_t c = 0; c < hidden; ++c) {
      expected[static_cast<size_t>(row * hidden + c)] = gate * value[static_cast<size_t>(token * hidden + c)];
    }
  }

  OpTester test("EngramGate", 1, kMSDomain);
  test.AddAttribute<float>("epsilon", kEpsilon);
  test.AddInput<T>("key", {kBatch, kSequence, kHcMult, hidden}, ToTensorType<T>(key));
  test.AddInput<T>("query", {kBatch, kSequence, kHcMult, hidden}, ToTensorType<T>(query));
  test.AddInput<T>("value", {kBatch, kSequence, hidden}, ToTensorType<T>(value));
  test.AddInput<T>("key_norm_scale", {kHcMult, hidden}, ToTensorType<T>(key_scale));
  test.AddInput<T>("query_norm_scale", {kHcMult, hidden}, ToTensorType<T>(query_scale));
  test.AddOutput<T>("output", {kBatch, kSequence, kHcMult, hidden}, ToTensorType<T>(expected), false, tolerance,
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

// Batch strides are only observable when batch_size > 1: with a single row every `b * stride` term
// is zero, so a wrong stride in the hash kernel or in the present-state walk is invisible.
template <typename T>
void RunNGramHashMappingBatchedTest() {
  constexpr int64_t kBatch = 3;
  constexpr int64_t kSequence = 4;
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  // Distinct per-row values so a row picked up from the wrong batch offset changes the result.
  const std::vector<std::vector<T>> rows{{3, 4, 5, 6}, {17, 2, 31, 8}, {40, 41, 42, 43}};
  const std::vector<std::vector<T>> past{{1, 2}, {19, 23}, {29, 37}};

  std::vector<T> ids;
  std::vector<T> past_ids;
  std::vector<T> expected_hash;
  std::vector<T> expected_present;
  for (int64_t b = 0; b < kBatch; ++b) {
    const auto& row = rows[static_cast<size_t>(b)];
    const auto& history = past[static_cast<size_t>(b)];
    const std::vector<T> row_hash = NGramHashMappingReference<T>(row, history, multipliers, vocab_sizes);
    ids.insert(ids.end(), row.begin(), row.end());
    past_ids.insert(past_ids.end(), history.begin(), history.end());
    expected_hash.insert(expected_hash.end(), row_hash.begin(), row_hash.end());
    expected_present.insert(expected_present.end(), row.end() - (kMaxNGramSize - 1), row.end());
  }

  OpTester test("NGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {kBatch, kSequence}, ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddInput<T>("past_ids", {kBatch, kMaxNGramSize - 1}, past_ids);
  test.AddOutput<T>("hash_ids", {kBatch, kSequence, (kMaxNGramSize - 1) * kHeadsPerNGram}, expected_hash);
  test.AddOutput<T>("present_ids", {kBatch, kMaxNGramSize - 1}, expected_present);
  test.Run();
}

// `MayInplace(3, 1)` is only a hint: the allocation planner honors it when past_ids is an
// intermediate whose last consumer is the node, and never for a graph input or a graph output.
// OpTester always feeds past_ids as a graph input, so the barrier-separated read/write design in the
// CUDA and WebGPU present-state kernels is unreachable from those tests. Chaining three decode steps
// makes the middle node's past_ids an intermediate with a single consumer, which is exactly the
// shape the planner aliases -- so the middle node runs with past_ids and present_ids on one buffer.
template <typename T>
void RunNGramHashMappingInPlaceTest(std::unique_ptr<IExecutionProvider> ep) {
  constexpr int64_t kBatch = 2;
  constexpr int64_t kStateLength = kMaxNGramSize - 1;
  constexpr int64_t kNumHeads = kStateLength * kHeadsPerNGram;
  constexpr size_t kSteps = 3;

  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  // Per batch row: the initial window, then one new token per decode step.
  const std::vector<std::vector<T>> initial_past{{1, 2}, {19, 23}};
  const std::vector<std::vector<T>> step_tokens{{3, 31}, {4, 37}, {5, 41}};

  std::vector<T> past_ids_feed;
  for (const auto& row : initial_past) {
    past_ids_feed.insert(past_ids_feed.end(), row.begin(), row.end());
  }

  // Reference: replay the decode loop on the host, one batch row at a time.
  std::vector<std::vector<T>> history = initial_past;
  std::vector<std::vector<T>> expected_hash(kSteps);
  std::vector<T> expected_final_present;
  for (size_t step = 0; step < kSteps; ++step) {
    for (int64_t b = 0; b < kBatch; ++b) {
      const std::vector<T> chunk{step_tokens[step][static_cast<size_t>(b)]};
      const std::vector<T> row_hash =
          NGramHashMappingReference<T>(chunk, history[static_cast<size_t>(b)], multipliers, vocab_sizes);
      expected_hash[step].insert(expected_hash[step].end(), row_hash.begin(), row_hash.end());
      auto& row_history = history[static_cast<size_t>(b)];
      row_history.erase(row_history.begin());
      row_history.push_back(chunk[0]);
    }
  }
  for (const auto& row : history) {
    expected_final_present.insert(expected_final_present.end(), row.begin(), row.end());
  }

  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 17}, {kMSDomain, 1}};
  Model model("NGramHashMappingInPlace", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  NodeAttributes attributes;
  attributes["max_ngram_size"] = utils::MakeAttribute(std::string("max_ngram_size"), kMaxNGramSize);
  attributes["n_head_per_ngram"] = utils::MakeAttribute(std::string("n_head_per_ngram"), kHeadsPerNGram);
  attributes["pad_id"] = utils::MakeAttribute(std::string("pad_id"), kPadId);

  auto* multipliers_arg = builder.MakeInput<T>({kMaxNGramSize}, multipliers);
  auto* vocab_sizes_arg = builder.MakeInput<T>({kNumHeads}, vocab_sizes);
  NodeArg* past_arg = builder.MakeInput<T>({kBatch, kStateLength}, past_ids_feed);

  std::vector<std::string> present_names;
  for (size_t step = 0; step < kSteps; ++step) {
    auto* input_ids_arg = builder.MakeInput<T>({kBatch, 1}, step_tokens[step]);
    auto* hash_arg = builder.MakeOutput();
    // Only the last step's present_ids is a graph output; the planner refuses to alias those.
    const bool is_last = step + 1 == kSteps;
    auto* present_arg = is_last ? builder.MakeOutput() : builder.MakeIntermediate();
    builder.AddNode("NGramHashMapping", {input_ids_arg, multipliers_arg, vocab_sizes_arg, past_arg},
                    {hash_arg, present_arg}, kMSDomain, &attributes);
    present_names.push_back(present_arg->Name());
    past_arg = present_arg;
  }
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  SessionOptions session_options;
  InferenceSessionWrapper session{session_options, GetEnvironment()};
  if (ep != nullptr) {
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  }
  std::istringstream model_istream(model_data);
  ASSERT_STATUS_OK(session.Load(model_istream));
  ASSERT_STATUS_OK(session.Initialize());

  // Pin the aliasing itself: without this the test would silently degrade to a plain chained run if
  // the hint were dropped or the planner changed.
  const SessionState& session_state = session.GetSessionState();
  int middle_present = -1;
  int middle_past = -1;
  ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx(present_names[1], middle_present));
  ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx(present_names[0], middle_past));
  const auto& alloc_plan = session_state.GetPerValueAllocPlan();
  EXPECT_EQ(alloc_plan[static_cast<size_t>(middle_present)].alloc_kind, AllocKind::kReuse);
  EXPECT_EQ(alloc_plan[static_cast<size_t>(middle_present)].reused_buffer, middle_past);

  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session.Run(RunOptions{}, builder.feeds_, builder.output_names_, &fetches));

  // MakeOutput() appends to output_names_ in creation order, so the fetches are hash_0, hash_1,
  // hash_2, then the final present_ids.
  ASSERT_EQ(fetches.size(), kSteps + 1);
  for (size_t step = 0; step < kSteps; ++step) {
    const Tensor& hash = fetches[step].Get<Tensor>();
    ASSERT_EQ(hash.Shape(), TensorShape({kBatch, 1, kNumHeads}));
    const auto span = hash.DataAsSpan<T>();
    EXPECT_EQ(std::vector<T>(span.begin(), span.end()), expected_hash[step]) << "step " << step;
  }
  const Tensor& present = fetches[kSteps].Get<Tensor>();
  ASSERT_EQ(present.Shape(), TensorShape({kBatch, kStateLength}));
  const auto present_span = present.DataAsSpan<T>();
  EXPECT_EQ(std::vector<T>(present_span.begin(), present_span.end()), expected_final_present);
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

TEST(EngramOpsTest, NGramHashMappingBatchedInt64) {
  RunNGramHashMappingBatchedTest<int64_t>();
}

TEST(EngramOpsTest, NGramHashMappingBatchedInt32) {
  RunNGramHashMappingBatchedTest<int32_t>();
}

TEST(EngramOpsTest, NGramHashMappingInPlaceCpu) {
  RunNGramHashMappingInPlaceTest<int64_t>(nullptr);
  RunNGramHashMappingInPlaceTest<int32_t>(nullptr);
}

#ifdef USE_CUDA
TEST(EngramOpsTest, NGramHashMappingInPlaceCuda) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA execution provider is not available";
  }
  RunNGramHashMappingInPlaceTest<int64_t>(DefaultCudaExecutionProvider());
  RunNGramHashMappingInPlaceTest<int32_t>(DefaultCudaExecutionProvider());
}
#endif

#ifdef USE_WEBGPU
// int32 is the only type the WebGPU kernel registers.
TEST(EngramOpsTest, NGramHashMappingInPlaceWebGpu) {
  // A null provider is the CPU convention in RunNGramHashMappingInPlaceTest, so fail closed here
  // instead of silently degrading into a CPU run that never exercises NGramPresentIdsProgram.
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  RunNGramHashMappingInPlaceTest<int32_t>(std::move(webgpu_ep));
}
#endif

TEST(EngramOpsTest, EngramGateFloat) {
  RunEngramGateTest<float>(1e-4f);
}

TEST(EngramOpsTest, EngramGateFloat16) {
  RunEngramGateTest<MLFloat16>(2e-3f);
}

TEST(EngramOpsTest, EngramGateVectorizedFloat) {
  RunEngramGateVectorizedTest<float>(1e-4f, 4);
}

TEST(EngramOpsTest, EngramGateVectorizedFloat16) {
  RunEngramGateVectorizedTest<MLFloat16>(3e-3f, 4);
}

// 260 channels is the smallest multiple of 4 above both accumulation strides (CUDA's blockDim.x of
// 256 and, with components == 4, the WGSL workgroup size of 64 over hidden_size / 4 == 65), so this
// is the only case where either strided reduction loop runs more than once per thread.
TEST(EngramOpsTest, EngramGateMultiIterationReductionFloat) {
  RunEngramGateVectorizedTest<float>(1e-4f, 260);
}

TEST(EngramOpsTest, EngramGateMultiIterationReductionFloat16) {
  RunEngramGateVectorizedTest<MLFloat16>(5e-3f, 260);
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
