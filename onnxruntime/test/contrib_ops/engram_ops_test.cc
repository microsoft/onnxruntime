// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/execution_provider.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

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

// ---------------------------------------------------------------------------------------------
// VarlenNGramHashMapping
// ---------------------------------------------------------------------------------------------

// Builds a cumulative_sequence_length tensor (batch_size + 1 offsets) from the per-request token
// counts, matching the VarlenCausalConvWithState convention this op follows.
template <typename T>
std::vector<int32_t> CuSeqLensFrom(const std::vector<std::vector<T>>& sequences) {
  std::vector<int32_t> cu_seqlens{0};
  int32_t total = 0;
  for (const auto& seq : sequences) {
    total += static_cast<int32_t>(seq.size());
    cu_seqlens.push_back(total);
  }
  return cu_seqlens;
}

// The n-gram hash ids for a packed batch must equal running NGramHashMapping once per request and
// concatenating token-major, because the varlen op is required to clamp its window at each
// request's own boundary instead of the flat buffer's.
template <typename T>
std::vector<T> VarlenNGramHashMappingReference(const std::vector<std::vector<T>>& sequences,
                                               const std::vector<std::vector<T>>& histories,
                                               const std::vector<T>& multipliers,
                                               const std::vector<T>& vocab_sizes,
                                               int64_t pad_id = kPadId) {
  std::vector<T> output;
  for (size_t b = 0; b < sequences.size(); ++b) {
    const std::vector<T> history = histories.empty() ? std::vector<T>{} : histories[b];
    const std::vector<T> per_request =
        NGramHashMappingReference<T>(sequences[b], history, multipliers, vocab_sizes, pad_id);
    output.insert(output.end(), per_request.begin(), per_request.end());
  }
  return output;
}

// present_ids for a single request: the right-aligned trailing window of (history ++ ids), padded
// with pad_id for positions before the start of the whole (unpacked) sequence.
template <typename T>
std::vector<T> PresentIdsReference(const std::vector<T>& ids, const std::vector<T>& history,
                                   int64_t pad_id = kPadId) {
  constexpr int64_t state_length = kMaxNGramSize - 1;
  const int64_t local_length = static_cast<int64_t>(ids.size());
  std::vector<T> present(static_cast<size_t>(state_length));
  for (int64_t j = 0; j < state_length; ++j) {
    const int64_t source_t = local_length - state_length + j;
    if (source_t >= 0) {
      present[static_cast<size_t>(j)] = ids[static_cast<size_t>(source_t)];
      continue;
    }
    const int64_t slot = state_length + source_t;
    present[static_cast<size_t>(j)] =
        (!history.empty() && slot >= 0 && slot < state_length) ? history[static_cast<size_t>(slot)]
                                                                : static_cast<T>(pad_id);
  }
  return present;
}

// Equivalence between one VarlenNGramHashMapping call over a packed multi-sequence buffer and
// running NGramHashMapping separately per sequence and concatenating the results.
template <typename T>
void RunVarlenNGramHashMappingTest() {
  const std::vector<std::vector<T>> sequences{{3, 4, 5}, {7, 8}, {1}};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};

  std::vector<T> flat_ids;
  for (const auto& seq : sequences) {
    flat_ids.insert(flat_ids.end(), seq.begin(), seq.end());
  }
  const std::vector<int32_t> cu_seqlens = CuSeqLensFrom(sequences);
  const int64_t total_tokens = static_cast<int64_t>(flat_ids.size());
  const int64_t batch_size = static_cast<int64_t>(sequences.size());

  const std::vector<T> expected_hash =
      VarlenNGramHashMappingReference<T>(sequences, {}, multipliers, vocab_sizes);
  std::vector<T> expected_present;
  for (const auto& seq : sequences) {
    const std::vector<T> present = PresentIdsReference<T>(seq, {});
    expected_present.insert(expected_present.end(), present.begin(), present.end());
  }

  OpTester test("VarlenNGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {total_tokens}, flat_ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddInput<int32_t>("cumulative_sequence_length", {batch_size + 1}, cu_seqlens);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {total_tokens, 4}, expected_hash);
  test.AddOutput<T>("present_ids", {batch_size, 2}, expected_present);
  test.Run();
}

// Pins that the packed op never reads across a sequence boundary: the reference for the packed
// buffer must actually disagree with what a naive (batch_size=1) reshape of the same flat buffer
// would produce at the boundary token, otherwise this test would pass vacuously.
template <typename T>
void RunVarlenNGramHashMappingBoundaryTest() {
  const std::vector<std::vector<T>> sequences{{100, 101, 102}, {5, 6}};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};

  std::vector<T> flat_ids;
  for (const auto& seq : sequences) {
    flat_ids.insert(flat_ids.end(), seq.begin(), seq.end());
  }
  const std::vector<int32_t> cu_seqlens = CuSeqLensFrom(sequences);
  const int64_t total_tokens = static_cast<int64_t>(flat_ids.size());
  const int64_t batch_size = static_cast<int64_t>(sequences.size());
  constexpr int64_t num_heads = (kMaxNGramSize - 1) * kHeadsPerNGram;

  const std::vector<T> expected_hash =
      VarlenNGramHashMappingReference<T>(sequences, {}, multipliers, vocab_sizes);
  const std::vector<T> naive_flattened = NGramHashMappingReference<T>(flat_ids, {}, multipliers, vocab_sizes);

  const int64_t boundary_token = static_cast<int64_t>(sequences[0].size());  // first token of sequence 1
  bool differs_at_boundary = false;
  for (int64_t h = 0; h < num_heads; ++h) {
    if (expected_hash[static_cast<size_t>(boundary_token * num_heads + h)] !=
        naive_flattened[static_cast<size_t>(boundary_token * num_heads + h)]) {
      differs_at_boundary = true;
      break;
    }
  }
  ASSERT_TRUE(differs_at_boundary) << "test fixture does not exercise the sequence-boundary clamp";

  std::vector<T> expected_present;
  for (const auto& seq : sequences) {
    const std::vector<T> present = PresentIdsReference<T>(seq, {});
    expected_present.insert(expected_present.end(), present.begin(), present.end());
  }

  OpTester test("VarlenNGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {total_tokens}, flat_ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddInput<int32_t>("cumulative_sequence_length", {batch_size + 1}, cu_seqlens);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {total_tokens, 4}, expected_hash);
  test.AddOutput<T>("present_ids", {batch_size, 2}, expected_present);
  test.Run();
}

// Negative ids exercise the same PositiveMod/WrappedMultiply corrections as the non-varlen
// negative-id test, across two concurrently packed requests.
template <typename T>
void RunVarlenNGramHashMappingNegativeIdsTest() {
  constexpr int64_t kNegativePadId = -4;
  const std::vector<std::vector<T>> sequences{{-5, 7, -3, 2}, {-1, 6}};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};

  std::vector<T> flat_ids;
  for (const auto& seq : sequences) {
    flat_ids.insert(flat_ids.end(), seq.begin(), seq.end());
  }
  const std::vector<int32_t> cu_seqlens = CuSeqLensFrom(sequences);
  const int64_t total_tokens = static_cast<int64_t>(flat_ids.size());
  const int64_t batch_size = static_cast<int64_t>(sequences.size());

  const std::vector<T> expected_hash =
      VarlenNGramHashMappingReference<T>(sequences, {}, multipliers, vocab_sizes, kNegativePadId);
  for (const T value : expected_hash) {
    ASSERT_GE(value, 0);
  }
  std::vector<T> expected_present;
  for (const auto& seq : sequences) {
    const std::vector<T> present = PresentIdsReference<T>(seq, {}, kNegativePadId);
    expected_present.insert(expected_present.end(), present.begin(), present.end());
  }

  OpTester test("VarlenNGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kNegativePadId);
  test.AddInput<T>("input_ids", {total_tokens}, flat_ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddInput<int32_t>("cumulative_sequence_length", {batch_size + 1}, cu_seqlens);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {total_tokens, 4}, expected_hash);
  test.AddOutput<T>("present_ids", {batch_size, 2}, expected_present);
  test.Run();
}

// Same rejection as NGramHashMapping's non-positive vocab_sizes test; validation is CPU-only.
template <typename T>
void RunVarlenNGramHashMappingNonPositiveVocabTest() {
  const std::vector<T> ids{3, 4, 5, 6};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 0, 109};
  const std::vector<int32_t> cu_seqlens{0, 4};

  OpTester test("VarlenNGramHashMapping", 1, kMSDomain);
  test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
  test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
  test.AddAttribute<int64_t>("pad_id", kPadId);
  test.AddInput<T>("input_ids", {4}, ids);
  test.AddInput<T>("multipliers", {3}, multipliers);
  test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
  test.AddInput<int32_t>("cumulative_sequence_length", {2}, cu_seqlens);
  test.AddOptionalInputEdge<T>();
  test.AddOutput<T>("hash_ids", {4, 4}, std::vector<T>(16, T{0}));
  test.AddOutput<T>("present_ids", {1, 2}, {ids[2], ids[3]});
  // Same as NGramHashMapping: vocab_sizes lives on the device for GPU EPs, so this check is CPU-only.
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "vocab_sizes must be positive", {}, nullptr,
           &execution_providers);
}

// Packed prefill of two concurrent requests followed by a packed single-token decode step per
// request, threading present_ids independently per request. Analogous to
// RunNGramHashMappingChunkedTest but exercising multiple sequences packed together per call.
template <typename T>
void RunVarlenNGramHashMappingChunkedTest() {
  const std::vector<std::vector<T>> sequences{{3, 4, 5, 6}, {20, 21, 22, 23}};
  const std::vector<T> multipliers{11, 13, 17};
  const std::vector<T> vocab_sizes{101, 103, 107, 109};
  std::vector<std::vector<T>> full_refs;
  for (const auto& seq : sequences) {
    full_refs.push_back(NGramHashMappingReference<T>(seq, {}, multipliers, vocab_sizes));
  }

  auto run_chunk = [&](const std::vector<std::vector<T>>& chunks, const std::vector<std::vector<T>>& pasts,
                       const std::vector<std::vector<T>>& expected_hash_per_request,
                       const std::vector<std::vector<T>>& expected_presents) {
    std::vector<T> flat_ids;
    for (const auto& c : chunks) {
      flat_ids.insert(flat_ids.end(), c.begin(), c.end());
    }
    std::vector<T> expected_hash;
    for (const auto& h : expected_hash_per_request) {
      expected_hash.insert(expected_hash.end(), h.begin(), h.end());
    }
    std::vector<T> expected_present;
    for (const auto& p : expected_presents) {
      expected_present.insert(expected_present.end(), p.begin(), p.end());
    }
    const std::vector<int32_t> cu_seqlens = CuSeqLensFrom(chunks);
    const int64_t total_tokens = static_cast<int64_t>(flat_ids.size());
    const int64_t batch_size = static_cast<int64_t>(chunks.size());

    OpTester test("VarlenNGramHashMapping", 1, kMSDomain);
    test.AddAttribute<int64_t>("max_ngram_size", kMaxNGramSize);
    test.AddAttribute<int64_t>("n_head_per_ngram", kHeadsPerNGram);
    test.AddAttribute<int64_t>("pad_id", kPadId);
    test.AddInput<T>("input_ids", {total_tokens}, flat_ids);
    test.AddInput<T>("multipliers", {3}, multipliers);
    test.AddInput<T>("vocab_sizes", {4}, vocab_sizes);
    test.AddInput<int32_t>("cumulative_sequence_length", {batch_size + 1}, cu_seqlens);
    if (pasts.empty()) {
      test.AddOptionalInputEdge<T>();
    } else {
      std::vector<T> flat_past;
      for (const auto& p : pasts) {
        flat_past.insert(flat_past.end(), p.begin(), p.end());
      }
      test.AddInput<T>("past_ids", {batch_size, 2}, flat_past);
    }
    test.AddOutput<T>("hash_ids", {total_tokens, 4}, expected_hash);
    test.AddOutput<T>("present_ids", {batch_size, 2}, expected_present);
    test.Run();
  };

  // Packed prefill: the first two tokens of each of the two concurrent requests.
  const std::vector<std::vector<T>> prefill{{sequences[0][0], sequences[0][1]},
                                            {sequences[1][0], sequences[1][1]}};
  run_chunk(prefill, {},
            {std::vector<T>(full_refs[0].begin(), full_refs[0].begin() + 8),
             std::vector<T>(full_refs[1].begin(), full_refs[1].begin() + 8)},
            prefill);

  // Packed decode of token index 2 for both requests, using each request's own prefill history.
  const std::vector<std::vector<T>> decode2{{sequences[0][2]}, {sequences[1][2]}};
  const std::vector<std::vector<T>> present_after_decode2{{prefill[0][1], sequences[0][2]},
                                                          {prefill[1][1], sequences[1][2]}};
  run_chunk(decode2, prefill,
            {std::vector<T>(full_refs[0].begin() + 8, full_refs[0].begin() + 12),
             std::vector<T>(full_refs[1].begin() + 8, full_refs[1].begin() + 12)},
            present_after_decode2);

  // Packed decode of token index 3 for both requests, using the history returned above.
  const std::vector<std::vector<T>> decode3{{sequences[0][3]}, {sequences[1][3]}};
  const std::vector<std::vector<T>> present_after_decode3{{sequences[0][2], sequences[0][3]},
                                                          {sequences[1][2], sequences[1][3]}};
  run_chunk(decode3, present_after_decode2,
            {std::vector<T>(full_refs[0].begin() + 12, full_refs[0].end()),
             std::vector<T>(full_refs[1].begin() + 12, full_refs[1].end())},
            present_after_decode3);
}

}  // namespace

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

TEST(EngramOpsTest, VarlenNGramHashMappingMatchesPerSequenceInt64) {
  RunVarlenNGramHashMappingTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so it must be covered explicitly.
TEST(EngramOpsTest, VarlenNGramHashMappingMatchesPerSequenceInt32) {
  RunVarlenNGramHashMappingTest<int32_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingNoCrossSequenceLeakageInt64) {
  RunVarlenNGramHashMappingBoundaryTest<int64_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingNoCrossSequenceLeakageInt32) {
  RunVarlenNGramHashMappingBoundaryTest<int32_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingChunkedMatchesFullSequenceInt64) {
  RunVarlenNGramHashMappingChunkedTest<int64_t>();
}

// int32 is the only type the WebGPU kernel supports, so this is the case that gives the WebGPU
// past_ids/present_ids shaders any execution coverage at all.
TEST(EngramOpsTest, VarlenNGramHashMappingChunkedMatchesFullSequenceInt32) {
  RunVarlenNGramHashMappingChunkedTest<int32_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingNegativeIdsInt64) {
  RunVarlenNGramHashMappingNegativeIdsTest<int64_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingNegativeIdsInt32) {
  RunVarlenNGramHashMappingNegativeIdsTest<int32_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingRejectsNonPositiveVocabSizeInt64) {
  RunVarlenNGramHashMappingNonPositiveVocabTest<int64_t>();
}

TEST(EngramOpsTest, VarlenNGramHashMappingRejectsNonPositiveVocabSizeInt32) {
  RunVarlenNGramHashMappingNonPositiveVocabTest<int32_t>();
}

}  // namespace test
}  // namespace onnxruntime
