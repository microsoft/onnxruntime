// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include <gsl/gsl>
#include "contrib_ops/cpu/transformers/logits_processor.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {
namespace test {

namespace {

// Minimal ISequences stub reporting a fixed sequence length. MinLengthLogitsProcessor only reads
// GetSequenceLength(); the device/sequence accessors are unused on this path and return empty spans.
class FixedLengthSequences : public ISequences {
 public:
  explicit FixedLengthSequences(int sequence_length) : sequence_length_(sequence_length) {}

  gsl::span<const int32_t> GetSequence(int /*beam_index*/) const override { return {}; }
  gsl::span<const int32_t> GetCurrentDeviceSequences() const override { return {}; }
  gsl::span<int32_t> GetNextDeviceSequences() override { return {}; }
  int GetSequenceLength() const override { return sequence_length_; }
  int GetMaxLength() const override { return sequence_length_; }

 private:
  int sequence_length_;
};

constexpr int kBatchBeamSize = 2;
constexpr int kVocabSize = 4;

}  // namespace

// A negative eos_token_id is the "no eos" sentinel used by greedy/sampling models. The processor
// must be a no-op and must never index scores with a negative token id.
TEST(MinLengthLogitsProcessorTest, NegativeEosTokenIdIsNoOp) {
  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  NextTokenScores<float> next_token_scores{scores_span, kBatchBeamSize, kVocabSize};

  FixedLengthSequences sequences(/*sequence_length=*/1);  // below min_length: eos would be demoted
  MinLengthLogitsProcessor<float> processor(/*min_length=*/5, /*eos_token_id=*/-1);
  processor.Process(&sequences, next_token_scores);

  for (float value : scores) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

// With a valid eos_token_id and a sequence shorter than min_length, the eos score is demoted so
// generation cannot stop early. This locks in the enforcement behavior (no regression).
TEST(MinLengthLogitsProcessorTest, ValidEosTokenIdBelowMinLengthDemotesEos) {
  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  NextTokenScores<float> next_token_scores{scores_span, kBatchBeamSize, kVocabSize};

  constexpr int kEosTokenId = 2;
  FixedLengthSequences sequences(/*sequence_length=*/1);
  MinLengthLogitsProcessor<float> processor(/*min_length=*/5, kEosTokenId);
  processor.Process(&sequences, next_token_scores);

  const float lowest = std::numeric_limits<float>::lowest();
  for (int beam = 0; beam < kBatchBeamSize; ++beam) {
    for (int token = 0; token < kVocabSize; ++token) {
      const float value = scores[static_cast<size_t>(beam) * kVocabSize + token];
      if (token == kEosTokenId) {
        EXPECT_FLOAT_EQ(value, lowest);
      } else {
        EXPECT_FLOAT_EQ(value, 1.0f);
      }
    }
  }
}

// Once the sequence reaches min_length, the eos score is left untouched even for a valid eos id.
TEST(MinLengthLogitsProcessorTest, ValidEosTokenIdAtMinLengthLeavesScoresUnchanged) {
  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  NextTokenScores<float> next_token_scores{scores_span, kBatchBeamSize, kVocabSize};

  FixedLengthSequences sequences(/*sequence_length=*/5);  // == min_length: no demotion expected
  MinLengthLogitsProcessor<float> processor(/*min_length=*/5, /*eos_token_id=*/2);
  processor.Process(&sequences, next_token_scores);

  for (float value : scores) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

}  // namespace test
}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
