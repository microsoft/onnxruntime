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

// Minimal ISequences stub reporting a fixed sequence length. The MinLength path only reads
// GetSequenceLength(); the device/sequence accessors are unused here and return empty spans.
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

// Builds a minimal GreedySearchParameters that activates only the MinLength processor path, so the
// list-level tests exercise the MinLength construction condition without other processors interfering.
GreedySearchParameters MakeMinLengthOnlyParameters(int min_length, int eos_token_id) {
  GreedySearchParameters parameters{};
  parameters.model_type = IGenerationParameters::kModelTypeGpt;
  parameters.logits_processor = 0;
  parameters.eos_token_id = eos_token_id;
  parameters.min_length = min_length;
  parameters.no_repeat_ngram_size = 0;
  parameters.repetition_penalty = 1.0f;    // 1.0 means no penalty, so that processor is skipped
  parameters.temperature = 0.0f;           // <= 0 skips the temperature processor
  parameters.batch_size = kBatchBeamSize;  // GreedySearchParameters::BatchBeamSize() == batch_size
  parameters.vocab_size = kVocabSize;
  return parameters;
}

}  // namespace

// Backstop: SetScore ignores a negative token id, which is the "no eos" sentinel. This guards the
// runtime path against indexing scores with a negative token id even if a processor is reached.
TEST(MinLengthLogitsProcessorTest, SetScoreIgnoresNegativeTokenId) {
  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  NextTokenScores<float> next_token_scores{scores_span, kBatchBeamSize, kVocabSize};

  next_token_scores.SetScore(/*token_id=*/-1, std::numeric_limits<float>::lowest());

  for (float value : scores) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

// With a negative "no eos" sentinel there is no token to demote, so the Init guard skips constructing
// the MinLength processor as a guaranteed no-op (a defensive/performance skip). The scores would be
// unchanged here regardless, because SetScore also ignores a negative token id; the enforcement path
// itself is covered by the eos >= 0 positive-control test below.
TEST(MinLengthLogitsProcessorTest, ListInitSkipsProcessorForNegativeEosTokenId) {
  GreedySearchParameters parameters = MakeMinLengthOnlyParameters(/*min_length=*/5, /*eos_token_id=*/-1);
  LogitsProcessorList processor_list;
  processor_list.Init(parameters);

  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  FixedLengthSequences sequences(/*sequence_length=*/1);  // below min_length
  processor_list.Process(&sequences, scores_span, /*step=*/1);

  for (float value : scores) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

// Enforcement path: with a valid eos_token_id the Init call site adds the MinLength processor, so a
// below-min-length run demotes the eos score. This is the discriminating test for the eos >= 0 branch
// of the guard and for the min-length enforcement behavior.
TEST(MinLengthLogitsProcessorTest, ListInitDemotesEosBelowMinLength) {
  constexpr int kEosTokenId = 2;
  GreedySearchParameters parameters = MakeMinLengthOnlyParameters(/*min_length=*/5, kEosTokenId);
  LogitsProcessorList processor_list;
  processor_list.Init(parameters);

  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  FixedLengthSequences sequences(/*sequence_length=*/1);  // below min_length
  processor_list.Process(&sequences, scores_span, /*step=*/1);

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

// Once the sequence reaches min_length, a valid eos score is left untouched: min_length is no longer
// enforced. This locks in the boundary behavior and confirms the guard change is limited to the
// negative-sentinel case (no regression for valid eos ids).
TEST(MinLengthLogitsProcessorTest, ListInitLeavesScoresUnchangedAtMinLength) {
  GreedySearchParameters parameters = MakeMinLengthOnlyParameters(/*min_length=*/5, /*eos_token_id=*/2);
  LogitsProcessorList processor_list;
  processor_list.Init(parameters);

  std::vector<float> scores(kBatchBeamSize * kVocabSize, 1.0f);
  gsl::span<float> scores_span(scores);
  FixedLengthSequences sequences(/*sequence_length=*/5);  // == min_length: no demotion expected
  processor_list.Process(&sequences, scores_span, /*step=*/1);

  for (float value : scores) {
    EXPECT_FLOAT_EQ(value, 1.0f);
  }
}

}  // namespace test
}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
