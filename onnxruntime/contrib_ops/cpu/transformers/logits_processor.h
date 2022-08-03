// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
struct NextTokenScores {
  gsl::span<T>& scores;
  int batch_beam_size;
  int vocab_size;

  gsl::span<T> GetScores(int batch_beam_index);

  void SetScore(int token_id, T score);
};

// Interface for all scorers for beam search or beam sample.
template <typename T>
class ILogitsProcessor {
 public:
  virtual ~ILogitsProcessor() {}

  virtual void Process(const ISequences* sequences,
                       NextTokenScores<T>& next_token_scores) = 0;
};

template <typename T>
class MinLengthLogitsProcessor : public ILogitsProcessor<T> {
 public:
  MinLengthLogitsProcessor(int min_length, int eos_token_id);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  int min_length_;
  int eos_token_id_;
};

template <typename T>
class RepetitionPenaltyLogitsProcessor : public ILogitsProcessor<T> {
 public:
  RepetitionPenaltyLogitsProcessor(float penalty);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  float penalty_;
};

template <typename T>
class NoRepeatNGramLogitsProcessor : public ILogitsProcessor<T> {
 public:
  NoRepeatNGramLogitsProcessor(int ngram_size);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  int ngram_size_;
};

template <typename T>
class VocabMaskLogitsProcessor : public ILogitsProcessor<T> {
 public:
  VocabMaskLogitsProcessor(const gsl::span<const int32_t>& vocab_mask);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  gsl::span<const int32_t> vocab_mask_;
};

template <typename T>
class PrefixVocabMaskLogitsProcessor : public ILogitsProcessor<T> {
 public:
  PrefixVocabMaskLogitsProcessor(const gsl::span<const int32_t>& vocab_mask, int batch_size);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  gsl::span<const int32_t> prefix_vocab_mask_;
  const int batch_size_;
};

class LogitsProcessorList : public ILogitsProcessorList {
 public:
  LogitsProcessorList() = default;
  void Init(const BeamSearchParameters& parameters);
  void Init(const GreedySearchParameters& parameters);
  void Process(const ISequences* sequences, gsl::span<float>& next_token_scores, int step);

 private:
  template<typename GenerationParametersT>
  void LogitsProcessorInitImpl(const GenerationParametersT& parameters) {
    processor_list_.clear();

    if (parameters.repetition_penalty != 1.0f) {  // 1.0 means no penalty
      repetition_penalty_processor_ = std::make_unique<RepetitionPenaltyLogitsProcessor<float>>(
          parameters.repetition_penalty);
      processor_list_.push_back(repetition_penalty_processor_.get());
    }

    if (parameters.no_repeat_ngram_size > 0) {
      no_repeat_ngram_processor_ = std::make_unique<
                                     NoRepeatNGramLogitsProcessor<float>
                                   >(parameters.no_repeat_ngram_size);
      processor_list_.push_back(no_repeat_ngram_processor_.get());
    }

    if (!parameters.vocab_mask.empty()) {
      vocab_mask_processor_ = std::make_unique<VocabMaskLogitsProcessor<float>>(parameters.vocab_mask);
      processor_list_.push_back(vocab_mask_processor_.get());
    }

    if (!parameters.prefix_vocab_mask.empty()) {
      prefix_vocab_mask_processor_ = std::make_unique<
                                       PrefixVocabMaskLogitsProcessor<float>
                                     >(parameters.prefix_vocab_mask,
                                       parameters.batch_size);
      processor_list_.push_back(prefix_vocab_mask_processor_.get());
    }

    if (parameters.min_length > 0) {
      min_length_processor_ = std::make_unique<MinLengthLogitsProcessor<float>>(parameters.min_length,
                                                                                parameters.eos_token_id);
      processor_list_.push_back(min_length_processor_.get());
    }

    batch_beam_size_ = parameters.BatchBeamSize();
    vocab_size_ = parameters.vocab_size;
  }

  int batch_beam_size_;
  int vocab_size_;
  InlinedVector<ILogitsProcessor<float>*> processor_list_;

  std::unique_ptr<RepetitionPenaltyLogitsProcessor<float>> repetition_penalty_processor_;
  std::unique_ptr<NoRepeatNGramLogitsProcessor<float>> no_repeat_ngram_processor_;
  std::unique_ptr<VocabMaskLogitsProcessor<float>> vocab_mask_processor_;
  std::unique_ptr<PrefixVocabMaskLogitsProcessor<float>> prefix_vocab_mask_processor_;
  std::unique_ptr<MinLengthLogitsProcessor<float>> min_length_processor_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
