// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/beam_search_parameters.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"
#include "contrib_ops/cpu/transformers/sampling_parameters.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
struct NextTokenScores {
  gsl::span<T>& scores;
  int batch_beam_size;
  int vocab_size;

  gsl::span<T> GetScores(int batch_beam_index) {
    assert(batch_beam_index >= 0 && batch_beam_index < batch_beam_size);
    return scores.subspan(static_cast<gsl::index>(batch_beam_index) * vocab_size, vocab_size);
  }

  void SetScore(int token_id, T score) {
    assert(token_id >= 0 && token_id < vocab_size);
    for (int i = 0; i < batch_beam_size; i++) {
      scores[static_cast<gsl::index>(i) * vocab_size + token_id] = score;
    }
  }
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

template <typename T>
class TemperatureLogitsProcessor : public ILogitsProcessor<T> {
 public:
  TemperatureLogitsProcessor(float temperature);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  float temperature_;
};

// template <typename T>
// class TopPLogitsProcessor : public ILogitsProcessor<T> {
//  public:
//   TopPLogitsProcessor(float top_p, float filter_value,
//                       onnxruntime::concurrency::ThreadPool* thread_pool);

//   void Process(const ISequences* sequences,
//                NextTokenScores<T>& next_token_scores) override;

//  private:
//   float top_p_;
//   float filter_value_;
//   onnxruntime::concurrency::ThreadPool* thread_pool_;
// };

template <typename T>
class PresencePenaltyLogitsProcessor : public ILogitsProcessor<T> {
 public:
  PresencePenaltyLogitsProcessor(const gsl::span<const int32_t>& presence_mask,
                                 float presence_penalty);

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override;

 private:
  gsl::span<const int32_t> presence_mask_;
  float presence_penalty_;
};

template <typename T>
class TimestampLogitsProcessor : public ILogitsProcessor<T> {
 public:
  TimestampLogitsProcessor(int eos_token_id, int max_initial_timestamp_index)
      : eos_token_id_(eos_token_id), max_initial_timestamp_index_(max_initial_timestamp_index) {}

  void Process(const ISequences* sequences,
               NextTokenScores<T>& next_token_scores) override {
    // TODO: translate_token_id_ and transcribe_token_id_ need to support both multilingual and English-only models.
    const int beg_token_id_ = eos_token_id_ + 107;
    const int not_token_id_ = eos_token_id_ + 106;
    const int solm_token_id_ = eos_token_id_ + 105;
    const int sot_token_id_ = eos_token_id_ + 1;
    constexpr int translate_token_id_ = 50358;
    constexpr int transcribe_token_id_ = 50359;

    const int batch_beam_size = next_token_scores.batch_beam_size;
    const int vocab_size = next_token_scores.vocab_size;
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
      gsl::span<const int32_t> sequence = sequences->GetSequence(i);
      const size_t seq_length = sequence.size();

      // Find first timestamp
      size_t sample_begin = 0;
      for (size_t j = 0; j < seq_length; j++) {
        sample_begin++;
        if (sequence[j] >= beg_token_id_) {
          break;
        }
      }

      // Suppress tokens
      for (int j = 0; j < vocab_size; j++) {
        // Suppress notimestamps and solm tokens
        if (j == not_token_id_ || j == solm_token_id_) {
          beam_token_scores[j] = std::numeric_limits<T>::lowest();
        }

        // Suppress sot, translate and transcribe tokens
        if (seq_length > sample_begin) {
          if (j == sot_token_id_ || j == translate_token_id_ || j == transcribe_token_id_) {
            beam_token_scores[j] = std::numeric_limits<T>::lowest();
          }
        }
      }

      // Timestamps should be in pair except the first one
      const bool last_was_timestamp = seq_length > 0 && sequence.back() >= beg_token_id_;
      const bool penultimate_was_timestamp = seq_length <= sample_begin || sequence[seq_length - 2] >= beg_token_id_;
      if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
          // If timestamps show up in pair, or it's the first timestamp, no more timestamp is generated
          for (int j = beg_token_id_; j < vocab_size; j++) {
            beam_token_scores[j] = std::numeric_limits<T>::lowest();
          }
        } else {
          // If timestamp doesn't show up in pair, generate timestamp
          for (int j = 0; j < eos_token_id_; j++) {
            beam_token_scores[j] = std::numeric_limits<T>::lowest();
          }
        }
      }

      // Find timestamp tokens
      std::vector<int32_t> timestamps;
      for (const auto& word_id : sequence) {
        if (word_id >= beg_token_id_) {
          timestamps.push_back(word_id);
        }
      }

      // Timestamps will not decrease
      const size_t timestamps_len = timestamps.size();
      if (timestamps_len > 0) {
        int timestamp_last = 0;
        if (last_was_timestamp && !penultimate_was_timestamp) {
          // For single timestamp at the end, next timestamp must not be smaller
          timestamp_last = timestamps.back();
        } else {
          // For paired timestamp at the end, next timestamp must be greater
          timestamp_last = timestamps.back() + 1;
        }

        for (int j = beg_token_id_; j < timestamp_last; j++) {
          beam_token_scores[j] = std::numeric_limits<T>::lowest();
        }
      }

      if (seq_length == sample_begin) {
        const int last_allowed = beg_token_id_ + max_initial_timestamp_index_;
        for (int j = last_allowed + 1; j < vocab_size; j++) {
          beam_token_scores[j] = std::numeric_limits<T>::lowest();
        }
      }

      // Caculate logsumexp on timestamps
      float timestamp_logprob = std::numeric_limits<T>::lowest();
      {
        float logsumexp = 0.0f;
        const float logprob_max = *std::max_element(beam_token_scores.begin() + beg_token_id_, beam_token_scores.end());
        for (int j = beg_token_id_; j < vocab_size; ++j) {
          if (beam_token_scores[j] > std::numeric_limits<T>::lowest()) {
            logsumexp += expf(beam_token_scores[j] - logprob_max);
          }
        }
        if (logsumexp > 0.0f) {
          timestamp_logprob = logf(logsumexp) + logprob_max;
        }
      }

      const float max_text_token_logprob = *std::max_element(beam_token_scores.begin(), beam_token_scores.begin() + beg_token_id_);
      if (timestamp_logprob > max_text_token_logprob) {
        for (int j = 0; j < beg_token_id_; ++j) {
          beam_token_scores[j] = std::numeric_limits<T>::lowest();
        }
      }
    }

// #ifdef DEBUG_GENERATION
//     DumpScores("TimestampLogitsProcessor", next_token_scores);
// #endif
  }

 private:
  int eos_token_id_;
  int max_initial_timestamp_index_;
};

class LogitsProcessorList : public ILogitsProcessorList {
 public:
  LogitsProcessorList() = default;
  void Init(const BeamSearchParameters& parameters);
  void Init(const GreedySearchParameters& parameters);
  void Init(const SamplingParameters& parameters);
  void Process(const ISequences* sequences, gsl::span<float>& next_token_scores, int step);

 private:
  template <typename GenerationParametersT>
  void LogitsProcessorInitImpl(const GenerationParametersT& parameters) {
    processor_list_.clear();

    if (parameters.repetition_penalty != 1.0f) {  // 1.0 means no penalty
      repetition_penalty_processor_ = std::make_unique<RepetitionPenaltyLogitsProcessor<float>>(
          parameters.repetition_penalty);
      processor_list_.push_back(repetition_penalty_processor_.get());
    }

    if (parameters.no_repeat_ngram_size > 0) {
      no_repeat_ngram_processor_ = std::make_unique<
          NoRepeatNGramLogitsProcessor<float>>(parameters.no_repeat_ngram_size);
      processor_list_.push_back(no_repeat_ngram_processor_.get());
    }

    if (!parameters.vocab_mask.empty()) {
      vocab_mask_processor_ = std::make_unique<VocabMaskLogitsProcessor<float>>(parameters.vocab_mask);
      processor_list_.push_back(vocab_mask_processor_.get());
    }

    if (!parameters.prefix_vocab_mask.empty()) {
      prefix_vocab_mask_processor_ = std::make_unique<
          PrefixVocabMaskLogitsProcessor<float>>(parameters.prefix_vocab_mask,
                                                 parameters.batch_size);
      processor_list_.push_back(prefix_vocab_mask_processor_.get());
    }

    if (parameters.min_length > 0) {
      min_length_processor_ = std::make_unique<MinLengthLogitsProcessor<float>>(parameters.min_length,
                                                                                parameters.eos_token_id);
      processor_list_.push_back(min_length_processor_.get());
    }

    if (parameters.temperature > 0) {
      temperature_processor_ = std::make_unique<TemperatureLogitsProcessor<float>>(parameters.temperature);
      processor_list_.push_back(temperature_processor_.get());
    }

    if (!parameters.presence_mask.empty()) {
      presence_penalty_processor_ = std::make_unique<
          PresencePenaltyLogitsProcessor<float>>(parameters.presence_mask,
                                                 parameters.presence_penalty);
      processor_list_.push_back(presence_penalty_processor_.get());
    }

    // Add timestamp processor for whisper model
    if (parameters.model_type == IGenerationParameters::kModelTypeWhisper && parameters.logits_processor == IGenerationParameters::kLogitsProcessorTypeWhisper) {
      constexpr int max_initial_timestamp_index = 50;
      timestamp_processor_ = std::make_unique<TimestampLogitsProcessor<float>>(parameters.eos_token_id, max_initial_timestamp_index);
      processor_list_.push_back(timestamp_processor_.get());
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
  std::unique_ptr<TemperatureLogitsProcessor<float>> temperature_processor_;
  std::unique_ptr<PresencePenaltyLogitsProcessor<float>> presence_penalty_processor_;
  std::unique_ptr<TimestampLogitsProcessor<float>> timestamp_processor_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
