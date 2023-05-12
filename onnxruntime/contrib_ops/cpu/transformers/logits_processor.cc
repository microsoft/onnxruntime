// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <assert.h>
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "core/providers/cpu/math/softmax_shared.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
gsl::span<T> NextTokenScores<T>::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < batch_beam_size);
  return scores.subspan(static_cast<gsl::index>(batch_beam_index) * vocab_size, vocab_size);
}

template <typename T>
void NextTokenScores<T>::SetScore(int token_id, T score) {
  assert(token_id >= 0 && token_id < vocab_size);
  for (int i = 0; i < batch_beam_size; i++) {
    scores[static_cast<gsl::index>(i) * vocab_size + token_id] = score;
  }
}

#ifdef DEBUG_GENERATION
template <typename T>
void DumpScores(const char* name, const NextTokenScores<T>& next_token_scores) {
  std::cout << name << std::endl;
  ORT_UNUSED_PARAMETER(next_token_scores);
}
#endif

// Interface for all scorers for beam search or beam sample.
template <typename T>
MinLengthLogitsProcessor<T>::MinLengthLogitsProcessor(int min_length, int eos_token_id)
    : min_length_(min_length), eos_token_id_(eos_token_id) {}

template <typename T>
void MinLengthLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {
  if (sequences->GetSequenceLength() < min_length_) {
    next_token_scores.SetScore(eos_token_id_, std::numeric_limits<T>::lowest());
  }

#ifdef DEBUG_GENERATION
  DumpScores("MinLengthLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
RepetitionPenaltyLogitsProcessor<T>::RepetitionPenaltyLogitsProcessor(float penalty) : penalty_(penalty) {
}

template <typename T>
void RepetitionPenaltyLogitsProcessor<T>::Process(const ISequences* sequences,
                                                  NextTokenScores<T>& next_token_scores) {
  const int batch_beam_size = next_token_scores.batch_beam_size;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
    gsl::span<const int32_t> sequence = sequences->GetSequence(i);

    // Find unique word IDs in sequence.
    std::unordered_set<int32_t> unique_word_ids;
    for (const auto& word_id : sequence) {
      unique_word_ids.insert(word_id);
    }

    for (const int32_t word_id : unique_word_ids) {
      T score = beam_token_scores[word_id];

      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * penalty_ : score / penalty_);
    }
  }

#ifdef DEBUG_GENERATION
  DumpScores("RepetitionPenaltyLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
NoRepeatNGramLogitsProcessor<T>::NoRepeatNGramLogitsProcessor(int ngram_size) : ngram_size_(ngram_size) {
}

template <typename T>
void NoRepeatNGramLogitsProcessor<T>::Process(const ISequences* sequences,
                                              NextTokenScores<T>& next_token_scores) {
  if (ngram_size_ == 0 || ngram_size_ > sequences->GetSequenceLength()) {
    return;
  }

  const gsl::index prefix_length = static_cast<gsl::index>(ngram_size_) - 1;
  int batch_beam_size = next_token_scores.batch_beam_size;

  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
    gsl::span<const int32_t> sequence = sequences->GetSequence(i);

    gsl::span<const int32_t> prefix = sequence.subspan(sequence.size() - prefix_length);
    ORT_ENFORCE(prefix.size() == narrow<size_t>(prefix_length));

    std::unordered_set<int32_t> blocked_word_ids;
    for (int j = 0; j <= static_cast<int>(sequence.size()) - ngram_size_; j++) {
      // Here we use naive algorithm for matching. The complexity is O(batch_beam_size * ngram_size * sequence_length)
      // TODO(tianleiwu): build N-Gram index (hash table with prefix of length NGram - 1 as key,
      //                  and list of last word of NGram as value) for fast matching.
      if (ngram_size_ == 1 || SpanEq(prefix, sequence.subspan(j, prefix_length))) {
        blocked_word_ids.insert(sequence[static_cast<gsl::index>(j) + prefix_length]);
      }
    }

    for (const int32_t word_id : blocked_word_ids) {
      beam_token_scores[word_id] = std::numeric_limits<T>::lowest();
    }
  }

#ifdef DEBUG_GENERATION
  DumpScores("NoRepeatNGramLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
VocabMaskLogitsProcessor<T>::VocabMaskLogitsProcessor(const gsl::span<const int32_t>& vocab_mask)
    : vocab_mask_(vocab_mask) {
}

template <typename T>
void VocabMaskLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                          NextTokenScores<T>& next_token_scores) {
  assert(!vocab_mask_.empty());

  // Process vocabulary mask and set tokens with mask value 0 to -inf.
  T* p = next_token_scores.scores.data();
  // next_token_scores shape (batch_size * num_beams, vocab_size)
  // vocab_mask shape (vocab_size).
  for (int i = 0; i < next_token_scores.batch_beam_size; i++) {
    for (int j = 0; j < next_token_scores.vocab_size; j++, p++) {
      if (vocab_mask_[j] == 0) {
        *p = std::numeric_limits<T>::lowest();
      }
    }
  }

#ifdef DEBUG_GENERATION
  DumpScores("VocabMaskLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
PrefixVocabMaskLogitsProcessor<T>::PrefixVocabMaskLogitsProcessor(const gsl::span<const int32_t>& prefix_vocab_mask,
                                                                  int batch_size)
    : prefix_vocab_mask_(prefix_vocab_mask),
      batch_size_(batch_size) {
}

template <typename T>
void PrefixVocabMaskLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                                NextTokenScores<T>& next_token_scores) {
  assert(!prefix_vocab_mask_.empty());

  // next_token_scores shape (batch_size * num_beams, vocab_size)
  int num_beams = next_token_scores.batch_beam_size / batch_size_;
  assert(num_beams * batch_size_ == next_token_scores.batch_beam_size);

  // Process prefix vocabulary mask and set tokens with mask value 0 to -inf.
  // prefix_vocab_mask shape (batch_size, vocab_size).
  T* p = next_token_scores.scores.data();
  for (int i = 0; i < batch_size_; i++) {
    size_t prefix_vocab_mask_offset = SafeInt<size_t>(i) * next_token_scores.vocab_size;
    for (int j = 0; j < num_beams; j++) {
      for (int k = 0; k < next_token_scores.vocab_size; k++, p++) {
        if (prefix_vocab_mask_[prefix_vocab_mask_offset + static_cast<size_t>(k)] == 0) {
          *p = std::numeric_limits<T>::lowest();
        }
      }
    }
  }

#ifdef DEBUG_GENERATION
  DumpScores("PrefixVocabMaskLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
TemperatureLogitsProcessor<T>::TemperatureLogitsProcessor(float temperature) : temperature_(temperature) {
}

template <typename T>
void TemperatureLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                            NextTokenScores<T>& next_token_scores) {
  if (temperature_ == 1.0f) {
    return;
  }

  T* p = next_token_scores.scores.data();
  for (size_t i = 0; i < next_token_scores.scores.size(); i++) {
    *p /= temperature_;
    ++p;
  }

#ifdef DEBUG_GENERATION
  DumpScores("TemperatureLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
PresencePenaltyLogitsProcessor<T>::PresencePenaltyLogitsProcessor(const gsl::span<const int32_t>& presence_mask,
                                                                  float presence_penalty)
    : presence_mask_(presence_mask), presence_penalty_(presence_penalty) {
}

template <typename T>
void PresencePenaltyLogitsProcessor<T>::Process(const ISequences*,
                                                NextTokenScores<T>& next_token_scores) {
  if (presence_penalty_ == 0.0f) {
    return;
  }

  assert(!presence_mask_.empty());

  T* p = next_token_scores.scores.data();
  for (size_t i = 0; i < next_token_scores.scores.size(); i++) {
    *p -= presence_mask_[i] * presence_penalty_;
  }

#ifdef DEBUG_GENERATION
  DumpScores("PresencePenaltyLogitsProcessor", next_token_scores);
#endif
}

template <typename T>
TimestampLogitsProcessor<T>::TimestampLogitsProcessor(int eos_token_id, int max_initial_timestamp_index)
    : eos_token_id_(eos_token_id), max_initial_timestamp_index_(max_initial_timestamp_index) {}

template <typename T>
void TimestampLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {
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

#ifdef DEBUG_GENERATION
  DumpScores("TimestampLogitsProcessor", next_token_scores);
#endif
}

void LogitsProcessorList::Init(const BeamSearchParameters& parameters) {
  LogitsProcessorInitImpl<BeamSearchParameters>(parameters);
}

void LogitsProcessorList::Init(const GreedySearchParameters& parameters) {
  LogitsProcessorInitImpl<GreedySearchParameters>(parameters);
}

void LogitsProcessorList::Init(const SamplingParameters& parameters) {
  LogitsProcessorInitImpl<SamplingParameters>(parameters);
}

void LogitsProcessorList::Process(const ISequences* sequences,
                                  gsl::span<float>& next_token_scores,
                                  int step) {
  NextTokenScores<float> input_scores = {next_token_scores, batch_beam_size_, vocab_size_};
  for (size_t i = 0; i < processor_list_.size(); i++) {
    // Prefix vocab mask is applied to first iteration only.
    if (step > 1 && processor_list_[i] == prefix_vocab_mask_processor_.get()) {
      continue;
    }
    processor_list_[i]->Process(sequences, input_scores);
  }
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
