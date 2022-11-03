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
TopPLogitsProcessor<T>::TopPLogitsProcessor(float top_p, float filter_value,
                                            onnxruntime::concurrency::ThreadPool* thread_pool)
    : top_p_(top_p), filter_value_(filter_value), thread_pool_(thread_pool) {
}

template <typename T>
void TopPLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                     NextTokenScores<T>& next_token_scores) {
  if (top_p_ == 0.0f) {
    return;
  }

  const int batch_beam_size = next_token_scores.batch_beam_size;
  const int vocab_size = next_token_scores.vocab_size;

  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);

    std::vector<T> sorted_scores(beam_token_scores.begin(), beam_token_scores.end());

    // decending sort
    std::vector<size_t> sorted_indices(beam_token_scores.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(),
              sorted_indices.end(),
              [&sorted_scores](size_t i1, size_t i2) {
                return sorted_scores[i1] > sorted_scores[i2];
              });

    std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<T>());
    std::vector<T> cumulative_probs(vocab_size);
    // todo: batch
    ORT_UNUSED_PARAMETER(SoftmaxCPU<T>(1,
                                       vocab_size,
                                       sorted_scores.data(),
                                       cumulative_probs.data(),
                                       false,
                                       thread_pool_));

    std::unordered_set<size_t> sorted_indices_to_remove;
    if (cumulative_probs[0] > top_p_) {
      sorted_indices_to_remove.insert(1);
    }
    for (size_t j = 1; j < static_cast<size_t>(vocab_size) - 1; j++) {
      cumulative_probs[j] += cumulative_probs[j - 1];
      if (cumulative_probs[j] > top_p_) {
        sorted_indices_to_remove.insert(j + 1);
      }
    }

    for (auto it = sorted_indices_to_remove.begin(); it != sorted_indices_to_remove.end(); ++it) {
      size_t index_to_remove = sorted_indices[*it];
      beam_token_scores[index_to_remove] = filter_value_;
    }
  }

#ifdef DEBUG_GENERATION
  DumpScores("TopPLogitsProcessor", next_token_scores);
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

void LogitsProcessorList::Init(const BeamSearchParameters& parameters,
                               onnxruntime::concurrency::ThreadPool* thread_pool) {
  LogitsProcessorInitImpl<BeamSearchParameters>(parameters, thread_pool);
}

void LogitsProcessorList::Init(const GreedySearchParameters& parameters,
                               onnxruntime::concurrency::ThreadPool* thread_pool) {
  LogitsProcessorInitImpl<GreedySearchParameters>(parameters, thread_pool);
}

void LogitsProcessorList::Init(const SamplingParameters& parameters,
                               onnxruntime::concurrency::ThreadPool* thread_pool) {
  LogitsProcessorInitImpl<SamplingParameters>(parameters, thread_pool);
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
