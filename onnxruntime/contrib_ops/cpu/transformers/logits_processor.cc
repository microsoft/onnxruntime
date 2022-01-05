#include <assert.h>
#include "logits_processor.h"
#include "dump_tensor.h"
#ifdef _MSC_VER
// Could reduce the chance of arithmetic overflow. TODO: fix it
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
gsl::span<T> NextTokenScores<T>::GetScores(int batch_beam_index) {
  assert(batch_beam_index >= 0 && batch_beam_index < batch_beam_size);
  return scores.subspan(batch_beam_index * vocab_size, vocab_size);
}

template <typename T>
void NextTokenScores<T>::SetScore(int token_id, T score) {
  assert(token_id >= 0 && token_id < vocab_size);
  for (int i = 0; i < batch_beam_size; i++) {
    scores[i * vocab_size + token_id] = score;
  }
}

#ifdef DEBUG_BEAM_SEARCH
template <typename T>
void DumpScores(const char* name, gsl::span<T>& scores) {
  DumpString(name, 0, true);
  ORT_UNUSED_PARAMETER(scores);
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

#ifdef DEBUG_BEAM_SEARCH
  DumpScores("MinLengthLogitsProcessor", next_token_scores.scores);
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
    gsl::span<const int64_t> sequence = sequences->GetSequence(i);

    // Find unique word IDs in sequence.
    std::unordered_set<int64_t> unique_word_ids;
    for (const auto& word_id : sequence) {
      unique_word_ids.insert(word_id);
    }

    for (const int64_t word_id : unique_word_ids) {
      T score = beam_token_scores[word_id];

      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * penalty_ : score / penalty_);
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  DumpScores("RepetitionPenaltyLogitsProcessor", next_token_scores.scores);
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

  const gsl::index prefix_length = static_cast<gsl::index>(ngram_size_ - 1);
  int batch_beam_size = next_token_scores.batch_beam_size;

  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
    gsl::span<const int64_t> sequence = sequences->GetSequence(i);

    gsl::span<const int64_t> prefix = sequence.subspan(sequence.length() - prefix_length);
    ORT_ENFORCE(prefix.length() == prefix_length);

    std::unordered_set<int64_t> blocked_word_ids;
    for (int j = 0; j <= static_cast<int>(sequence.length()) - ngram_size_; j++) {
      // Here we use naive algorithm for matching. The complexity is O(batch_beam_size * ngram_size * sequence_length)
      // TODO: build N-Gram index (hash table with prefix of length NGram - 1 as key, and list of last word of NGram as value) for fast matching.
      if (ngram_size_ == 1 || prefix == sequence.subspan(j, prefix_length)) {
        blocked_word_ids.insert(sequence[j + prefix_length]);
      }
    }

    for (const int64_t word_id : blocked_word_ids) {
      beam_token_scores[word_id] = std::numeric_limits<T>::lowest();
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  DumpScores("NoRepeatNGramLogitsProcessor", next_token_scores.scores);
#endif
}

template <typename T>
VocabMaskLogitsProcessor<T>::VocabMaskLogitsProcessor(const gsl::span<const int32_t>& vocab_mask) : vocab_mask_(vocab_mask) {
}

template <typename T>
void VocabMaskLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                          NextTokenScores<T>& next_token_scores) {
  assert(!vocab_mask_.empty());

  // Process vocabulary mask and set tokens with mask value 0 to -inf.
  T* p = next_token_scores.scores.data();
  // next_token_scores shape (batch_size * num_beams, vocab_size)
  // vocab_mask shape (vocab_size). TODO: support shape (batch_size, vocab_size)
  for (int i = 0; i < next_token_scores.batch_beam_size; i++) {
    for (int j = 0; j < next_token_scores.vocab_size; j++, p++) {
      if (vocab_mask_[j] == 0) {
        *p = std::numeric_limits<T>::lowest();
      }
    }
  }

#ifdef DEBUG_BEAM_SEARCH
  DumpScores("VocabMaskLogitsProcessor", next_token_scores.scores);
#endif
}

template <typename T>
void LogitsProcessorList<T>::Init(const BeamSearchParameters& parameters) {
  processor_list_.clear();

  if (parameters.repetition_penalty != 1.0f) {  // 1.0 means no penalty
    repetition_penalty_processor_ = std::make_unique<RepetitionPenaltyLogitsProcessor<T>>(parameters.repetition_penalty);
    processor_list_.push_back(repetition_penalty_processor_.get());
  }

  if (parameters.no_repeat_ngram_size > 0) {
    no_repeat_ngram_processor_ = std::make_unique<NoRepeatNGramLogitsProcessor<T>>(parameters.no_repeat_ngram_size);
    processor_list_.push_back(no_repeat_ngram_processor_.get());
  }

  if (!parameters.vocab_mask.empty()) {
    vocab_mask_processor_ = std::make_unique<VocabMaskLogitsProcessor<T>>(parameters.vocab_mask);
    processor_list_.push_back(vocab_mask_processor_.get());
  }

  if (parameters.min_length > 0) {
    min_length_processor_ = std::make_unique<MinLengthLogitsProcessor<T>>(parameters.min_length, parameters.eos_token_id);
    processor_list_.push_back(min_length_processor_.get());
  }

  batch_beam_size_ = parameters.BatchBeamSize();
  vocab_size_ = parameters.vocab_size;
}

template <typename T>
void LogitsProcessorList<T>::Process(const ISequences* sequences,
                                     gsl::span<T>& next_token_scores) {
  NextTokenScores<T> input_scores = {next_token_scores, batch_beam_size_, vocab_size_};
  for (size_t i = 0; i < processor_list_.size(); i++) {
    processor_list_[i]->Process(sequences, input_scores);
  }
}

// Instantiation
template class MinLengthLogitsProcessor<float>;
template class RepetitionPenaltyLogitsProcessor<float>;
template class NoRepeatNGramLogitsProcessor<float>;
template class VocabMaskLogitsProcessor<float>;
template class LogitsProcessorList<float>;

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime