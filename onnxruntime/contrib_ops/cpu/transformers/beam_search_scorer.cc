// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "beam_search_scorer.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {
using ::onnxruntime::rnn::detail::Allocate;

template <typename T>
BeamHypotheses<T>::BeamHypotheses(int num_beams, T length_penalty, bool early_stopping)
    : num_beams_(num_beams),
      length_penalty_(length_penalty),
      early_stopping_(early_stopping),
      worst_score_(1e9) {}

template <typename T>
void BeamHypotheses<T>::Add(gsl::span<const int64_t>& hypothesis, T sum_logprobs) {
  auto length = hypothesis.size();
  // TODO: when T is FP16, compute in FP32, then cast result back to FP16. length_penalty_ might also be float.
  T score = sum_logprobs / pow(static_cast<T>(length), length_penalty_);

  if (this->Size() < num_beams_ || score > worst_score_) {
    HypothesisScore<T> item(hypothesis, score);
    beams_.push(item);
    if (this->Size() > num_beams_) {
      beams_.pop();
    }
    worst_score_ = beams_.top().score;
  }
}

template <typename T>
bool BeamHypotheses<T>::IsDone(T best_sum_logprobs, int current_length) {
  // If there are enough hypotheses and that none of the hypotheses being generated can become better
  // than the worst one in the heap, then we are done with this sentence.

  if (Size() < num_beams_)
    return false;

  if (early_stopping_)
    return true;

  T current_score = best_sum_logprobs / pow(static_cast<T>(current_length), length_penalty_);
  return worst_score_ >= current_score;
}

template <typename T>
void BeamHypotheses<T>::Output(
    int top_k,
    int max_length,
    gsl::span<int32_t>& sequences,   // buffer filled with pad token ID, shape (num_return_sequences, max_length)
    gsl::span<T>& sequences_scores)  // buffer of shape (num_return_sequences) or empty
{
  ORT_ENFORCE(top_k <= Size());
  int remove_count = Size() - top_k;
  for (int i = 0; i < remove_count; i++) {
    beams_.pop();
  }

  // Since pop get the worst sequence, so output it in the reverse order.
  // The frist (worst) beam shall be put at the last position among top_k sequences.
  int index = top_k - 1;
  while (!beams_.empty()) {
    auto item = beams_.top();
    gsl::span<const int64_t>& source = item.hypothesis;
    gsl::span<int32_t> target = sequences.subspan(index * max_length, max_length);

    // Note that word_ids might be less than max_length.
    // Since the sequences has been filled with pad token ID, so padding is not needed here.
    // Since data type need cast from int64_t to int32_t, we cannot use gsl::copy(word_ids, sequence) here.
    for (size_t i = 0; i < source.length(); i++) {
      target[i] = static_cast<int32_t>(source[i]);
    }

    if (!sequences_scores.empty())
      sequences_scores[index] = item.score;

    beams_.pop();
    index--;
  }
}

template <typename T>
BeamSearchScorer<T>::BeamSearchScorer(int batch_size,
                                      int num_beams,
                                      int max_length,
                                      T length_penalty,
                                      bool early_stopping,
                                      int num_return_sequences,
                                      int pad_token_id,
                                      int eos_token_id)
    : batch_size_(batch_size),
      num_beams_(num_beams),
      max_length_(max_length),
      num_beam_hyps_to_keep_(num_return_sequences),
      pad_token_id_(pad_token_id),
      eos_token_id_(eos_token_id),
      hypothesis_buffer_length_(0),
      hypothesis_buffer_offset_(0) {
  for (int batch = 0; batch < batch_size; batch++) {
    beam_hyps.push_back(BeamHypotheses(num_beams, length_penalty, early_stopping));
  }
}

template <typename T>
bool BeamSearchScorer<T>::IsDone() {
  for (int batch = 0; batch < batch_size_; batch++) {
    if (!done_[batch])
      return false;
  }
  return true;
}

template <typename T>
void BeamSearchScorer<T>::Initialize(AllocatorPtr& allocator, int sequence_length){
  ORT_ENFORCE(next_beam_scores_.empty()); // Make sure this is called only once.

  size_t batch_beam_size = static_cast<size_t>(batch_size_ * num_beams_);
  const bool no_fill = false; // do not fill values after allocation
  next_beam_scores_ = Allocate<T>(allocator, batch_beam_size, next_beam_scores_ptr_, no_fill);
  next_beam_tokens_ = Allocate<int64_t>(allocator, batch_beam_size, next_beam_tokens_ptr_, no_fill);
  next_beam_indices_ = Allocate<int64_t>(allocator, batch_beam_size, next_beam_indices_ptr_, no_fill);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  int buffer_per_beam = (max_length_ * (max_length_ + 1) - (sequence_length - 1) * sequence_length) / 2;
  hypothesis_buffer_length_ = batch_beam_size * static_cast<size_t>(buffer_per_beam);
  hypothesis_buffer_ = Allocate<int64_t>(allocator, hypothesis_buffer_length_, hypothesis_buffer_ptr_, no_fill);

  done_ = Allocate<bool>(allocator, static_cast<size_t>(batch_size_), done_ptr_, no_fill);
  std::fill_n(done_.data(), done_.size(), false);
}

template <typename T>
void BeamSearchScorer<T>::Process(ISequences* sequences,
                                  gsl::span<const T>& next_scores,
                                  gsl::span<const int64_t>& next_tokens,
                                  gsl::span<const int64_t>& next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  const int sequence_length = sequences->GetSequenceLength();

  ORT_ENFORCE(next_scores.size() == next_tokens.size());
  ORT_ENFORCE(next_scores.size() == next_indices.size());

  for (int batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses<T>& beam_hyp = beam_hyps[batch];
    if (done_[batch]) {
      ORT_ENFORCE(beam_hyp.Size() >= num_beams_, "Batch can only be done if all beams have been generated");

      // Pad the batch.
      for (int j = 0; j < num_beams_; j++) {
        next_beam_scores_[batch * num_beams_ + j] = 0.0f;
        next_beam_tokens_[batch * num_beams_ + j] = pad_token_id_;
        next_beam_indices_[batch * num_beams_ + j] = 0;
      }
      continue;
    }

    // Next tokens for this sentence.
    int beam_idx = 0;
    int top_k = 2 * num_beams_;
    for (int j = 0; j < top_k; j++) {
      int64_t next_token = next_tokens[batch * top_k + j];
      T next_score = next_scores[batch * top_k + j];
      int64_t next_index = next_indices[batch * top_k + j];

      int batch_beam_idx = batch * num_beams_ + static_cast<int>(next_index);
      // Add to generated hypotheses if end of sentence.
      if ((eos_token_id_ >= 0) && (next_token == eos_token_id_)) {
        bool is_beam_token_worse_than_top_num_beams = (j >= num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and append to buffer.
        gsl::span<const int64_t> src = sequences->GetSequence(batch_beam_idx);
        auto clone = hypothesis_buffer_.subspan(hypothesis_buffer_offset_, sequence_length);
        gsl::copy(src, clone);
        hypothesis_buffer_offset_ += sequence_length;
        auto sequence = clone.template as_span<const int64_t>();
        beam_hyp.Add(sequence, next_score);
      } else {
        // Add next predicted token since it is not eos_token.
        next_beam_scores_[batch * num_beams_ + beam_idx] = next_score;
        next_beam_tokens_[batch * num_beams_ + beam_idx] = next_token;
        next_beam_indices_[batch * num_beams_ + beam_idx] = batch_beam_idx;
        ++beam_idx;
      }

      // Once the beam for next step is full, don't add more tokens to it.
      if (beam_idx == num_beams_)
        break;
    }

    ORT_ENFORCE(beam_idx == num_beams_);
    ORT_ENFORCE(hypothesis_buffer_offset_ <= batch_size_ * num_beams_ * max_length_);

    //  Check if we are done so that we can save a pad step if all(done)
    if (!done_[batch]) {
      gsl::span<const T> topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const T* best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.IsDone(*best_sum_logprobs, sequence_length)) {
        done_[batch] = true;
      }
    }
  }
}

template <typename T>
void BeamSearchScorer<T>::Finalize(ISequences* sequences,
                                   gsl::span<const T>& final_beam_scores,
                                   Tensor* output_sequences,
                                   Tensor* output_sequence_scores) {
  ORT_ENFORCE(sequences != nullptr);
  ORT_ENFORCE(output_sequences != nullptr);

  // Finalize all open beam hypotheses and add to generated hypotheses.
  for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses<T>& beam_hyp = beam_hyps[batch_index];
    if (done_[batch_index]) {
      continue;
    }

    for (int beam_index = 0; beam_index < num_beams_; beam_index++) {
      int batch_beam_index = batch_index * num_beams_ + beam_index;
      T final_score = final_beam_scores[batch_beam_index];
      auto final_tokens = sequences->GetSequence(batch_beam_index);
      beam_hyp.Add(final_tokens, final_score);
    }
  }

  // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length).
  gsl::span<int32_t> output = output_sequences->MutableDataAsSpan<int32_t>();

  // Fill output sequences with pad token ID so that we do not need append it later.
  std::fill_n(output.data(), output.size(), pad_token_id_);

  // Score of each sequence, with shape (batch_size * num_return_sequences).
  gsl::span<T> sequence_scores;
  if (output_sequence_scores != nullptr) {
    sequence_scores = output_sequence_scores->MutableDataAsSpan<T>();
  }

  // Span is empty when output_sequence_scores is NULL.
  gsl::span<T> batch_sequence_score;

  // Select the best hypotheses according to number of sequences to return.
  for (int batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses<T>& beam_hyp = beam_hyps[batch_index];

    const int num_return_sequences = num_beam_hyps_to_keep_;
    auto batch_output = output.subspan(batch_index * num_return_sequences * max_length_, num_return_sequences * max_length_);

    if (output_sequence_scores != nullptr) {
      batch_sequence_score = sequence_scores.subspan(batch_index * num_return_sequences, num_return_sequences);
    }

    beam_hyp.Output(
        num_return_sequences,
        max_length_,
        batch_output,
        batch_sequence_score);
  }
}

// Instantiation
template class HypothesisScoreCompare<float>;
template class BeamHypotheses<float>;
template class BeamSearchScorer<float>;

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime