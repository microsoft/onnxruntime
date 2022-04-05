// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/common/safeint.h"
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

BeamHypotheses::BeamHypotheses(int num_beams,
                               float length_penalty,
                               bool early_stopping,
                               onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator)
    : num_beams_(num_beams),
      length_penalty_(length_penalty),
      early_stopping_(early_stopping),
      worst_score_(1e9),
      beams_(hypothesis_score_allocator) {
  best_net_ = std::numeric_limits<float>::lowest();
}

void BeamHypotheses::Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs) {
  /* TODO
   * Is the length_penalty_ needed for final hypothesus?
   * Understand the repercussions of this. Can be made optional.
   *
  auto length = hypothesis.size();
  float score = sum_logprobs / pow(static_cast<float>(length), length_penalty_);
  */
  float score = sum_logprobs;

  if (this->Size() < num_beams_ || score > worst_score_) {
    HypothesisScore item(hypothesis, score);
    beams_.push(item);
    if (this->Size() > num_beams_) {
      beams_.pop();
    }
    worst_score_ = beams_.top().score;
  }
}

bool BeamHypotheses::IsEcsBetterThanLastIteration(ISequences* sequences,
                                        int batch,
                                        int initial_seq_length,
                                        const int prefix_length,
                                        int num_beams,
                                        gsl::span<const int32_t>* ids2_len,
                                        int min_chars,
                                        float ecs_cost,
                                        float log_prob_threshold,
                                        gsl::span<const float>& next_scores,
                                        gsl::span<const int32_t>& next_tokens) {
  int max_index = 0;
  float max_generated_net = std::numeric_limits<float>::lowest();

  for (int i = 0; i < num_beams; i++) {
    int batch_beam_idx = batch * num_beams + i; 
    gsl::span<const int32_t> src = sequences->GetSequence(batch_beam_idx);

    int generated_length = 0; 
    for (int j = initial_seq_length; j < sequences->GetSequenceLength(); j++) {
      generated_length = generated_length + (*ids2_len)[src[j]];
    }

    // Each beam would have 2 as internally we are extracting topK = 2* num_beams_
    for (int j = 0; j < 2; j++) {
      int beam_idx = batch * 2 * num_beams + j;
      int beam_generated_length = generated_length + (*ids2_len)[next_tokens[beam_idx]];
      beam_generated_length = beam_generated_length - prefix_length;
      float generated_net = std::numeric_limits<float>::lowest();;

      if (beam_generated_length < min_chars) {
        generated_net = std::numeric_limits<float>::lowest();
      } else {
        float generated_prob = exp(next_scores[beam_idx]);
        float generated_ecs = beam_generated_length * generated_prob;
        float generated_cost = beam_generated_length * ecs_cost;
        generated_net = generated_ecs - generated_cost;
      }

      // beam_generated_length is less than min_chars, it will always default to the first index, topK are sorted
      // so that is OK. may be a better way to make sure it will find the maximum
      if (generated_net > max_generated_net) {
        max_generated_net = generated_net;
        max_index = beam_idx;
      }
    }
  }

  if (max_generated_net >= best_net_ && next_scores[max_index] >= log_prob_threshold) {
    best_net_ = max_generated_net;
    return false;
  }

  return true;
}

bool BeamHypotheses::IsDone(float best_sum_logprobs) {
  // If there are enough hypotheses and that none of the hypotheses being generated can become better
  // than the worst one in the heap, then we are done with this sentence.

  if (Size() < num_beams_)
    return false;

  if (early_stopping_)
    return true;

  // TODO Add is changed so is this
  // float current_score = best_sum_logprobs / pow(static_cast<float>(current_length), length_penalty_);
  return worst_score_ >= best_sum_logprobs;
}

void BeamHypotheses::Output(
    int top_k,
    int max_length,
    gsl::span<int32_t>& sequences,       // buffer filled with pad token ID, shape (num_return_sequences, max_length)
    gsl::span<float>& sequences_scores)  // buffer of shape (num_return_sequences) or empty
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
    gsl::span<const int32_t>& source = item.hypothesis;
    gsl::span<int32_t> target = sequences.subspan(static_cast<gsl::index>(index) * max_length, max_length);

    // Note that word_ids might be less than max_length.
    // Since the sequences has been filled with pad token ID, so padding is not needed here.
    gsl::copy(source, target);

    if (!sequences_scores.empty())
      sequences_scores[index] = item.score;

    beams_.pop();
    index--;
  }
}

BeamSearchScorer::BeamSearchScorer(size_t batch_size,
                                   size_t num_beams,
                                   size_t max_length,
                                   float length_penalty,
                                   bool early_stopping,
                                   size_t num_return_sequences,
                                   int pad_token_id,
                                   int eos_token_id,
                                   int ecs_min_chars,
                                   float ecs_log_prob_threshold,
                                   float ecs_cost,
                                   onnxruntime::OrtStlAllocator<HypothesisScore>& hypothesis_score_allocator,
                                   onnxruntime::OrtStlAllocator<BeamHypotheses>& beam_hyps_allocator)
    : batch_size_(batch_size),
      num_beams_(num_beams),
      max_length_(max_length),
      num_beam_hyps_to_keep_(num_return_sequences),
      pad_token_id_(pad_token_id),
      eos_token_id_(eos_token_id),
      ecs_min_chars_(ecs_min_chars),
      ecs_log_prob_threshold_(ecs_log_prob_threshold),
      ecs_cost_(ecs_cost),
      hypothesis_buffer_length_(0),
      hypothesis_buffer_offset_(0),
      beam_hyps_(beam_hyps_allocator) {
  for (size_t i = 0; i < batch_size; i++) {
    beam_hyps_.push_back(BeamHypotheses(num_beams, length_penalty, early_stopping, hypothesis_score_allocator));
  }
}

bool BeamSearchScorer::IsDone() {
  for (size_t batch = 0; batch < batch_size_; batch++) {
    if (!done_[batch])
      return false;
  }
  return true;
}

void BeamSearchScorer::Initialize(AllocatorPtr& allocator,
                                  int sequence_length,
                                  gsl::span<const int32_t>& ids2_len,
                                  gsl::span<const int32_t>& prefix_lens) {
  ORT_ENFORCE(next_beam_scores_.empty());  // Make sure this is called only once.

  size_t batch_beam_size = batch_size_ * num_beams_;
  constexpr bool no_fill = false;  // do not fill values after allocation

  done_ = Allocate<bool>(allocator, batch_size_, done_ptr_, no_fill);
  std::fill_n(done_.data(), done_.size(), false);

  next_beam_scores_ = Allocate<float>(allocator, batch_beam_size, next_beam_scores_ptr_, no_fill);
  next_beam_tokens_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_tokens_ptr_, no_fill);
  next_beam_indices_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_indices_ptr_, no_fill);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t buffer_per_beam = (SafeInt<size_t>(max_length_) * (max_length_ + 1) - SafeInt<size_t>(sequence_length - 1) * sequence_length) / 2;
  hypothesis_buffer_length_ = batch_beam_size * buffer_per_beam;
  hypothesis_buffer_ = Allocate<int32_t>(allocator, hypothesis_buffer_length_, hypothesis_buffer_ptr_, no_fill);

  ids2_len_ = &ids2_len;
  prefix_lens_ = &prefix_lens;
  intial_sequence_length_ = sequence_length;
}

void BeamSearchScorer::Process(ISequences* sequences,
                               gsl::span<const float>& next_scores,
                               gsl::span<const int32_t>& next_tokens,
                               gsl::span<const int32_t>& next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  const int sequence_length = sequences->GetSequenceLength();

  ORT_ENFORCE(next_scores.size() == next_tokens.size());
  ORT_ENFORCE(next_scores.size() == next_indices.size());

  for (size_t batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch];

    // ecs_min_chars is never negative, using this to decide whether to call ECS
    if (ecs_min_chars_ >= 0) {
      bool done_here = beam_hyp.IsEcsBetterThanLastIteration(sequences,
                                                             batch,
                                                             intial_sequence_length_,
                                                             (*prefix_lens_)[batch],
                                                             num_beams_,
                                                             ids2_len_,
                                                             ecs_min_chars_,
                                                             ecs_cost_,
                                                             ecs_log_prob_threshold_,
                                                             next_scores,
                                                             next_tokens);

      if (done_here) {
        for (size_t beam = 0; beam < num_beams_; beam++) {
          int batch_beam_idx = batch * num_beams_ + beam;

          // Clone the sequence and append to buffer.
          gsl::span<const int32_t> src = sequences->GetSequence(batch_beam_idx);
          float score = sequences->GetSequenceScore(batch_beam_idx);

          auto clone = hypothesis_buffer_.subspan(hypothesis_buffer_offset_, sequence_length);
          gsl::copy(src, clone);
          hypothesis_buffer_offset_ += static_cast<size_t>(sequence_length);
          auto sequence = clone.template as_span<const int32_t>();

          beam_hyp.Add(sequence, score);
        }
        done_[batch] = true;
      }
    }

    if (done_[batch]) {
      ORT_ENFORCE(beam_hyp.Size() >= gsl::narrow_cast<int>(num_beams_), "Batch can only be done if all beams have been generated");

      // Pad the batch.
      for (size_t j = 0; j < num_beams_; j++) {
        next_beam_scores_[batch * num_beams_ + j] = 0.0f;
        next_beam_tokens_[batch * num_beams_ + j] = pad_token_id_;
        next_beam_indices_[batch * num_beams_ + j] = 0;
      }
      continue;
    }

    // Next tokens for this sentence.
    size_t beam_idx = 0;
    size_t top_k = 2 * num_beams_;
    for (size_t j = 0; j < top_k; j++) {
      int32_t next_token = next_tokens[batch * top_k + j];
      float next_score = next_scores[batch * top_k + j];
      int32_t next_index = next_indices[batch * top_k + j];

      int batch_beam_idx = static_cast<int>(batch * num_beams_) + next_index;
      // Add to generated hypotheses if end of sentence.
      if ((eos_token_id_ >= 0) && (next_token == eos_token_id_)) {
        bool is_beam_token_worse_than_top_num_beams = (j >= num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and append to buffer.
        gsl::span<const int32_t> src = sequences->GetSequence(batch_beam_idx);
        auto clone = hypothesis_buffer_.subspan(hypothesis_buffer_offset_, sequence_length);
        gsl::copy(src, clone);
        hypothesis_buffer_offset_ += static_cast<size_t>(sequence_length);
        auto sequence = clone.template as_span<const int32_t>();
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
      gsl::span<const float> topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const float* best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.IsDone(*best_sum_logprobs)) {
        done_[batch] = true;
      }
    }
  }
}

void BeamSearchScorer::Finalize(ISequences* sequences,
                                gsl::span<const float>& final_beam_scores,
                                Tensor* output_sequences,
                                Tensor* output_sequence_scores) {
  ORT_ENFORCE(sequences != nullptr);
  ORT_ENFORCE(output_sequences != nullptr);

  // Finalize all open beam hypotheses and add to generated hypotheses.
  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];
    if (done_[batch_index]) {
      continue;
    }

    for (size_t beam_index = 0; beam_index < num_beams_; beam_index++) {
      size_t batch_beam_index = batch_index * num_beams_ + beam_index;
      float final_score = final_beam_scores[batch_beam_index];
      auto final_tokens = sequences->GetSequence(batch_beam_index);
      beam_hyp.Add(final_tokens, final_score);
    }
  }

  // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length).
  gsl::span<int32_t> output = output_sequences->MutableDataAsSpan<int32_t>();

  // Fill output sequences with pad token ID so that we do not need append it later.
  std::fill_n(output.data(), output.size(), pad_token_id_);

  // Score of each sequence, with shape (batch_size * num_return_sequences).
  gsl::span<float> sequence_scores;
  if (output_sequence_scores != nullptr) {
    sequence_scores = output_sequence_scores->MutableDataAsSpan<float>();
  }

  // Span is empty when output_sequence_scores is NULL.
  gsl::span<float> batch_sequence_score;

  // Select the best hypotheses according to number of sequences to return.
  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];

    const size_t num_return_sequences = num_beam_hyps_to_keep_;
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

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime