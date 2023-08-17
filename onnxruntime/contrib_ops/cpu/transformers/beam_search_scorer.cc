// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "contrib_ops/cpu/transformers/beam_search_scorer.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {
using ::onnxruntime::rnn::detail::Allocate;

void BeamHypotheses::Init(float length_penalty, gsl::span<HypothesisScore> beams) {
  beams_ = beams;
  beams_used_ = 0;
  length_penalty_ = length_penalty;
  done_ = false;
}

void BeamHypotheses::Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs) {
  auto length = hypothesis.size();
  float score = sum_logprobs / pow(static_cast<float>(length), length_penalty_);

  size_t index = beams_used_;
  // If the array is full, don't add unless it's better than the worst element
  if (index == beams_.size()) {
    if (score <= beams_[--index].score)
      return;
  } else
    beams_used_++;

  // Rotate existing elements over while the new element scores higher
  for (; index > 0 && score > beams_[index - 1].score; index--)
    beams_[index] = beams_[index - 1];

  beams_[index] = HypothesisScore{hypothesis, score};
}

bool BeamHypotheses::CanImprove(float best_sum_logprobs, int current_length) const {
  float current_score = best_sum_logprobs / pow(static_cast<float>(current_length), length_penalty_);
  return beams_.back().score < current_score;
}

void BeamHypotheses::Output(
    int top_k,
    int max_length,
    gsl::span<int32_t>& sequences,       // buffer filled with pad token ID, shape (num_return_sequences, max_length)
    gsl::span<float>& sequences_scores)  // buffer of shape (num_return_sequences) or empty
{
  // Copy the top_k beams into the sequences
  ORT_ENFORCE(top_k <= beams_used_);
  for (int index = 0; index < top_k; index++) {
    auto& item = beams_[index];
    gsl::span<int32_t> target = sequences.subspan(static_cast<gsl::index>(index) * max_length, max_length);

    // Note that word_ids might be less than max_length.
    // Since the sequences has been filled with pad token ID, so padding is not needed here.
    gsl::copy(item.hypothesis, target);

    if (!sequences_scores.empty())
      sequences_scores[index] = item.score;
  }
}

BeamSearchScorer::BeamSearchScorer(const IGenerationParameters& parameters,
                                   AllocatorPtr& allocator)
    : batch_size_{static_cast<size_t>(parameters.batch_size)},
      num_beams_{static_cast<size_t>(parameters.num_beams)},
      max_length_{static_cast<size_t>(parameters.max_length)},
      num_return_sequences_{static_cast<size_t>(parameters.num_return_sequences)},
      pad_token_id_{parameters.pad_token_id},
      eos_token_id_{parameters.eos_token_id},
      early_stopping_{parameters.early_stopping},
      not_done_count_{parameters.batch_size} {
  size_t batch_beam_size = batch_size_ * num_beams_;

  auto beams = Allocate<HypothesisScore>(allocator, batch_beam_size, hypothesis_scores_ptr_);
  beam_hyps_ = Allocate<BeamHypotheses>(allocator, batch_size_, beam_hyps_ptr_);
  for (size_t i = 0; i < batch_size_; i++)
    beam_hyps_[i].Init(parameters.length_penalty, beams.subspan(i * num_beams_, num_beams_));

  next_beam_scores_ = Allocate<float>(allocator, batch_beam_size, next_beam_scores_ptr_);
  next_beam_tokens_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_tokens_ptr_);
  next_beam_indices_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_indices_ptr_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t per_beam = (SafeInt<size_t>(max_length_) * (max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ = Allocate<int32_t>(allocator, batch_beam_size * per_beam, hypothesis_buffer_ptr_);
}

void BeamSearchScorer::Process(ISequences& sequences,
                               gsl::span<const float>& next_scores,
                               gsl::span<const int32_t>& next_tokens,
                               gsl::span<const int32_t>& next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  const int sequence_length = sequences.GetSequenceLength();

  ORT_ENFORCE(next_scores.size() == next_tokens.size());
  ORT_ENFORCE(next_scores.size() == next_indices.size());

  for (size_t batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch];
    if (beam_hyp.done_) {
      ORT_ENFORCE(beam_hyp.beams_used_ == gsl::narrow_cast<int>(num_beams_),
                  "Batch can only be done if all beams have been generated");

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
        gsl::span<const int32_t> src = sequences.GetSequence(batch_beam_idx);
        auto clone = hypothesis_buffer_.subspan(static_cast<size_t>(hypothesis_buffer_used_), sequence_length);

        gsl::copy(src, clone);
        hypothesis_buffer_used_ += sequence_length;
        auto sequence = ReinterpretAsSpan<const int32_t>(clone);
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
    ORT_ENFORCE(static_cast<size_t>(hypothesis_buffer_used_) <= hypothesis_buffer_.size());

    //  Check if we are done so that we can save a pad step if all(done)
    if (static_cast<size_t>(beam_hyp.beams_used_) < num_beams_)
      continue;

    if (!early_stopping_) {
      gsl::span<const float> topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const auto best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.CanImprove(*best_sum_logprobs, sequence_length))
        continue;
    }

    beam_hyp.done_ = true;
    not_done_count_--;
  }
}

void BeamSearchScorer::Finalize(ISequences& sequences,
                                gsl::span<const float>& final_beam_scores,
                                Tensor* output_sequences,
                                Tensor* output_sequence_scores) {
  ORT_ENFORCE(output_sequences != nullptr);

  // Finalize all open beam hypotheses and add to generated hypotheses.
  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];
    if (beam_hyp.done_) {
      continue;
    }

    for (size_t beam_index = 0; beam_index < num_beams_; beam_index++) {
      size_t batch_beam_index = batch_index * num_beams_ + beam_index;
      float final_score = final_beam_scores[batch_beam_index];
      auto final_tokens = sequences.GetSequence(narrow<int>(batch_beam_index));
      beam_hyp.Add(final_tokens, final_score);
    }
  }

  // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length).
  gsl::span<int32_t> output = output_sequences->MutableDataAsSpan<int32_t>();

  // Fill output sequences with pad token ID so that we do not need append it later.
  std::fill_n(output.data(), output.size(), pad_token_id_);

  // Score of each sequence, with shape (batch_size * num_return_sequences).
  gsl::span<float> sequence_scores;
  if (output_sequence_scores) {
    sequence_scores = output_sequence_scores->MutableDataAsSpan<float>();
  }

  // Select the best hypotheses according to number of sequences to return.
  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];

    auto batch_output = output.subspan(batch_index * num_return_sequences_ * max_length_,
                                       num_return_sequences_ * max_length_);
    gsl::span<float> sequence_scores_buffer;
    if (!sequence_scores.empty())
      sequence_scores_buffer = sequence_scores.subspan(batch_index * num_return_sequences_, num_return_sequences_);

    beam_hyp.Output(narrow<int>(num_return_sequences_), narrow<int>(max_length_), batch_output,
                    sequence_scores_buffer);
  }
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
