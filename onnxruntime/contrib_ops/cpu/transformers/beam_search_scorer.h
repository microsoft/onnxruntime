// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The implementation is based on huggingface transformers generation_beam_search.py

#pragma once
#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/containers.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct HypothesisScore {
  gsl::span<const int32_t> hypothesis;
  float score;
};

struct BeamHypotheses {
  void Init(const IGenerationParameters& parameters, gsl::span<HypothesisScore> beams);

  // Number of hypotheses
  int Size() const { return beams_used_; }

  // Add a new hypothesis
  void Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs);

  bool IsDone(float best_sum_logprobs, int current_length) const;

  // Output results
  void Output(int top_k,                            // number of sequences to return
              int max_length,                       // max sequence length
              gsl::span<int32_t>& sequences,        // buffer with pad token, shape (num_return_sequences, max_length)
              gsl::span<float>& sequences_scores);  // buffer for sequence scores, with shape (num_return_sequences)

 private:
  float length_penalty_;
  bool early_stopping_;
  gsl::span<HypothesisScore> beams_;  // Beam width sized array of hypotheses, sorted by highest scoring
  int beams_used_;                    // Number of elements used in beams_
};

class BeamSearchScorer : public IBeamScorer {
 public:
  BeamSearchScorer(const IGenerationParameters& parameters,
                   AllocatorPtr& allocator);

  void Process(ISequences& sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Finalize(ISequences& sequences,
                gsl::span<const float>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  bool IsDone() const;

  gsl::span<float>& GetNextScores() { return next_beam_scores_; }
  gsl::span<int32_t>& GetNextTokens() { return next_beam_tokens_; }
  gsl::span<int32_t>& GetNextIndices() override { return next_beam_indices_; }

 private:
  size_t batch_size_;
  size_t num_beams_;
  size_t max_length_;
  size_t num_beam_hyps_to_keep_;
  int pad_token_id_;
  int eos_token_id_;

  IAllocatorUniquePtr<bool> done_ptr_;  // Allocated buffer for done_
  gsl::span<bool> done_;                // Flags indicates whether each batch is finished or not. Shape is (batch_size).

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_length_{};                   // Total number of elements
  size_t hypothesis_buffer_offset_{};                   // Offset of available buffer, or length of used buffer.

  IAllocatorUniquePtr<HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  IAllocatorUniquePtr<BeamHypotheses> beam_hyps_ptr_;
  gsl::span<BeamHypotheses> beam_hyps_;  // batch_size_ count
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
