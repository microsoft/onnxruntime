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
namespace onnxruntime {
namespace contrib {

class ISequences {
public:
  virtual gsl::span<const int64_t> GetSequence(int beam_index) = 0;
  virtual int GetSequenceLength() = 0;
};

// Interface for all scorers for beam search or beam sample.
class IBeamScorer {
 public:
  virtual void Process(ISequences* sequences,
                       gsl::span<const float>& next_scores,
                       gsl::span<const int64_t>& next_tokens,
                       gsl::span<const int64_t>& next_indices,
                       AllocatorPtr& allocator) = 0;

  virtual void Finalize(ISequences* sequences,
                        gsl::span<const float>& final_beam_scores,
                        Tensor* output_sequences,
                        Tensor* output_sequence_scores) = 0;
};

struct HypothesisScore {
  HypothesisScore(gsl::span<const int64_t>& _hypothesis, float _score)
      : hypothesis(_hypothesis), score(_score) {}

  gsl::span<const int64_t> hypothesis;
  float score;
};

class HypothesisScoreCompare {
 public:
  bool operator()(const HypothesisScore& a, const HypothesisScore& b) {
    return a.score > b.score;
  }
};

class BeamHypotheses {
 public:
  BeamHypotheses(int num_beams, float length_penalty, bool early_stopping);

  // Number of hypotheses
  int Size() { return static_cast<int>(beams_.size()); }

  // Add a new hypothesis
  void Add(gsl::span<const int64_t>& hypothesis, float sum_logprobs);

  bool IsDone(float best_sum_logprobs, int current_length);

  // Output results. Note that it will clear all beams.
  void Output(int top_k,                           // number of sequences to return
              int max_length,                      // max sequence length
              gsl::span<int32_t>& sequences,       // buffer filled with pad token ID, with shape (num_return_sequences, max_length)
              gsl::span<float>& sequences_scores); // buffer for sequence scores, with shape (num_return_sequences)

 private:
  int num_beams_;
  float length_penalty_;
  bool early_stopping_;
  float worst_score_;
  std::priority_queue<HypothesisScore, std::vector<HypothesisScore>, HypothesisScoreCompare> beams_;  // min-heap for top k
};

class BeamSearchScorer : public IBeamScorer {
 public:
  BeamSearchScorer(int batch_size,
                   int num_beams,
                   int max_length,
                   float length_penalty,
                   bool early_stopping,
                   int num_return_sequences,
                   int pad_token_id,
                   int eos_token_id);

  bool IsDone();

  void Process(ISequences* sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int64_t>& next_tokens,
               gsl::span<const int64_t>& next_indices,
               AllocatorPtr& allocator) override;

  void Finalize(ISequences* sequences,
                gsl::span<const float>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  gsl::span<float>& GetNextScores() { return next_beam_scores_; }
  gsl::span<int64_t>& GetNextTokens() { return next_beam_tokens_; }
  gsl::span<int64_t>& GetNextIndices() { return next_beam_indices_; }

 private:
  int batch_size_;
  int num_beams_;
  int max_length_;
  int num_beam_hyps_to_keep_;
  int pad_token_id_;
  int eos_token_id_;

  std::vector<BeamHypotheses> beam_hyps;  // List of batch result of beam search. Its shape is (batch_size)
  std::vector<bool> done_;                // List of flags indicates whether each batch is finished or not. Its shape is (batch_size).

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int64_t> next_beam_tokens_ptr_;
  gsl::span<int64_t> next_beam_tokens_;

  IAllocatorUniquePtr<int64_t> next_beam_indices_ptr_;
  gsl::span<int64_t> next_beam_indices_;

  IAllocatorUniquePtr<int64_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int64_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_length_;                     // Total number of elements
  int hypothesis_buffer_offset_;                        // Offset of avaiable buffer, or length of used buffer.
};

}  // namespace contrib
}  // namespace onnxruntime