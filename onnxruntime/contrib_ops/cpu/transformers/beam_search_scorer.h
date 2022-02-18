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
#include "sequences.h"
#include "beam_search_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
struct HypothesisScore {
  HypothesisScore(gsl::span<const int32_t>& _hypothesis, T _score)
      : hypothesis(_hypothesis), score(_score) {}

  gsl::span<const int32_t> hypothesis;
  T score;
};

template <typename T>
class HypothesisScoreCompare {
 public:
  bool operator()(const HypothesisScore<T>& a, const HypothesisScore<T>& b) {
    return a.score > b.score;
  }
};

template <typename T>
class BeamHypotheses {
 public:
  BeamHypotheses() = default;

  void Init(int num_beams, T length_penalty, bool early_stopping);

  // Number of hypotheses
  int Size() { return static_cast<int>(beams_.size()); }

  // Add a new hypothesis
  void Add(gsl::span<const int32_t>& hypothesis, T sum_logprobs);

  bool IsDone(T best_sum_logprobs, int current_length);

  // Output results. Note that it will clear all beams.
  void Output(int top_k,                        // number of sequences to return
              int max_length,                   // max sequence length
              gsl::span<int32_t>& sequences,    // buffer filled with pad token ID, with shape (num_return_sequences, max_length)
              gsl::span<T>& sequences_scores);  // buffer for sequence scores, with shape (num_return_sequences)

 private:
  int num_beams_;
  T length_penalty_;
  bool early_stopping_;
  T worst_score_;
  std::priority_queue<HypothesisScore<T>, std::vector<HypothesisScore<T>>, HypothesisScoreCompare<T>> beams_;  // min-heap for top k
};

template <typename T>
class BeamSearchScorer : public IBeamScorer {
 public:
  BeamSearchScorer(size_t batch_size,
                   size_t num_beams,
                   size_t max_length,
                   T length_penalty,
                   bool early_stopping,
                   size_t num_return_sequences,
                   int pad_token_id,
                   int eos_token_id);

  void Initialize(AllocatorPtr& allocator, int sequence_length) override;

  void Process(ISequences* sequences,
               gsl::span<const T>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Finalize(ISequences* sequences,
                gsl::span<const T>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  bool IsDone();

  gsl::span<T>& GetNextScores() { return next_beam_scores_; }
  gsl::span<int32_t>& GetNextTokens() { return next_beam_tokens_; }
  gsl::span<int32_t>& GetNextIndices() { return next_beam_indices_; }

 private:
  size_t batch_size_;
  size_t num_beams_;
  size_t max_length_;
  T length_penalty_;
  bool early_stopping_;
  size_t num_beam_hyps_to_keep_;
  int pad_token_id_;
  int eos_token_id_;

  IAllocatorUniquePtr<BeamHypotheses<T>> beam_hyps_ptr_; // Allocated buffer for beam_hyps_
  gsl::span<BeamHypotheses<T>> beam_hyps_;               // List of batch result of beam search. Its shape is (batch_size)

  IAllocatorUniquePtr<bool> done_ptr_;  // Allocated buffer for done_
  gsl::span<bool> done_;                // List of flags indicates whether each batch is finished or not. Its shape is (batch_size).

  IAllocatorUniquePtr<T> next_beam_scores_ptr_;
  gsl::span<T> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  size_t hypothesis_buffer_length_;                     // Total number of elements
  size_t hypothesis_buffer_offset_;                     // Offset of avaiable buffer, or length of used buffer.
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime