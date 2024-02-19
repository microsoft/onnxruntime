// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>
#include <random>
#include "core/common/gsl.h"
#include "core/framework/allocator.h"
#include "contrib_ops/cpu/utils/console_dumper.h"

namespace onnxruntime {

namespace concurrency {
class ThreadPool;
}

namespace contrib {
namespace transformers {

template <typename T>
struct IBeamSearchState {
  gsl::span<T> next_token_logits;      // shape (batch_size * num_beams, vocab_size)
  gsl::span<float> next_token_scores;  // shape (batch_size, num_beams * vocab_size)
  gsl::span<int32_t> next_tokens;      // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_indices;     // shape (batch_size, 2 * num_beams)
  gsl::span<float> next_scores;        // shape (batch_size, 2 * num_beams)
  gsl::span<int32_t> next_positions;   // shape (batch_size, num_beams), empty for T5. Next position for position_ids.
  gsl::span<float> beam_scores;        // shape (batch_size, num_beams)
  gsl::span<float> scores;             // shape (max_length - sequence_length + 1, batch_size, num_beams * vocab_size)
  gsl::span<float> remaining_scores;   // portion of scores that is available for appending next token scores.
  gsl::span<float> topk_buffer;        // temp buffer for topk computation, including:
                                       // 1st stage needs:
                                       //   temp score: (batch_size * num_beams * parts_vocab, 2 * num_beams)
                                       //   temp token: (batch_size * num_beams * parts_vocab, 2 * num_beams)
                                       // 2nd stage needs:
                                       //   temp score: (batch_size * num_beams, 2 * num_beams)
                                       //   temp token: (batch_size * num_beams, 2 * num_beams)
                                       // in total, it will be:
                                       // 2 * (batch_size * num_beams * (parts_vocab + 1), 2 * num_beams)

  gsl::span<int32_t> sequences_device;  // shape (2 * batch_size * max_length)

  Tensor staging_for_past_state_reorder;  // Tensor of shape (batch_size * num_beams, num_heads, max_length, head_size)
};

struct IBeamSearchCpuState {
  gsl::span<int32_t> sequence_lengths;  // shape (batch_size, num_beams), initial sequence length
  gsl::span<int32_t> sequences_space;   // shape (2, batch_size, num_beams, max_seq_length)

  // The following are used only by CUDA operator for data copied from device.
  gsl::span<float> topk_scores;        // shape (batch_size, 2*num_beams), scores of topk candidates (K=2*num_beams).
  gsl::span<int32_t> topk_tokens;      // shape (batch_size, 2*num_beams), tokens of topk candidates.
  gsl::span<int32_t> topk_indices;     // shape (batch_size, 2*num_beams), beam indices of topk candidates.
  gsl::span<float> final_beam_scores;  // shape (batch_size, num_beams)
  gsl::span<float> next_token_scores;  // shape (batch_size, num_beams * vocab_size)
};

template <typename T>
struct IGreedySearchState {
  gsl::span<int32_t> sequences_space;          // shape (2, batch_size, max_length)
  gsl::span<int32_t> sequence_lengths;         // shape (batch_size)
  gsl::span<int32_t> next_positions;           // shape (batch_size, num_beams). Next position value for position_ids.
  gsl::span<bool> eos_meet;                    // shape (batch_size)
  gsl::span<T> next_token_scores;              // shape (batch_size, vocab_size)
  gsl::span<int32_t> next_tokens;              // shape (batch_size)
  gsl::span<T> temp_topk_scores_buffer;        // shape (batch_size, parts_of_vocab), temp buffer for topk stage 1 (GPU only)
  gsl::span<int32_t> temp_topk_tokens_buffer;  // shape (batch_size, parts_of_vocab), temp buffer for topk stage 1(GPU only)
  gsl::span<T> topk_scores_buffer;             // shape (batch_size), output buffer for topk stage 2 (GPU only)
  gsl::span<int32_t> topk_tokens_buffer;       // shape (batch_size), output buffer for topk stage 2 (GPU only)
  Tensor staging_for_past_state_reorder;       // Tensor of shape (batch_size * num_beams(1), num_heads, max_length, head_size)
};

template <typename T>
struct ISamplingState {
  gsl::span<int> d_index_in;
  gsl::span<int> d_index_out;
  gsl::span<int> d_offset;
  gsl::span<T> d_sorted_score;
  gsl::span<float> d_sorted_softmaxed_score;
  gsl::span<float> d_softmaxed_score;
  gsl::span<float> h_softmaxed_score;
  gsl::span<float> d_sampled;
  gsl::span<float> h_sampled_all;
  gsl::span<int32_t> d_indices;
  gsl::span<int> d_presence_mask;

  BufferUniquePtr storage_buffer;
  size_t temp_storage_bytes;
  std::default_random_engine generator;

  gsl::span<T> sorted_scores;
  gsl::span<T> cumulative_probs;
};

struct ISequences {
  virtual ~ISequences() {}
  virtual gsl::span<const int32_t> GetSequence(int beam_index) const = 0;
  virtual gsl::span<const int32_t> GetCurrentDeviceSequences() const = 0;  // Get all current beam_index sequences in one continuous block (to pass to CUDA)
  virtual gsl::span<int32_t> GetNextDeviceSequences() = 0;                 // Get all next beam_index sequences in one continuous block (to pass to CUDA)
  virtual int GetSequenceLength() const = 0;
};

struct ILogitsProcessorList {
  virtual ~ILogitsProcessorList() {}
  virtual void Process(const ISequences* sequences, gsl::span<float>& next_token_scores, int step) = 0;
};

// Interface for all scorers for beam search or beam sample.
struct IBeamScorer {
  virtual ~IBeamScorer() {}

  virtual void Process(ISequences& sequences,
                       gsl::span<const float>& next_scores,
                       gsl::span<const int32_t>& next_tokens,
                       gsl::span<const int32_t>& next_indices) = 0;

  virtual void Finalize(ISequences& sequences,
                        gsl::span<const float>& final_beam_scores,
                        Tensor* output_sequences,
                        Tensor* output_sequence_scores) = 0;

  virtual void OutputScores(gsl::span<const float>& final_scores,
                            Tensor* output_scores) = 0;

  virtual bool IsDone() const = 0;                    // GPU version will return false here, as it asynchronously queues up the event
  virtual bool IsDoneLater() const { return false; }  // GPU version waits for the asynchous result to complete here

  virtual gsl::span<float> GetNextScores() = 0;
  virtual gsl::span<int32_t> GetNextTokens() = 0;
  virtual gsl::span<int32_t> GetNextIndicesCPU() = 0;
  virtual gsl::span<int32_t> GetNextIndicesGPU() { return {}; }  // If this is non CPU, returns the device buffer of the indices
};

struct IGenerationParameters {
  static constexpr int kModelTypeGpt = 0;
  static constexpr int kModelTypeT5 = 1;
  static constexpr int kModelTypeWhisper = 2;

  static constexpr int kLogitsProcessorTypeWhisper = 1;

  // Parameters from node attributes
  int model_type;  // 0 for GPT-2; 1 for encoder-decoder like T5; 2 for float inputs like Whisper
  int eos_token_id;
  int pad_token_id;
  int decoder_start_token_id;
  int no_repeat_ngram_size;
  bool early_stopping;

  // Parameters from inputs
  int min_length;
  int max_length;
  int num_beams;
  int num_return_sequences;
  float length_penalty;
  float repetition_penalty;
  int batch_size;       // deduce from first dimension of input_ids
  int sequence_length;  // deduce from second dimension of input_ids of GPT-2 or decoder_input_ids of T5
  int logits_processor;

  gsl::span<const int32_t> vocab_mask;
  gsl::span<const int32_t> prefix_vocab_mask;
  gsl::span<const int32_t> presence_mask;

  // Parameters from outputs.
  bool output_scores;  // whether scores existed in output

  // Parameters from subgraph.
  int vocab_size;
  int num_heads;
  int head_size;
  int num_layers;

  // Parameters for TopK/TopP sampling.
  float presence_penalty;
  float filter_value;
  float temperature = 1.0f;
  float top_p = 0.0f;
  int seed = 0;
  int min_tokens_to_keep = 1;
  bool custom_sampling = false;

  // Parameters for whisper model
  bool decoder_output_cross_qk = false;
  gsl::span<const int32_t> extra_decoding_ids;

  // Token ids are defined below in the order that they appear in the tokenizer
  int32_t translate_token_id = -1;
  int32_t transcribe_token_id = -1;
  int32_t start_of_lm_token_id = -1;
  int32_t no_speech_token_id = -1;
  int32_t no_timestamps_token_id = -1;
  int32_t beginning_timestamp_token_id = -1;
  void* no_speech_probs = nullptr;

  int cross_qk_layer_head_input_id = -1;
  int extra_decoding_ids_input_id = -1;
  int cross_qk_output_id = -1;
  int no_speech_probs_output_id = -1;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
