// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <functional>
#include "gsl/gsl"
#include "core/common/common.h"
#include "core/framework/feeds_fetches_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "beam_search_parameters.h"
#include "beam_search_scorer.h"

namespace onnxruntime {
namespace contrib {

struct GptSubgraphInfo {
  GptSubgraphInfo(const onnxruntime::Node& node, const GraphViewer& subgraph_in);

  const GraphViewer& subgraph;

  int num_implicit_inputs;

  int num_subgraph_inputs;   // same as subgraph_input_names.size(), keep it for convenience.
  int num_subgraph_outputs;  // same as subgraph_output_names.size()

  std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;
};

// This class keeps track of sequences generated.
class Sequences : public ISequences {
 public:
  Sequences() {}

  // Initialize the sequence with initial input_ids and related parameters.
  void Init(const OrtValue& input_ids, int batch_beam_size, int sequence_length, int max_length);

  // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
  gsl::span<const int64_t> GetSequence(int beam_index) override;

  // Returns current sequence length.
  int GetSequenceLength() override;

  // Print the sequences to StdOut in debug mode
  void PrintSequences();

  // Select sequences based on beam indices, then append next token to selected sequences.
  void AppendNextTokenToSequences(
      gsl::span<int64_t>& beam_indices,
      gsl::span<int64_t>& beam_next_tokens);

 private:
  // Two buffers of shape (batch_size, num_beams, max_seq_length) to store sequences.
  // At each time, there is only one buffer is active. The other one will be active in next token.
  // Each AppendNextTokenToSequences call will trigger a rotation of active buffer.
  std::vector<int64_t> sequences[2];

  // Index (either 0 or 1) of two buffers that is currently is active.
  int current_sequences_buffer;

  int batch_beam_size_;
  int max_length_;
  int current_length_;
};

template <typename T>
struct BeamSearchState {
  // TODO: use allocater to allocate a buffer, and point each data to a span of the buffer
  //       so as to reuse related code in CUDA.
  std::vector<bool> done;      // shape (batch_size)
  std::vector<T> beam_scores;  // shape (batch_size, num_beams)

  std::vector<T> next_token_logits;  // shape (batch_size * num_beams, vocab_size)
  std::vector<T> next_token_scores;  // shape (batch_size, num_beams * vocab_size)

  std::vector<int64_t> next_tokens;   // shape (batch_size, num_beams)
  std::vector<int64_t> next_indices;  // shape (batch_size, num_beams)

  Sequences sequences;

  std::vector<T> scores;  // shape (max_length - sequence_length + 1, batch_size, num_beams * vocab_size)

  void Init(const OrtValue& input_ids, int batch_size, int num_beams, int vocab_size, int sequence_length, int max_length, bool output_scores) {
    int batch_beam_size = batch_size * num_beams;
    done.assign(batch_size, 0);
    beam_scores.assign(batch_beam_size, 0.0f);
    for (int i = 0; i < batch_size; i++) {
      for (int j = 1; j < num_beams; j++) {
        beam_scores[i * num_beams + j] = -1e9;
      }
    }

    next_token_logits.assign(batch_beam_size * vocab_size, 0.0f);
    next_token_scores.assign(batch_beam_size * vocab_size, 0.0f);

    next_tokens.assign(batch_beam_size, 0);
    next_indices.assign(batch_beam_size, 0);

    sequences.Init(input_ids, batch_beam_size, sequence_length, max_length);

    if (output_scores) {
      scores.reserve((max_length - sequence_length) * batch_size * num_beams * vocab_size);
    }
  }
};

template <typename T>
class BeamSearch : public controlflow::IControlFlowKernel {
 public:
  BeamSearch(const OpKernelInfo& info) : IControlFlowKernel(info) { Init(info); }
  void Init(const OpKernelInfo& info);

  Status Compute(OpKernelContext* ctx) const override;

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  static std::unique_ptr<OpKernel> Create(const OpKernelInfo& info, void* stream);

 protected:
  Status CheckSubgraph(const std::vector<const NodeArg*>& subgraph_inputs,
                       const std::vector<const NodeArg*>& subgraph_outputs);

  void SetComputeStream(void* stream) { stream_ = stream; }

 private:
  // Subgraph info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<GptSubgraphInfo> subgraph_info_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;

  void* stream_;

  BeamSearchParameters parameters_;
};

}  // namespace contrib
}  // namespace onnxruntime
