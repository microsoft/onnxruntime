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
#include "gpt_subgraph.h"
#include "sequences.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

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

  void SetComputeStream(void* stream) { stream_ = stream; }

 private:
  // Subgraph info and FeedsFetchesManager re-used for each subgraph execution.
  std::unique_ptr<GptSubgraph> gpt_subgraph_;
  FeedsFetchesManager* feeds_fetches_manager_;

  void* stream_;

  BeamSearchParameters parameters_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
