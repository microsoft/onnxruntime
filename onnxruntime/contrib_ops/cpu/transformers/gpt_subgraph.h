// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
//#include <functional>
#include "gsl/gsl"
#include "core/framework/allocator.h"
#include "core/framework/session_state.h"
#include "core/framework/feeds_fetches_manager.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for GPT-2 subgraph inputs and outputs preparation.
struct GptSubgraph {
  GptSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in);

  const onnxruntime::Node& node;  // node that contains the subgraph
  const std::string& attribute;   // attribute of th node that contains the subgraph. Not used yet.
  const GraphViewer& subgraph;    // the subgraph

  int num_implicit_inputs;

  int num_subgraph_inputs;   // same as subgraph_input_names.size(), keep it for convenience.
  int num_subgraph_outputs;  // same as subgraph_output_names.size()

  std::vector<std::string> subgraph_input_names;
  std::vector<std::string> subgraph_output_names;

  // Parameters deduced from the subgraph
  int num_heads;
  int head_size;
  int vocab_size;
  int num_layers;

  // Setup exectuion
  Status Setup(const SessionState& session_state,
               const SessionState& subgraph_session_state);

  // Create inputs for first inference of subgraph.
  void CreateInitialFeeds(
      const Tensor& input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int pad_token_id,
      gsl::span<int64_t>& next_positions,
      std::vector<OrtValue>& feeds);

  Status UpdateFeeds(
      const std::vector<OrtValue>& last_outputs,
      std::vector<OrtValue>& next_inputs,
      int current_length,
      gsl::span<int64_t>& next_positions,
      gsl::span<const int64_t> beam_next_tokens,
      gsl::span<const int64_t> beam_indices,
      int num_beams);

  FeedsFetchesManager* GetFeedsFetchesManager() const { return feeds_fetches_manager_.get(); }

 protected:
  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs);

  OrtValue ExpandInputs(const OrtValue& input, int num_beams) const;

  void PickPastState(const std::vector<OrtValue>& last_outputs,
                     std::vector<OrtValue>& next_inputs,
                     gsl::span<const int64_t>& beam_indices);

  AllocatorPtr allocator_;
  const SessionState* session_state_;
  const SessionState* subgraph_session_state_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
