// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "gsl/gsl"
#include "core/framework/allocator.h"
#include "core/framework/feeds_fetches_manager.h"
#include "contrib_ops/cpu/transformers/beam_search_device_helper.h"

namespace onnxruntime {
class SessionState;
}

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
  Status CreateInitialFeeds(
      const Tensor& input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int pad_token_id,
      gsl::span<int32_t>& sequence_lengths,
      OrtValue& expanded_input_ids,
      std::vector<OrtValue>& feeds,
      const BeamSearchDeviceHelper::CreateInputsFunc& create_inputs_func,
      const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer);

  FeedsFetchesManager* GetFeedsFetchesManager() const { return feeds_fetches_manager_.get(); }

  const IExecutionProvider* GetProvider() const;

  bool IsOutputFloat16() const { return is_output_float16_; }

 protected:
  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs);

  AllocatorPtr allocator_;
  const SessionState* session_state_;
  const SessionState* subgraph_session_state_;
  std::unique_ptr<FeedsFetchesManager> feeds_fetches_manager_;
  bool is_output_float16_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
