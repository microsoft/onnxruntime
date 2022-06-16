// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/subgraph_base.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for GPT-2 subgraph inputs and outputs preparation.
class GptSubgraph : public Subgraph {
 public:
  GptSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : Subgraph(node_in, attribute_name, subgraph_in) {}

  // Create inputs for first inference of subgraph.
  Status CreateInitialFeeds(
      const Tensor& input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int pad_token_id,
      gsl::span<int32_t>& sequence_lengths,
      OrtValue& expanded_input_ids,
      std::vector<OrtValue>& feeds,
      const BeamSearchDeviceHelper::CreateGptInputsFunc& create_gpt_inputs_func,
      const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;

  constexpr static int kFirstPastInputIndex = 3;
  constexpr static int kFirstPresentOutputIndex = 1;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
