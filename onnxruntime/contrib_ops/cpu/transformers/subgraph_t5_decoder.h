// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "subgraph_base.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for T5 decoder subgraph inputs and outputs preparation
class T5DecoderSubgraph : public Subgraph {
 public:
  T5DecoderSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : Subgraph(node_in, attribute_name, subgraph_in) {}

  // Create inputs for first inference of decoder subgraph.
  Status CreateInitialFeeds(
      const Tensor& encoder_input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int decoder_start_token_id,
      std::vector<OrtValue>& decoder_feeds,
      const std::vector<OrtValue>& encoder_feeds,
      const std::vector<OrtValue>& encoder_fetches,
      IAllocatorUniquePtr<char>& buffer);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
