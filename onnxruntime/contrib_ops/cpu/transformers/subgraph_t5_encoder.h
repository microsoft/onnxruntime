// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/subgraph_base.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for T5 encoder subgraph inputs and outputs preparation.
class T5EncoderSubgraph : public Subgraph {
 public:
  T5EncoderSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : Subgraph(node_in, attribute_name, subgraph_in) {}

  // Create inputs for first inference of subgraph.
  Status CreateInitialFeeds(
      const Tensor& encoder_input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      int num_beams,
      int pad_token_id,
      int start_token_id,
      std::vector<OrtValue>& feeds,
      const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
      const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer,
      OrtValue& expanded_decoder_input_ids);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;

  constexpr static int kFirstPresentOutputIndex = 2;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
