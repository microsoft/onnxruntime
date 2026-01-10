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
      const GraphViewer& subgraph_in) : Subgraph(node_in, attribute_name, subgraph_in) {
    has_logits_output_ = num_subgraph_outputs > 0 && subgraph_output_names[0] == "logits";

    // Old format: The first output is logits, the second one is encoder_hidden_states.
    // New format: No logits and encoder_hidden_states. All outputs are cross.
    first_present_output_index_ = HasLogitsOutput() ? 2 : 0;
  }

  // Create inputs for first inference of subgraph.
  Status CreateInitialFeeds(
      const Tensor& encoder_input_ids,
      const OrtValue* attn_mask_value,
      const std::vector<const OrtValue*>& implicit_inputs,
      int pad_token_id,
      int start_token_id,
      std::vector<OrtValue>& feeds,
      const GenerationDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
      const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer,
      OrtValue& expanded_decoder_input_ids,
      Stream* ort_stream);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;

#ifdef DEBUG_GENERATION
  int GetFirstPresentOutputIndex() const {
    return first_present_output_index_;
  }
#endif

  bool HasLogitsOutput() const {
    return has_logits_output_;
  }

 protected:
  bool has_logits_output_;
  int first_present_output_index_;
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
