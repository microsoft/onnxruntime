// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/subgraph_base.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_encoder.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for whisper encoder subgraph with validation to support float inputs.
class WhisperEncoderSubgraph : public T5EncoderSubgraph {
 public:
  WhisperEncoderSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : T5EncoderSubgraph(node_in, attribute_name, subgraph_in) {}

  // Create inputs for first inference of subgraph.
  Status CreateInitialFeeds(
      const Tensor& encoder_input_ids,
      // const OrtValue* attn_mask_value,
      const Tensor& original_decoder_input_ids,
      const std::vector<const OrtValue*>& implicit_inputs,
      // int pad_token_id,
      std::vector<OrtValue>& feeds,
      const GenerationDeviceHelper::CreateWhisperEncoderInputsFunc& create_encoder_inputs_func,
      const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
      IAllocatorUniquePtr<char>& buffer,
      OrtValue& expanded_decoder_input_ids,
      Stream* ort_stream);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;

};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
