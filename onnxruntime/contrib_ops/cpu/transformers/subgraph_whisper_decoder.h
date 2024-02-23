// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cpu/transformers/subgraph_base.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

// A class for Whisper decoder subgraph inputs and outputs preparation.
class WhisperDecoderSubgraph : public T5DecoderSubgraph {
 public:
  WhisperDecoderSubgraph(
      const onnxruntime::Node& node_in,
      const std::string& attribute_name,
      const GraphViewer& subgraph_in) : T5DecoderSubgraph(node_in, attribute_name, subgraph_in) {
    first_past_input_index_ = 1;
  }

  // Create inputs for first inference of decoder subgraph.
  Status CreateInitialFeeds(
      AllocatorPtr cpu_allocator,
      gsl::span<const int32_t> beam_next_tokens,
      const std::vector<const OrtValue*>& implicit_inputs,
      const std::vector<OrtValue>& encoder_feeds,
      const std::vector<OrtValue>& encoder_fetches,
      std::vector<OrtValue>& decoder_feeds,
      const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
      const GenerationDeviceHelper::ExpandBufferFunc<float>& expand_buffer_float_func,
      const GenerationDeviceHelper::ExpandBufferFunc<MLFloat16>& expand_buffer_float16_func,
      int num_beam,
      Stream* stream,
      bool use_sequence_as_input_ids,
      int cur_len,
      transformers::Sequences& sequences,
      int past_present_share_buffer_max_seq_len = -1,
      bool need_cache_indir = false);

  Status Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                  const std::vector<const NodeArg*>& subgraph_outputs) override;

  void SetPastInputIndex(bool has_hidden_state) {
    has_hidden_state_ = has_hidden_state;
    if (!has_hidden_state_) {
      first_past_input_index_ = 1;
    } else {
      first_past_input_index_ = 2;
    }
  }
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
