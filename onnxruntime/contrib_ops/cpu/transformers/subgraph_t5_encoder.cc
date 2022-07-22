// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "gsl/gsl"
#include "contrib_ops/cpu/transformers/subgraph_t5_encoder.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* T5 Encoder Subgraph (It also contains decoder initialization where decoder_input_ids are filled with start token ID).

   Inputs:
      encoder_input_ids: int32 (B, encode_sequence_length)
      encoder_attention_mask: int32 (B, encode_sequence_length)
      decoder_input_ids: int32 (B, 1)

    Outputs:
      logits: (B, 1, vocab_size)
      encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)

      present_key_self_0: (B, num_heads, 1, head_size)
      present_value_self_0: (B, num_heads, 1, head_size)
      ... (for each self attention layer)

      present_key_cross_0: (B, num_heads, encode_sequence_length, head_size)
      present_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
      ... (for each cross attention layer)

    Note:
      Here, B = batch_size * num_beams since we expand the inputs.
      Ideally, we could use B=batch_size and expand the outputs with a factor of num_beams.
      Data type of input or output is float or float16 if not specified.
*/

Status T5EncoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                   const std::vector<const NodeArg*>& subgraph_outputs) {
  ORT_RETURN_IF(num_subgraph_inputs != 3, "expect 3 inputs, got:", num_subgraph_inputs);

  ORT_RETURN_IF(num_subgraph_outputs < 6, "expect >=6 outputs, got:", num_subgraph_outputs);
  ORT_RETURN_IF((static_cast<int>(subgraph_outputs.size()) - first_present_output_index_) % 4 != 0,
                "number of outputs expected to be 2 + 4 * layers, got:", num_subgraph_outputs);

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "encoder_input_ids",
                "encoder subgraph input 0 shall be named as encoder_input_ids, got: ", subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "encoder_attention_mask",
                "encoder subgraph input 1 shall be named as encoder_attention_mask, got: ", subgraph_inputs[1]->Name());
  ORT_RETURN_IF(subgraph_inputs[2]->Name() != "decoder_input_ids",
                "encoder subgraph input 2 shall be named as decoder_input_ids, got: ", subgraph_inputs[2]->Name());

  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits",
                "encoder subgraph output 0 shall be named as logits, got: ", subgraph_outputs[0]->Name());
  ORT_RETURN_IF(subgraph_outputs[1]->Name() != "encoder_hidden_states",
                "encoder subgraph output 1 shall be named encoder_hidden_states, got: ", subgraph_outputs[1]->Name());
  ORT_RETURN_IF(subgraph_outputs[2]->Name() != "present_key_self_0",
                "encoder subgraph output 2 shall be named as present_key_self_0, got: ", subgraph_outputs[2]->Name());
  ORT_RETURN_IF(subgraph_outputs[3]->Name() != "present_value_self_0",
                "encoder subgraph output 3 shall be named as present_value_self_0, got: ", subgraph_outputs[3]->Name());

  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_outputs[2]->Shape();
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();

  // Save parameters related to the subgraph.
  ORT_RETURN_IF_ERROR(GetParameters(past_shape, logits_shape, false));
  num_layers = (static_cast<int>(subgraph_outputs.size()) - first_present_output_index_) / 4;

  constexpr auto int32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
  constexpr auto float32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
  constexpr auto float16_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "encoder subgraph input 0 (encoder_input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "encoder subgraph input 1 (encoder_attention_mask) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "encoder subgraph input 2 (decoder_input_ids) shall have int32 type");

  auto output_type = subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(output_type != float32_type && output_type != float16_type,
                "encoder subgraph output 0 (logits) shall be float or float16 data type");

  for (int i = 1; i < num_subgraph_outputs; i++) {
    ORT_RETURN_IF(subgraph_outputs[i]->TypeAsProto()->tensor_type().elem_type() != output_type,
                  "encoder subgraph outputs 1, 2, ... shall have same data type");
  }

  is_output_float16_ = (output_type == float16_type);

  return Status::OK();
}

// Create inputs for first inference of subgraph.
Status T5EncoderSubgraph::CreateInitialFeeds(
    const Tensor& original_encoder_input_ids,
    const OrtValue* attn_mask_value,
    const std::vector<const OrtValue*>& implicit_inputs,
    int pad_token_id,
    int start_token_id,
    std::vector<OrtValue>& feeds,
    const GenerationDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
    const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
    IAllocatorUniquePtr<char>& buffer,
    OrtValue& decoder_input_ids) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  // The ordering is the same as used in Setup.
  feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // Allocate subgraph inputs to be same device as encoder_input_ids.
  AllocatorPtr cpu_allocator = session_state_->GetAllocator(original_encoder_input_ids.Location());
  if (cpu_allocator == nullptr) {
    const IExecutionProvider* provider = GetProvider();
    cpu_allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  }
  ORT_RETURN_IF(cpu_allocator == nullptr, "cpu_allocator shouldn't be nullptr");

  // TODO(tianleiwu): expand the outputs instead of inputs to save computation.
  OrtValue encoder_input_ids;
  OrtValue encoder_attention_mask;
  ORT_RETURN_IF_ERROR(create_encoder_inputs_func(&original_encoder_input_ids,
                                                 attn_mask_value,
                                                 pad_token_id,
                                                 start_token_id,
                                                 cpu_allocator,
                                                 encoder_input_ids,
                                                 encoder_attention_mask,
                                                 decoder_input_ids));

  const IExecutionProvider* provider = GetProvider();
  ORT_RETURN_IF_ERROR(add_to_feeds_func(
      provider,
      {encoder_input_ids, encoder_attention_mask, decoder_input_ids},
      feeds,
      buffer));

  for (const auto* entry : implicit_inputs) {
    feeds.push_back(*entry);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
