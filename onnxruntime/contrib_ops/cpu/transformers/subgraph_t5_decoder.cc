// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"
#include "contrib_ops/cpu/transformers/sequences.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* T5 Decoder Subgraph.

   Inputs:
      input_ids: int32 (B, 1)
      encoder_attention_mask: int32 (B, encode_sequence_length)
      encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)

      past_key_self_0: (B, num_heads, past_decode_sequence_length, head_size)
      past_value_self_0: (B, num_heads, past_decode_sequence_length, head_size)
      ... (for each self attention layer)

      past_key_cross_0: (B, num_heads, encode_sequence_length, head_size)
      past_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
      ... (for each cross attention layer)

    Outputs:
      logits: (B, 1, vocab_size)

      present_key_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
      present_value_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
      ... (for each self attention layer)

    Note:
      B = batch_size * num_beams
      Data type of input or output is float or float16 if not specified.
*/

Status T5DecoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                   const std::vector<const NodeArg*>& subgraph_outputs) {
  bool has_hidden_state = subgraph_inputs[2]->Name() == "encoder_hidden_states" ? true : false;
  SetPastInputIndex(has_hidden_state);

  ORT_RETURN_IF(first_past_input_index_ != 2 && first_past_input_index_ != 3,
                "kFirstPastInputIndex currently only supports 2 or 3");
  ORT_RETURN_IF(num_subgraph_inputs < 4 + first_past_input_index_ ||
                    (num_subgraph_inputs - first_past_input_index_) % 4 != 0,
                "number of outputs expected to be kFirstPastInputIndex + 4 * layers, got:", num_subgraph_inputs);
  ORT_RETURN_IF(num_subgraph_outputs < 3 || (num_subgraph_outputs - first_present_output_index_) % 2 != 0,
                "number of outputs expected to be 1 + 2 * layers, got:", num_subgraph_outputs);

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids",
                "decoder subgraph input 0 shall be named as input_ids, got: ", subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "encoder_attention_mask",
                "decoder subgraph input 1 shall be named as encoder_attention_mask, got: ",
                subgraph_inputs[1]->Name());
  if (first_past_input_index_ == 3) {
    ORT_RETURN_IF(subgraph_inputs[2]->Name() != "encoder_hidden_states",
                  "decoder subgraph input 2 shall be named as encoder_hidden_states, got: ",
                  subgraph_inputs[2]->Name());
  }

  // check subgraph outputs
  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits",
                "decoder subgraph output 0 shall be named as logits, got: ", subgraph_outputs[0]->Name());

  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();
  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_outputs[first_present_output_index_]->Shape();

  // Save parameters related to the subgraph.
  ORT_RETURN_IF_ERROR(GetParameters(past_shape, logits_shape, false));
  num_layers = (static_cast<int>(subgraph_outputs.size()) - first_present_output_index_) / 2;

  // If input_ids's shape is ['batch_size', 1] then use next token as input_ids.
  // Otherwise in the case of shape ['batch_size', 'sequence'], use sequence as input_ids.
  const ONNX_NAMESPACE::TensorShapeProto* input_ids_shape = subgraph_inputs[0]->Shape();
  if (input_ids_shape->dim(1).has_dim_value() && input_ids_shape->dim(1).dim_value() == 1) {
    use_sequence_as_input_ids_ = false;
  }

  constexpr auto int32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
  constexpr auto float32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
  constexpr auto float16_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "decoder subgraph input 0 (input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "decoder subgraph input 1 (encoder_attention_mask) shall have int32 type");

  auto float_type = subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(float_type != float32_type && float_type != float16_type,
                "decoder subgraph input 2 (encoder_hidden_states) shall have float or float16 type");

  for (int i = first_past_input_index_; i < num_subgraph_inputs; i++) {
    ORT_RETURN_IF(subgraph_inputs[i]->TypeAsProto()->tensor_type().elem_type() != float_type,
                  "decoder subgraph past inputs shall have same data type as that of encoder_hidden_states");
  }

  for (int i = 0; i < num_subgraph_outputs; i++) {
    ORT_RETURN_IF(subgraph_outputs[i]->TypeAsProto()->tensor_type().elem_type() != float_type,
                  "decoder subgraph output shall have same data type as that of encoder_hidden_states");
  }

  is_output_float16_ = (subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type() == float16_type);

  return Status::OK();
}

// Create inputs for decoder from the following data sources:
// encoder feeds: encoder_input_ids, encoder_attention_mask, decoder_input_ids (with start tokens)
// encoder fetches: logits,
//                  encoder_hidden_states,
//                  present_key_self_0, present_value_self_0, ..., present_key_cross_0, present_value_cross_0, ...
// decoder_feeds: input_ids,
//                encoder_attention_mask,
//                encoder_hidden_states,
//                present_key_self_0, present_value_self_0, ..., present_key_cross_0, present_value_cross_0, ...
Status T5DecoderSubgraph::CreateInitialFeeds(
    gsl::span<const int32_t> beam_next_tokens,
    const std::vector<const OrtValue*>& implicit_inputs,
    const std::vector<OrtValue>& encoder_feeds,
    const std::vector<OrtValue>& encoder_fetches,
    std::vector<OrtValue>& decoder_feeds,
    const GenerationDeviceHelper::DeviceCopyFunc<int32_t>& device_copy_int32_func,
    const GenerationDeviceHelper::ExpandBufferFunc<int32_t>& expand_buffer_int32_func,
    const GenerationDeviceHelper::ExpandBufferFunc<float>& expand_buffer_float_func,
    const GenerationDeviceHelper::ExpandBufferFunc<MLFloat16>& expand_buffer_float16_func,
    int num_beam,
    Stream* stream,
    bool use_sequence_as_input_ids,
    int cur_len,
    transformers::Sequences& sequences) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  // Allocate subgraph inputs from same device as inputs of encoder subgraph.
  AllocatorPtr allocator = session_state_->GetAllocator(encoder_feeds[0].Get<Tensor>().Location());

  // Copy beam next tokens in CPU to input_ids in provider device (CPU for CPU EP, or GPU for CUDA EP).
  int batch_beam_size = static_cast<int>(beam_next_tokens.size());
  int sequence_length = !use_sequence_as_input_ids ? 1 : cur_len;
  int64_t dims[] = {batch_beam_size, sequence_length};
  TensorShape input_ids_shape(&dims[0], 2);
  OrtValue input_ids;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), input_ids_shape, allocator, input_ids);
  int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();

  if (!use_sequence_as_input_ids_) {
    ORT_RETURN_IF_ERROR(device_copy_int32_func(
        input_ids.GetMutable<Tensor>()->MutableDataAsSpan<int32_t>(),
        beam_next_tokens,
        stream,
        DeviceCopyDirection::hostToDevice));
  } else {
    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<const int32_t> sequence = sequences.GetSequence(i);
      const int32_t* sequence_data = sequence.data();
      for (int j = 0; j < cur_len; j++) {
        input_ids_data[i * cur_len + j] = sequence_data[j];
      }
    }
  }

  // The ordering is the same as used in Setup.
  decoder_feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));
  decoder_feeds.push_back(input_ids);

  // The encoder_attention_mask is copied from the second input of encoder.
  OrtValue expanded_decoder_attention_masks;
  ORT_RETURN_IF_ERROR(expand_buffer_int32_func(stream,
                                               encoder_feeds[1],
                                               num_beam,
                                               allocator,
                                               expanded_decoder_attention_masks,
                                               false));

  decoder_feeds.push_back(expanded_decoder_attention_masks);

  // When first_past_input_index_ == 3, the encoder_hidden_states and past states are copied from the second output
  // of encoder.
  // When first_past_input_index_ == 2, the past states are copied from the second output of encoder.
  for (size_t j = static_cast<size_t>(4) - first_past_input_index_; j < encoder_fetches.size(); j++) {
    if (j == 1) {
      ORT_RETURN_IF(has_hidden_state_ == false, "Invalid hidden_states expension: has_hidden_state_ == false");
      OrtValue expanded_hidden_states;
      if (is_output_float16_) {
        ORT_RETURN_IF_ERROR(expand_buffer_float16_func(stream,
                                                       encoder_fetches[j],
                                                       num_beam,
                                                       allocator,
                                                       expanded_hidden_states,
                                                       true));
      } else {
        ORT_RETURN_IF_ERROR(expand_buffer_float_func(stream,
                                                     encoder_fetches[j],
                                                     num_beam,
                                                     allocator,
                                                     expanded_hidden_states,
                                                     true));
      }
      decoder_feeds.push_back(expanded_hidden_states);
    } else {
      OrtValue expanded_cache;
      if (is_output_float16_) {
        ORT_RETURN_IF_ERROR(expand_buffer_float16_func(stream,
                                                       encoder_fetches[j],
                                                       num_beam,
                                                       allocator,
                                                       expanded_cache,
                                                       false));
      } else {
        ORT_RETURN_IF_ERROR(expand_buffer_float_func(stream,
                                                     encoder_fetches[j],
                                                     num_beam,
                                                     allocator,
                                                     expanded_cache,
                                                     false));
      }
      decoder_feeds.push_back(expanded_cache);
    }
  }

  // Pass through implicit inputs.
  for (const auto* entry : implicit_inputs) {
    decoder_feeds.push_back(*entry);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
