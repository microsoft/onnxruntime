// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/transformers/subgraph_whisper_decoder.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"
#include "contrib_ops/cpu/transformers/sequences.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* Whisper Decoder Subgraph.

   Inputs:
      input_ids: int32 (B, 1)
      encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size)

      past_key_self_0: (B, num_heads, past_decode_sequence_length, head_size)
      past_value_self_0: (B, num_heads, past_decode_sequence_length, head_size)
      ... (for each self attention layer)

      past_key_cross_0: (B, num_heads, encode_sequence_length, head_size)
      past_value_cross_0: (B, num_heads, encode_sequence_length, head_size)
      ... (for each cross attention layer)

      past_seq_len: int32 (1) - the length of past sequence(optional)
      num_beams: int32 (1) - the number of beams(optional)
      cache_indirection: int32 (B, num_beams, max_seq_length) - the cache indirection(optional)

    Outputs:
      logits: (B, 1, vocab_size)

      present_key_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
      present_value_self_0: (B, num_heads, past_decode_sequence_length + 1, head_size)
      ... (for each self attention layer)

    Note:
      B = batch_size * num_beams
      Data type of input or output is float or float16 if not specified.
*/

Status WhisperDecoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                        const std::vector<const NodeArg*>& subgraph_outputs) {
  bool has_hidden_state = subgraph_inputs[1]->Name() == "encoder_hidden_states" ? true : false;
  SetPastInputIndex(has_hidden_state);

  ORT_RETURN_IF(first_past_input_index_ != 1 && first_past_input_index_ != 2,
                "kFirstPastInputIndex currently only supports 1 or 2");

  if (!past_present_share_buffer_) {
    ORT_RETURN_IF(has_decoder_masked_attention_, "decoder_masked_attention shall use with past_present_share_buffer");
    ORT_RETURN_IF(num_subgraph_inputs < 4 + first_past_input_index_ ||
                      (num_subgraph_inputs - first_past_input_index_) % 4 != 0,
                  "number of inputs expected to be kFirstPastInputIndex + 4 * layers, got:", num_subgraph_inputs);
  } else if (has_decoder_masked_attention_) {
    ORT_RETURN_IF(num_subgraph_inputs < 7 + first_past_input_index_ ||
                      (num_subgraph_inputs - first_past_input_index_ - 3) % 4 != 0,
                  "number of inputs expected to be kFirstPastInputIndex + 4 * layers + 3, got:", num_subgraph_inputs);
  } else {
    ORT_RETURN_IF(num_subgraph_inputs < 5 + first_past_input_index_ ||
                      (num_subgraph_inputs - first_past_input_index_ - 1) % 4 != 0,
                  "number of inputs expected to be kFirstPastInputIndex + 4 * layers + 1, got:", num_subgraph_inputs);
  }

  ORT_RETURN_IF(num_subgraph_outputs < 3 || (num_subgraph_outputs - first_present_output_index_) % 2 != 0,
                "number of outputs expected to be 1 + 2 * layers, got:", num_subgraph_outputs);

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids",
                "decoder subgraph input 0 shall be named as input_ids, got: ", subgraph_inputs[0]->Name());
  if (first_past_input_index_ == 2) {
    ORT_RETURN_IF(subgraph_inputs[1]->Name() != "encoder_hidden_states",
                  "decoder subgraph input 1 shall be named as encoder_hidden_states, got: ",
                  subgraph_inputs[1]->Name());
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

  auto float_type = subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(float_type != float32_type && float_type != float16_type,
                "decoder subgraph input 1 (encoder_hidden_states) shall have float or float16 type");

  for (int i = first_past_input_index_; i < first_past_input_index_ + 4 * num_layers; i++) {
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
// encoder feeds: encoder_input_ids, decoder_input_ids (with start tokens)
// encoder fetches: logits,
//                  encoder_hidden_states,
//                  present_key_self_0, present_value_self_0, ..., present_key_cross_0, present_value_cross_0, ...
// decoder_feeds: input_ids,
//                encoder_hidden_states,
//                present_key_self_0, present_value_self_0, ..., present_key_cross_0, present_value_cross_0, ...
Status WhisperDecoderSubgraph::CreateInitialFeeds(
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
    int past_present_share_buffer_max_seq_len,
    bool need_cache_indir) {
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

  AllocatorPtr buffer_allocator = std::make_shared<onnxruntime::CPUAllocator>();
  size_t total_size = static_cast<size_t>(static_cast<long long>(cur_len) * batch_beam_size * sizeof(int));
  auto seq_copy = IAllocator::MakeUniquePtr<int>(buffer_allocator, total_size);
  int* seq_copy_ptr = seq_copy.get();

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
      long long seq_index = (long long)i * cur_len;
      memcpy(seq_copy_ptr + seq_index, sequence_data, total_size);
    }
    gsl::span<int> temp_input(input_ids_data, total_size);
    gsl::span<int> temp_sequence(seq_copy_ptr, total_size);
    ORT_RETURN_IF_ERROR(device_copy_int32_func(
        temp_input,
        temp_sequence,
        stream,
        DeviceCopyDirection::hostToDevice));
  }

  // The ordering is the same as used in Setup.
  decoder_feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));
  decoder_feeds.push_back(input_ids);

  if (!past_present_share_buffer_) {
    past_present_share_buffer_max_seq_len = 0;
  }

  // When first_past_input_index_ == 2, the encoder_hidden_states and past states are copied from the second output
  // of encoder.
  // When first_past_input_index_ == 1, the past states are copied from the second output of encoder.
  // TODO: MAKE IT MORE READABLE
  for (size_t j = static_cast<size_t>(3) - first_past_input_index_; j < encoder_fetches.size(); j++) {
    if (j == 1) {
      ORT_RETURN_IF(has_hidden_state_ == false, "Invalid hidden_states expension: has_hidden_state_ == false");
      OrtValue expanded_hidden_states;
      if (is_output_float16_) {
        ORT_RETURN_IF_ERROR(expand_buffer_float16_func(stream,
                                                       encoder_fetches[j],
                                                       num_beam,
                                                       allocator,
                                                       expanded_hidden_states,
                                                       true,
                                                       0 /*max_sequence_length*/));
      } else {
        ORT_RETURN_IF_ERROR(expand_buffer_float_func(stream,
                                                     encoder_fetches[j],
                                                     num_beam,
                                                     allocator,
                                                     expanded_hidden_states,
                                                     true,
                                                     0 /*max_sequence_length*/));
      }
      decoder_feeds.push_back(expanded_hidden_states);
    } else {
      // past key/value for cross attention does not need to be initialized with max_seq_len since they are static.
      bool use_max_seq_len = (j - first_past_input_index_) <= 2 * static_cast<size_t>(num_layers);

      OrtValue expanded_cache;
      if (is_output_float16_) {
        ORT_RETURN_IF_ERROR(expand_buffer_float16_func(stream,
                                                       encoder_fetches[j],
                                                       num_beam,
                                                       allocator,
                                                       expanded_cache,
                                                       false,
                                                       use_max_seq_len ? past_present_share_buffer_max_seq_len : 0));
      } else {
        ORT_RETURN_IF_ERROR(expand_buffer_float_func(stream,
                                                     encoder_fetches[j],
                                                     num_beam,
                                                     allocator,
                                                     expanded_cache,
                                                     false,
                                                     use_max_seq_len ? past_present_share_buffer_max_seq_len : 0));
      }
      decoder_feeds.push_back(expanded_cache);
    }
  }

  if (past_present_share_buffer_) {
    // Past sequence length feed
    ORT_RETURN_IF_ERROR(AppendPastSequenceLength(decoder_feeds, cpu_allocator, cur_len - 1));
    // Add beam search specific inputs
    if (need_cache_indir) {
      const int64_t batch_size = static_cast<int64_t>(batch_beam_size / num_beam);
      ORT_RETURN_IF_ERROR(AppendBeamWidthAndCacheIndir(decoder_feeds, cpu_allocator, allocator, batch_size, num_beam,
                                                       past_present_share_buffer_max_seq_len));
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
