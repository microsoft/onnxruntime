// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "contrib_ops/cpu/transformers/subgraph_t5_decoder.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cpu/transformers/generation_device_helper.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include <gsl/gsl>
#include <memory>

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* T5 Decoder Subgraph.

   Inputs:
      input_ids: int32 (B, 1)
      encoder_input_ids: int32 (B, encode_sequence_length) (optional for old format; removed in new format)
      encoder_attention_mask: int32 (B, encode_sequence_length)
      encoder_hidden_states: (B, encode_sequence_length, encoder_hidden_size) (optional for old format; removed in new format)

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

Status T5DecoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                   const std::vector<const NodeArg*>& subgraph_outputs) {
  bool has_encoder_input_ids = subgraph_inputs[1]->Name() == "encoder_input_ids";
  bool has_hidden_state = subgraph_inputs[2 + has_encoder_input_ids]->Name() == "encoder_hidden_states";
  SetPastInputIndex(has_hidden_state, has_encoder_input_ids);

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
  const int enc_attn_mask_index = 1 + has_encoder_input_ids_;
  const int enc_hidden_state_index = enc_attn_mask_index + 1;
  ORT_RETURN_IF(subgraph_inputs[enc_attn_mask_index]->Name() != "encoder_attention_mask",
                "decoder subgraph input ", std::to_string(enc_attn_mask_index),
                " shall be named as encoder_attention_mask, got: ",
                subgraph_inputs[enc_attn_mask_index]->Name());
  if (has_hidden_state_) {
    ORT_RETURN_IF(subgraph_inputs[enc_hidden_state_index]->Name() != "encoder_hidden_states",
                  "decoder subgraph input ", std::to_string(enc_hidden_state_index),
                  " shall be named as encoder_hidden_states, got: ",
                  subgraph_inputs[enc_hidden_state_index]->Name());
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
  if (has_encoder_input_ids_) {
    ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                  "decoder subgraph input 1 (encoder_input_ids) shall have int32 type");
  }
  ORT_RETURN_IF(subgraph_inputs[enc_attn_mask_index]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "decoder subgraph input ", std::to_string(enc_attn_mask_index),
                " (encoder_attention_mask) shall have int32 type");

  auto float_type = subgraph_inputs[enc_hidden_state_index]->TypeAsProto()->tensor_type().elem_type();
  if (has_hidden_state_) {
    ORT_RETURN_IF(float_type != float32_type && float_type != float16_type,
                  "decoder subgraph input ", std::to_string(enc_hidden_state_index), " (encoder_hidden_states) shall have float or float16 type");
  }

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
// New format:
//   encoder feeds: encoder_input_ids, encoder_attention_mask
//   encoder fetches: present_key_cross_0, present_value_cross_0, ...
//   decoder_feeds: input_ids, encoder_attention_mask,
//                  present_key_self_0, present_value_self_0, ...,
//                  present_key_cross_0, present_value_cross_0, ...
//                  past_seq_len (optional), num_beams (optional), cache_indirection (optional)
//
// Old format:
//   encoder feeds: encoder_input_ids, encoder_attention_mask, decoder_input_ids (with start tokens)
//   encoder fetches: logits, encoder_hidden_states,
//                    present_key_self_0, present_value_self_0, ...,
//                    present_key_cross_0, present_value_cross_0, ...
//   decoder_feeds: input_ids, encoder_input_ids (optional), encoder_attention_mask, encoder_hidden_states (optional),
//                  present_key_self_0, present_value_self_0, ...,
//                  present_key_cross_0, present_value_cross_0, ...
//                  past_seq_len (optional), num_beams (optional), cache_indirection (optional)
Status T5DecoderSubgraph::CreateInitialFeeds(
    AllocatorPtr cpu_allocator,
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
    bool copy_sequence_to_input_ids,
    transformers::Sequences& sequences,
    int past_present_share_buffer_max_seq_len,
    bool need_cache_indir,
    bool use_cuda) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  // Allocate subgraph inputs from same device as inputs of encoder subgraph.
  AllocatorPtr allocator = session_state_->GetAllocator(encoder_feeds[0].Get<Tensor>().Location());

  int batch_beam_size = static_cast<int>(encoder_fetches[0].Get<Tensor>().Shape()[0]) * num_beam;

  // Copy beam next tokens in CPU to input_ids in provider device (CPU for CPU EP, or GPU for CUDA EP).
  int sequence_length = !copy_sequence_to_input_ids ? 1 : sequences.GetSequenceLength();
  int64_t dims[] = {batch_beam_size, sequence_length};
  TensorShape input_ids_shape(&dims[0], 2);
  OrtValue input_ids;
  Tensor::InitOrtValue(DataTypeImpl::GetType<int32_t>(), input_ids_shape, allocator, input_ids);

  // Prepare data for input_ids.
  if (!copy_sequence_to_input_ids) {  // use next tokens for input_ids.
    ORT_RETURN_IF_ERROR(device_copy_int32_func(
        input_ids.GetMutable<Tensor>()->MutableDataAsSpan<int32_t>(),
        beam_next_tokens,
        stream,
        DeviceCopyDirection::hostToDevice));
  } else {  // use whole sequences for input_ids.
    int32_t* input_ids_data = input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
    if (use_cuda) {
      auto sequences_buffer = sequences.GetCurrentDeviceSequences();
      for (int i = 0; i < batch_beam_size; i++) {
        size_t offset = static_cast<size_t>(i) * static_cast<size_t>(sequences.GetMaxLength());
        gsl::span<const int32_t> sequence = sequences_buffer.subspan(offset, sequence_length);
        gsl::span<int> temp_input(input_ids_data + static_cast<ptrdiff_t>(i) * sequence_length, sequence_length);
        ORT_RETURN_IF_ERROR(device_copy_int32_func(
            temp_input,
            sequence,
            stream,
            DeviceCopyDirection::deviceToDevice));
      }
    } else {
      size_t total_size = static_cast<size_t>(sequence_length) * static_cast<size_t>(batch_beam_size);
      size_t total_size_bytes = total_size * sizeof(int);
      AllocatorPtr buffer_allocator = std::make_shared<onnxruntime::CPUAllocator>();
      // TODO: not need extra buffer. Copy directly to input_ids_data instead like the user_cuda above.
      auto seq_copy = IAllocator::MakeUniquePtr<int>(buffer_allocator, total_size_bytes, false, stream);
      int* seq_copy_ptr = seq_copy.get();

      const size_t sequence_bytes = sequence_length * sizeof(int);
      for (int i = 0; i < batch_beam_size; i++) {
        gsl::span<const int32_t> sequence = sequences.GetSequence(i);
        const int32_t* sequence_data = sequence.data();
        ptrdiff_t seq_index = static_cast<ptrdiff_t>(i) * sequence_length;
        memcpy(seq_copy_ptr + seq_index, sequence_data, sequence_bytes);
      }
      gsl::span<int> temp_input(input_ids_data, total_size);
      gsl::span<int> temp_sequence(seq_copy_ptr, total_size);
      ORT_RETURN_IF_ERROR(device_copy_int32_func(
          temp_input,
          temp_sequence,
          stream,
          DeviceCopyDirection::hostToDevice));
    }
  }

  // The ordering is the same as used in Setup.
  decoder_feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // input 0: input_ids
  decoder_feeds.push_back(input_ids);

  if (has_encoder_input_ids_) {  // encoder_input_ids is optional
    // The encoder_input_ids is copied from the first input of encoder.
    OrtValue expanded_encoder_input_ids;
    ORT_RETURN_IF_ERROR(expand_buffer_int32_func(stream,
                                                 encoder_feeds[0],
                                                 num_beam,
                                                 allocator,
                                                 expanded_encoder_input_ids,
                                                 false,
                                                 0 /*max_sequence_length*/));
    decoder_feeds.push_back(expanded_encoder_input_ids);
  }

  // The encoder_attention_mask is copied from the second input of encoder.
  OrtValue expanded_decoder_attention_masks;
  ORT_RETURN_IF_ERROR(expand_buffer_int32_func(stream,
                                               encoder_feeds[1],
                                               num_beam,
                                               allocator,
                                               expanded_decoder_attention_masks,
                                               false,
                                               0 /*max_sequence_length*/));
  decoder_feeds.push_back(expanded_decoder_attention_masks);

  if (!past_present_share_buffer_) {
    past_present_share_buffer_max_seq_len = 0;
  }

// macro to expand encoder outputs and append to decoder feeds.
#define ADD_DECODER_FEED(encoder_output, is_dynamic_kv_cache)                                                         \
  OrtValue expanded;                                                                                                  \
  if (is_output_float16_) {                                                                                           \
    ORT_RETURN_IF_ERROR(expand_buffer_float16_func(stream, encoder_output, num_beam, allocator, expanded, false,      \
                                                   is_dynamic_kv_cache ? past_present_share_buffer_max_seq_len : 0)); \
  } else {                                                                                                            \
    ORT_RETURN_IF_ERROR(expand_buffer_float_func(stream, encoder_output, num_beam, allocator, expanded, false,        \
                                                 is_dynamic_kv_cache ? past_present_share_buffer_max_seq_len : 0));   \
  }                                                                                                                   \
  decoder_feeds.push_back(expanded);

  // The encoder_hidden_states is copied from the second output of encoder.
  if (has_hidden_state_) {
    ADD_DECODER_FEED(encoder_fetches[1], false);
  }

  // New format of encoder has only cross outputs.
  bool is_new_format = (static_cast<int>(encoder_fetches.size()) == 2 * num_layers);
  if (is_new_format) {
    for (int i = 0; i < 2 * num_layers; i++) {
      // cross shape is (batch_size, num_heads, encode_sequence_length, head_size)
      const TensorShape& cross_shape = encoder_fetches[0].Get<Tensor>().Shape();
      ORT_ENFORCE(cross_shape.NumDimensions() == 4);

      // Shape for kv cache: (batch_size * num_beam, num_heads, max_seq_len, head_size)
      int64_t cache_dims[4] = {0};
      cross_shape.CopyDims(cache_dims, cross_shape.NumDimensions());
      cache_dims[0] *= num_beam;
      cache_dims[2] = past_present_share_buffer_max_seq_len;
      TensorShape expanded_shape(&cache_dims[0], cross_shape.NumDimensions());

      MLDataType element_type = encoder_fetches[0].Get<Tensor>().DataType();
      OrtValue past;
      Tensor::InitOrtValue(element_type, expanded_shape, allocator, past);
      decoder_feeds.push_back(past);
    }

    // Add cross inputs from encoder output.
    for (size_t j = 0; j < encoder_fetches.size(); j++) {
      ADD_DECODER_FEED(encoder_fetches[j], false);
    }
  } else {
    // present_* output of encoder are added as decoder inputs.
    for (size_t j = 2; j < encoder_fetches.size(); j++) {
      // past key/value for cross attention does not need to be initialized with max_seq_len since they are static.
      bool is_dynamic_kv_cache = (j - first_past_input_index_) < 2 * static_cast<size_t>(num_layers);
      ADD_DECODER_FEED(encoder_fetches[j], is_dynamic_kv_cache);
    }
  }

  if (past_present_share_buffer_) {
    // Past sequence length set to 0
    ORT_RETURN_IF_ERROR(AppendPastSequenceLength(decoder_feeds, cpu_allocator, is_new_format ? 0 : 1));
    // Add beam search specific inputs
    if (need_cache_indir) {
      const int64_t batch_size = static_cast<int64_t>(batch_beam_size / num_beam);
      ORT_RETURN_IF_ERROR(AppendBeamWidthAndCacheIndir(decoder_feeds, cpu_allocator, allocator, batch_size, num_beam,
                                                       past_present_share_buffer_max_seq_len));
    }
  }

  // Pass through implicit inputs.
  for (size_t i = 0; i < implicit_inputs.size(); ++i) {
    const auto* entry = implicit_inputs[i];
    if (used_implicit_inputs[i]) {
      decoder_feeds.push_back(*entry);
    }
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
