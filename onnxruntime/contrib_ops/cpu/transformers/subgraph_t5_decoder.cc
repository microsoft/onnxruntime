// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "gsl/gsl"
#include "subgraph_t5_decoder.h"
#include "dump_tensor.h"
#include "beam_search_device_helper.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* T5 Decoder Subgraph.

   Inputs:
      input_ids: int32 (batch_size, 1)
      encoder_attention_mask: int32 (batch_size, encode_sequence_length)
      encoder_hidden_states: (batch_size, encode_sequence_length, encoder_hidden_size)

      past_key_self_0: (batch_size, num_heads, past_decode_sequence_length, head_size)
      past_value_self_0: (batch_size, num_heads, past_decode_sequence_length, head_size)
      ... (for each self attention layer)

      past_key_cross_0: (batch_size, num_heads, encode_sequence_length, head_size)
      past_value_cross_0: (batch_size, num_heads, encode_sequence_length, head_size)
      ... (for each cross attention layer)

    Outputs:
      logits: (batch_size, 1, vocab_size)

      present_key_self_0: (batch_size, num_heads, past_decode_sequence_length + 1, head_size)
      present_value_self_0: (batch_size, num_heads, past_decode_sequence_length + 1, head_size)
      ... (for each self attention layer)

    Note: Data type of input or output is float or float16 if not specifed.
*/

Status T5DecoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                   const std::vector<const NodeArg*>& subgraph_outputs) {
  ORT_RETURN_IF(num_subgraph_inputs < 7 || (num_subgraph_inputs - 3) % 4 != 0, "number of outputs expected to be 3 + 4 * layers, got:", num_subgraph_inputs);
  ORT_RETURN_IF(num_subgraph_outputs < 5 || (num_subgraph_outputs - 1) % 4 != 0, "number of outputs expected to be 1 + 4 * layers, got:", num_subgraph_outputs);

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids", "decoder subgraph input 0 shall be named as input_ids, got: ",
                subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "encoder_attention_mask", "decoder subgraph input 1 shall be named as encoder_attention_mask, got: ",
                subgraph_inputs[1]->Name());
  ORT_RETURN_IF(subgraph_inputs[2]->Name() != "encoder_hidden_states", "decoder subgraph input 2 shall be named as encoder_hidden_states, got: ",
                subgraph_inputs[2]->Name());

  // check subgraph outputs
  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits", "decoder subgraph output 0 shall be named as logits, got: ",
                subgraph_outputs[0]->Name());

  // Logits shape is like (batch_size, seq_len, 32128). Here 32128 is the vocabulary size.
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();
  ORT_RETURN_IF(logits_shape->dim_size() != 3, "decoder subgraph logits output is expected to have 3 dimension, got ",
                logits_shape->dim_size());

  ORT_RETURN_IF(!logits_shape->dim(2).has_dim_value() || logits_shape->dim(2).dim_value() <= 0,
                "decoder subgraph past state dimension 2 shall have a positive value for vocabulary size");

  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_outputs[2]->Shape();
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();

  // Save parameters related to the subgraph.
  ORT_RETURN_IF_ERROR(GetParameters(past_shape, logits_shape, false));
  num_layers = (static_cast<int>(subgraph_outputs.size()) - 1) / 4;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "decoder subgraph input 0 (input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                "decoder subgraph input 1 (encoder_attention_mask) shall have int32 type");

  auto float_type = subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(float_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT && float_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
                "decoder subgraph input 2 (encoder_hidden_states) shall have float or float16 type");

  for (int i = 3; i < num_subgraph_inputs; i++) {
    ORT_RETURN_IF(subgraph_inputs[i]->TypeAsProto()->tensor_type().elem_type() != float_type,
                  "decoder subgraph past inputs shall have same data type as that of encoder_hidden_states");
  }

  for (int i = 0; i < num_subgraph_outputs; i++) {
    ORT_RETURN_IF(subgraph_outputs[i]->TypeAsProto()->tensor_type().elem_type() != float_type,
                  "decoder subgraph output shall have same data type as that of encoder_hidden_states");
  }

  is_output_float16_ = (output_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);

  return Status::OK();
}

Status T5DecoderSubgraph::CreateInitialFeeds(
    const Tensor& encoder_input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int num_beams,
    int decoder_start_token_id,
    std::vector<OrtValue>& decoder_feeds,
    const std::vector<OrtValue>& encoder_feeds,
    const std::vector<OrtValue>& encoder_fetches,
    IAllocatorUniquePtr<char>&) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  const IExecutionProvider* provider = GetProvider();

  const TensorShape& encoder_input_ids_shape = encoder_input_ids.Shape();
  ORT_ENFORCE(encoder_input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = encoder_input_ids_shape[0];

  // Decoder Subgraph inputs:
  //   input_ids: shape (B, 1) wher B is batch size
  //   attention_mask: shape (B, S), where S is the sequence length
  //   encoder_outputs: shape (B, S, NH), where NH is the hidden size
  // After expansion, their shapes will become (B*M, ...), where M is num_beams.

  // Allocate subgraph inputs to be same device as input_ids
  AllocatorPtr cpu_alloactor = session_state_->GetAllocator(encoder_input_ids.Location());

  // Store allocator, which will be used in remaining feeds
  auto default_allocator = provider->GetAllocator(0, OrtMemTypeDefault);
  allocator_ = default_allocator;
  const OrtMemoryInfo& location = cpu_alloactor->Info();

  // The ordering is the same as used in Setup
  decoder_feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  auto element_type = DataTypeImpl::GetType<int32_t>();

  OrtValue decoder_input_ids;
  TensorShape decoder_input_ids_shape({batch_size, 1});
  Tensor::InitOrtValue(element_type, decoder_input_ids_shape, cpu_alloactor, decoder_input_ids);
  int32_t* decoder_input_ids_data = decoder_input_ids.GetMutable<Tensor>()->MutableData<int32_t>();
  for (int i = 0; i < batch_size; i++) {
    *decoder_input_ids_data = decoder_start_token_id;
    decoder_input_ids_data++;
  }

  OrtValue decoder_attention_masks;
  const Tensor* encoder_attention_masks = &encoder_feeds[1].Get<Tensor>();
  Tensor::InitOrtValue(element_type, encoder_attention_masks->Shape(), const_cast<Tensor*>(encoder_attention_masks)->MutableData<int32_t>(), location, decoder_attention_masks);

  // bugbug: handle fp16 later
  OrtValue encoder_output;
  const Tensor* encoder_outputs = &encoder_fetches[0].Get<Tensor>();
  Tensor::InitOrtValue(element_type, encoder_outputs->Shape(), const_cast<Tensor*>(encoder_outputs)->MutableData<float>(), location, encoder_output);

  OrtValue expanded_decoder_input_ids = BeamSearchCpuDeviceHelper::ExpandInputs(decoder_input_ids, num_beams, cpu_alloactor);
  OrtValue expanded_decoder_attention_masks = BeamSearchCpuDeviceHelper::ExpandInputs(decoder_attention_masks, num_beams, cpu_alloactor);
  OrtValue expanded_encoder_output = BeamSearchCpuDeviceHelper::ExpandInputs(encoder_output, num_beams, cpu_alloactor);

  decoder_feeds.push_back(expanded_decoder_input_ids);
  decoder_feeds.push_back(expanded_decoder_attention_masks);
  decoder_feeds.push_back(expanded_encoder_output);

  // pass in implicit inputs
  for (const auto* entry : implicit_inputs) {
    decoder_feeds.push_back(*entry);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
