// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "gsl/gsl"
#include "subgraph_t5_encoder.h"
#include "dump_tensor.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace transformers {

/* T5 Encoder Subgraph (It also contains decoder initialization where decoder_input_ids are filled with start token ID).

   Inputs:
      encoder_input_ids: int64 (B, encode_sequence_length)
      encoder_attention_mask: int64 (B, encode_sequence_length)
      decoder_input_ids: int64 (B, 1) OPTIONAL

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
      Here, B = batch_size * num_beams since we expand the inputs. (Ideally, we could use B=batch_size and expand the outputs with a factor of num_beams).
      Data type of input or output is float or float16 if not specifed.
*/

Status T5EncoderSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                                   const std::vector<const NodeArg*>& subgraph_outputs) {
  // TODO: subgraph with 2 inputs is not supported in BeamSearchT5 yet
  ORT_RETURN_IF(num_subgraph_inputs != 2 && num_subgraph_inputs != 3, "expect 2 or 3 inputs, got:", num_subgraph_inputs);

  ORT_RETURN_IF(num_subgraph_outputs < 6, "expect >=6 outputs, got:", num_subgraph_outputs);
  ORT_RETURN_IF((static_cast<int>(subgraph_outputs.size()) - 2) % 4 != 0, "number of outputs expected to be 2 + 4 * layers, got:", num_subgraph_outputs);

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "encoder_input_ids", "encoder subgraph input 0 shall be named as encoder_input_ids, got: ",
                subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "encoder_attention_mask", "encoder subgraph input 1 shall be named as encoder_attention_mask, got: ",
                subgraph_inputs[1]->Name());

  if (num_subgraph_inputs == 3) {
    ORT_RETURN_IF(subgraph_inputs[2]->Name() != "decoder_input_ids", "encoder subgraph input 2 shall be named as decoder_input_ids, got: ",
                  subgraph_inputs[2]->Name());
  }

  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits", "encoder subgraph output 0 shall be named as logits, got: ",
                subgraph_outputs[0]->Name());

  ORT_RETURN_IF(subgraph_outputs[1]->Name() != "encoder_hidden_states", "encoder subgraph output 1 shall be named as encoder_hidden_states, got: ",
                subgraph_outputs[1]->Name());

  ORT_RETURN_IF(subgraph_outputs[2]->Name() != "present_key_self_0", "encoder subgraph output 2 shall be named as present_key_self_0, got: ",
                subgraph_outputs[2]->Name());

  ORT_RETURN_IF(subgraph_outputs[3]->Name() != "present_value_self_0", "encoder subgraph output 3 shall be named as present_value_self_0, got: ",
                subgraph_outputs[3]->Name());

  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_outputs[2]->Shape();
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();

  // Save parameters related to the subgraph.
  ORT_RETURN_IF_ERROR(GetParameters(past_shape, logits_shape, false));
  num_layers = (static_cast<int>(subgraph_outputs.size()) - 2) / 4;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                "encoder subgraph input 0 (encoder_input_ids) shall have int64 type");

  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                "encoder subgraph input 1 (encoder_attention_mask) shall have int64 type");

  if (num_subgraph_inputs == 3) {
    ORT_RETURN_IF(subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                  "encoder subgraph input 2 (decoder_input_ids) shall have int64 type");
  }

  auto output_type = subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT && output_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
                "encoder subgraph output 0 (logits) shall be float or float16 data type");

  for (int i = 1; i < num_subgraph_outputs; i++) {
    ORT_RETURN_IF(subgraph_outputs[i]->TypeAsProto()->tensor_type().elem_type() != output_type,
                  "encoder subgraph outputs shall have same data type");
  }

  is_output_float16_ = (output_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16);

  return Status::OK();
}

// Create inputs for first inference of subgraph.
Status T5EncoderSubgraph::CreateInitialFeeds(
    const Tensor& encoder_input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int num_beams,
    int pad_token_id,
    int start_token_id,
    std::vector<OrtValue>& feeds,
    const BeamSearchDeviceHelper::CreateEncoderInputsFunc& create_encoder_inputs_func,
    const BeamSearchDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
    IAllocatorUniquePtr<char>& buffer) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  // The ordering is the same as used in Setup
  feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  // Allocate subgraph inputs to be same device as encoder_input_ids
  AllocatorPtr cpu_alloactor = session_state_->GetAllocator(encoder_input_ids.Location());

  // TODO: expand the outputs instead of inputs to save computation.
  OrtValue expanded_encoder_input_ids;
  OrtValue expanded_encoder_attention_mask;
  OrtValue expanded_decoder_input_ids;  // filled with start token ID
  ORT_RETURN_IF_ERROR(create_encoder_inputs_func(&encoder_input_ids,
                                                 num_beams,
                                                 pad_token_id,
                                                 start_token_id,
                                                 cpu_alloactor,
                                                 expanded_encoder_input_ids,
                                                 expanded_encoder_attention_mask,
                                                 expanded_decoder_input_ids));

  const IExecutionProvider* provider = GetProvider();
  ORT_RETURN_IF_ERROR(add_to_feeds_func(provider, expanded_encoder_input_ids, expanded_encoder_attention_mask, expanded_decoder_input_ids, feeds, buffer));

  for (const auto* entry : implicit_inputs) {
    feeds.push_back(*entry);
  }

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
