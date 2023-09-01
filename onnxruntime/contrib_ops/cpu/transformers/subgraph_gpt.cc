// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/framework_common.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/common/gsl.h"
#include "contrib_ops/cpu/transformers/subgraph_gpt.h"
#include "contrib_ops/cpu/transformers/dump_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

Status GptSubgraph::CreateInitialFeeds(
    const Tensor& input_ids,
    const std::vector<const OrtValue*>& implicit_inputs,
    int num_beams,
    int pad_token_id,
    gsl::span<int32_t>& sequence_lengths,
    OrtValue& expanded_input_ids,
    const OrtValue* attn_mask_value,
    std::vector<OrtValue>& feeds,
    const GenerationDeviceHelper::CreateGptInputsFunc& create_gpt_inputs_func,
    const GenerationDeviceHelper::AddToFeedsFunc& add_to_feeds_func,
    IAllocatorUniquePtr<char>& buffer,
    Stream* ort_stream,
    int past_present_share_buffer_max_seq_len,
    bool need_cache_indir) {
  ORT_ENFORCE(session_state_ != nullptr, "Setup must be called before CreateInitialFeeds");

  const IExecutionProvider* provider = GetProvider();

  const TensorShape& input_ids_shape = input_ids.Shape();
  ORT_ENFORCE(input_ids_shape.NumDimensions() == 2);
  const int64_t& batch_size = input_ids_shape[0];

  // Subgraph inputs:
  //   input_ids: shape (B, S) where B is batch size, and S is sequence length
  //   position_ids: shape (B, S)
  //   attention_mask: shape (B, P+S), where past_sequence_length (P) is 0
  // After expansion, their shapes will become (B, M*S), where M is num_beams.

  // Allocate subgraph inputs to be same device as input_ids
  AllocatorPtr cpu_allocator = session_state_->GetAllocator(input_ids.Location());

  // Store allocator, which will be used in remaining feeds
  auto default_allocator = session_state_->GetAllocator(provider->GetOrtDeviceByMemType(OrtMemTypeDefault));
  allocator_ = default_allocator;

  // The ordering is the same as used in Setup
  feeds.reserve(static_cast<size_t>(num_subgraph_inputs) + static_cast<size_t>(num_implicit_inputs));

  OrtValue expanded_position_ids;
  OrtValue expanded_attention_mask;
  ORT_RETURN_IF_ERROR(create_gpt_inputs_func(&input_ids,
                                             attn_mask_value,
                                             num_beams,
                                             pad_token_id,
                                             sequence_lengths,
                                             cpu_allocator,
                                             expanded_input_ids,
                                             expanded_position_ids,
                                             expanded_attention_mask));

  AllocatorPtr pinned_allocator = session_state_->GetAllocator(provider->GetOrtDeviceByMemType(OrtMemTypeCPU));
  const OrtMemoryInfo& location = default_allocator->Info();
  ORT_RETURN_IF_ERROR(add_to_feeds_func(ort_stream,
                                        {expanded_input_ids, expanded_position_ids, expanded_attention_mask},
                                        feeds,
                                        buffer,
                                        default_allocator,
                                        pinned_allocator,
                                        location));

  auto past_type = IsOutputFloat16() ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  if (!past_present_share_buffer_) {
    // Initialize empty past state
    TensorShape past_shape{2, batch_size * num_beams, num_heads, 0, head_size};
    OrtValue empty_past;
    Tensor::InitOrtValue(past_type, past_shape, default_allocator, empty_past);

    // The remaining inputs are past state.
    for (int i = first_past_input_index_; i < num_subgraph_inputs; ++i) {
      feeds.push_back(empty_past);
    }
  } else {
    // Past state feeds
    TensorShape past_shape{2, batch_size * num_beams, num_heads, past_present_share_buffer_max_seq_len, head_size};

    // The remaining inputs are past state except the last one or three (see below for details)
    // If `need_cache_indir` is false, then the last input is `past_sequence_length`

    // If `need_cache_indir` is true, then the last inputs are `past_sequence_length`,
    // `beam_width`, and `cache_indirection`
    auto past_end_iter = need_cache_indir ? num_subgraph_inputs - 3 : num_subgraph_inputs - 1;
    for (int i = first_past_input_index_; i < past_end_iter; ++i) {
      OrtValue past_tensor;
      Tensor::InitOrtValue(past_type, past_shape, default_allocator, past_tensor);
      feeds.push_back(past_tensor);
    }

    // Past sequence length feed
    ORT_RETURN_IF_ERROR(AppendPastSequenceLength(feeds, cpu_allocator, 0));

    // Add beam search specific inputs
    if (need_cache_indir) {
      ORT_RETURN_IF_ERROR(AppendBeamWidthAndCacheIndir(feeds, cpu_allocator, default_allocator, batch_size, num_beams,
                                                       past_present_share_buffer_max_seq_len));
    }
  }

  // Pass in implicit inputs
  for (const auto* entry : implicit_inputs) {
    feeds.push_back(*entry);
  }

  return Status::OK();
}

Status GptSubgraph::Validate(const std::vector<const NodeArg*>& subgraph_inputs,
                             const std::vector<const NodeArg*>& subgraph_outputs) {
  ORT_RETURN_IF(num_subgraph_outputs <= first_present_output_index_,
                "Invalid GPT-2 subgraph: number of outputs shall be larger than 1 (Need past state in outputs).");

  ORT_RETURN_IF(!((num_subgraph_inputs == num_subgraph_outputs + 2) ||
                  (num_subgraph_inputs == num_subgraph_outputs + 3) ||
                  (num_subgraph_inputs == num_subgraph_outputs + 5)),
                "Invalid GPT-2 subgraph: number of inputs shall be number of outputs plus 2 or "
                "3 (if past_present_share_buffer) or "
                "5 (if past_present_share_buffer and use_decoder_masked_self_attention for BeamSearch)");

  ORT_RETURN_IF(subgraph_inputs[0]->Name() != "input_ids",
                "subgraph input 0 shall be named as input_ids, got: ", subgraph_inputs[0]->Name());
  ORT_RETURN_IF(subgraph_inputs[1]->Name() != "position_ids",
                "subgraph input 1 shall be named as position_ids, got: ", subgraph_inputs[1]->Name());
  ORT_RETURN_IF(subgraph_inputs[2]->Name() != "attention_mask",
                "subgraph input 2 shall be named as attention_mask, got: ", subgraph_inputs[2]->Name());
  ORT_RETURN_IF(subgraph_inputs[3]->Name() != "past_0",
                "subgraph input 3 shall be named as past_0, got: ", subgraph_inputs[3]->Name());

  // Past state shape is like (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads).
  const ONNX_NAMESPACE::TensorShapeProto* past_shape = subgraph_inputs[3]->Shape();
  ORT_RETURN_IF(past_shape->dim_size() != 5,
                "subgraph past state is expected to have 5 dimension, got ", past_shape->dim_size());

  ORT_RETURN_IF(!past_shape->dim(0).has_dim_value() || past_shape->dim(0).dim_value() != 2,
                "subgraph past state dimension 0 shall have length of 2");

  ORT_RETURN_IF(!past_shape->dim(2).has_dim_value() || past_shape->dim(2).dim_value() <= 0,
                "subgraph past state dimension 2 shall have a positive value for number of heads");

  ORT_RETURN_IF(!past_shape->dim(4).has_dim_value() || past_shape->dim(4).dim_value() <= 0,
                "subgraph past state dimension 4 shall have a positive value for hidden size per head");

  // check subgraph outputs
  ORT_RETURN_IF(subgraph_outputs[0]->Name() != "logits",
                "subgraph output 0 shall be named as logits, got: ", subgraph_outputs[0]->Name());

  ORT_RETURN_IF(subgraph_outputs[1]->Name() != "present_0",
                "subgraph input 1 shall be named as present_0, got: ", subgraph_outputs[1]->Name());

  // Logits shape is like (batch_size, seq_len, 50257). Here 50257 is the vocabulary size.
  const ONNX_NAMESPACE::TensorShapeProto* logits_shape = subgraph_outputs[0]->Shape();
  ORT_RETURN_IF(logits_shape->dim_size() != 3,
                "subgraph logits output is expected to have 3 dimension, got ", logits_shape->dim_size());

  ORT_RETURN_IF(!logits_shape->dim(2).has_dim_value() || logits_shape->dim(2).dim_value() <= 0,
                "subgraph past state dimension 2 shall have a positive value for vocabulary size");

  // Save parameters related to the subgraph.
  num_heads = static_cast<int>(past_shape->dim(2).dim_value());
  head_size = static_cast<int>(past_shape->dim(4).dim_value());
  vocab_size = static_cast<int>(logits_shape->dim(2).dim_value());
  num_layers = static_cast<int>(subgraph_outputs.size()) - 1;

  constexpr auto int32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32;
  constexpr auto float32_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT;
  constexpr auto float16_type = ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;

  ORT_RETURN_IF(subgraph_inputs[0]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "subgraph input 0 (input_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[1]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "subgraph input 1 (position_ids) shall have int32 type");
  ORT_RETURN_IF(subgraph_inputs[2]->TypeAsProto()->tensor_type().elem_type() != int32_type,
                "subgraph input 2 (attention_mask) shall have int32 type");

  auto output_type = subgraph_outputs[0]->TypeAsProto()->tensor_type().elem_type();
  ORT_RETURN_IF(output_type != float32_type && output_type != float16_type,
                "subgraph output 0 (logits) shall be float or float16 data type");

  ORT_RETURN_IF(subgraph_inputs[first_past_input_index_]->TypeAsProto()->tensor_type().elem_type() != output_type,
                "subgraph input 3 (past_0) shall shall have same data type of logits output");
  ORT_RETURN_IF(subgraph_outputs[first_present_output_index_]->TypeAsProto()->tensor_type().elem_type() != output_type,
                "subgraph output 1 (present_0) shall shall have same data type of logits output");

  is_output_float16_ = (output_type == float16_type);

  return Status::OK();
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
