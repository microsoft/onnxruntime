// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "beam_search_parameters.h"

namespace onnxruntime {
namespace contrib {

Status BeamSearchParameters::Validate() {
  ORT_RETURN_IF(eos_token_id < 0, "eos_token_id is invalid");
  ORT_RETURN_IF(pad_token_id < 0, "pad_token_id is invalid");
  return Status::OK();
}

void BeamSearchParameters::ParseFromAttributes(const OpKernelInfo& info) {
  early_stopping = info.GetAttrOrDefault<int64_t>("early_stopping", 0) == 1;
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
}

void BeamSearchParameters::ParseFromInputs(OpKernelContext* context){
  ORT_ENFORCE(context != nullptr);
  const Tensor* input_ids = context->Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  if (dims.size() == 2) {
    batch_size = static_cast<int>(dims[0]);
    sequence_length = static_cast<int>(dims[1]);
  } else {
    batch_size = 0;
    sequence_length = 0;
  }
  
  auto* max_length_tensor = context->Input<Tensor>(1);
  max_length = max_length_tensor ? static_cast<int>(*max_length_tensor->Data<int32_t>()) : 4096;

  auto* min_length_tensor = context->Input<Tensor>(2);
  min_length = min_length_tensor ? static_cast<int>(*min_length_tensor->Data<int32_t>()) : 0;

  auto* num_beams_tensor = context->Input<Tensor>(3);
  num_beams = num_beams_tensor ? static_cast<int>(*num_beams_tensor->Data<int32_t>()) : 1;

  auto* num_return_sequences_tensor = context->Input<Tensor>(4);
  num_return_sequences = num_return_sequences_tensor ? static_cast<int>(*num_return_sequences_tensor->Data<int32_t>()) : 1;

  auto* temperature_tensor = context->Input<Tensor>(5);
  temperature = temperature_tensor ? static_cast<float>(*temperature_tensor->Data<float>()) : 1;

  auto* length_penalty_tensor = context->Input<Tensor>(6);
  length_penalty = length_penalty_tensor ? static_cast<float>(*length_penalty_tensor->Data<float>()) : 1;

  auto* repetition_penalty_tensor = context->Input<Tensor>(7);
  repetition_penalty = repetition_penalty_tensor ? static_cast<float>(*repetition_penalty_tensor->Data<float>()) : 1.0f;
}

void BeamSearchParameters::SetSubgraphParameters(int heads, int hidden_size_per_head, int vocabulary_size, int layers){
  num_heads = heads;
  head_size = hidden_size_per_head;
  vocab_size = vocabulary_size;
  num_layers = layers;
}

}  // namespace contrib
}  // namespace onnxruntime
