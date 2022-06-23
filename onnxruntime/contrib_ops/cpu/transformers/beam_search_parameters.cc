// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/transformers/beam_search_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

constexpr int kMaxSequenceLength = 4096;
constexpr int kMaxNumBeams = 128;

Status BeamSearchParameters::Validate() const {
  ORT_RETURN_IF(eos_token_id < 0, "eos_token_id is invalid");
  ORT_RETURN_IF(pad_token_id < 0, "pad_token_id is invalid");
  ORT_RETURN_IF(min_length >= max_length, "min_length shall be smaller than max_length");
  return Status::OK();
}

void BeamSearchParameters::ParseFromAttributes(const OpKernelInfo& info) {
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", IBeamSearchParameters::kModelTypeGpt));
  early_stopping = info.GetAttrOrDefault<int64_t>("early_stopping", 0) == 1;
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  decoder_start_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("decoder_start_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
}

void BeamSearchParameters::ParseFromInputs(OpKernelContext* context) {
  ORT_ENFORCE(context != nullptr);
  const Tensor* input_ids = context->Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  ORT_ENFORCE(dims.size() == 2, "input_ids shall have 2 dimensions. Got ", dims.size());
  batch_size = static_cast<int>(dims[0]);

  // For T5, output sequence starts with decoder_start_token_id, so its sequence length is 1
  sequence_length = (this->model_type == IBeamSearchParameters::kModelTypeGpt) ? static_cast<int>(dims[1]) : 1;

  auto* max_length_tensor = context->Input<Tensor>(1);
  max_length = max_length_tensor ? static_cast<int>(*max_length_tensor->Data<int32_t>()) : kMaxSequenceLength;
  ORT_ENFORCE(max_length > sequence_length,
              "max_length (", max_length, ") shall be greater than input sequence length (", sequence_length, ")");
  ORT_ENFORCE(max_length <= kMaxSequenceLength,
              "max_length (", max_length, ") shall be no more than ", kMaxSequenceLength);

  auto* min_length_tensor = context->Input<Tensor>(2);
  min_length = min_length_tensor ? static_cast<int>(*min_length_tensor->Data<int32_t>()) : 0;

  auto* num_beams_tensor = context->Input<Tensor>(3);
  num_beams = num_beams_tensor ? static_cast<int>(*num_beams_tensor->Data<int32_t>()) : 1;
  // TODO(tianleiwu): limit num_beams > 1 when we can have another operator for greedy search.
  ORT_ENFORCE(num_beams >= 1 && num_beams <= kMaxNumBeams,
              "num_beams shall be a positive integer no more than ", kMaxNumBeams, ", got ", num_beams);

  auto* num_return_sequences_tensor = context->Input<Tensor>(4);
  num_return_sequences = num_return_sequences_tensor ? *num_return_sequences_tensor->Data<int32_t>() : 1;
  ORT_ENFORCE(num_return_sequences >= 1,
              "num_return_sequences shall be a positive integer, got ", num_return_sequences);
  ORT_ENFORCE(num_beams >= num_return_sequences,
              "num_return_sequences (", num_return_sequences, ") shall be be no more than num_beams (", num_beams, ")");

  auto* length_penalty_tensor = context->Input<Tensor>(5);
  length_penalty = length_penalty_tensor ? static_cast<float>(*length_penalty_tensor->Data<float>()) : 1;

  auto* repetition_penalty_tensor = context->Input<Tensor>(6);
  repetition_penalty = repetition_penalty_tensor ? static_cast<float>(*repetition_penalty_tensor->Data<float>()) : 1.0f;
  ORT_ENFORCE(repetition_penalty > 0.0f, "repetition_penalty shall be greater than 0, got ", repetition_penalty);
}

void BeamSearchParameters::SetSubgraphParameters(int vocabulary_size, int heads, int hidden_size_per_head, int layers) {
  vocab_size = vocabulary_size;
  num_heads = heads;
  head_size = hidden_size_per_head;
  num_layers = layers;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
