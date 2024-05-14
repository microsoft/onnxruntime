// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cpu/transformers/greedy_search_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

constexpr int kMaxSequenceLength = 16384;

void GreedySearchParameters::ParseFromAttributes(const OpKernelInfo& info) {
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", 0));
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  decoder_start_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("decoder_start_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
  vocab_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("vocab_size", -1));
}

void GreedySearchParameters::ParseFromInputs(OpKernelContext* context) {
  ORT_ENFORCE(context != nullptr);
  const Tensor* input_ids = context->Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  ORT_ENFORCE(dims.size() == 2, "input_ids shall have 2 dimensions. Got ", dims.size());
  batch_size = static_cast<int>(dims[0]);
  sequence_length = static_cast<int>(dims[1]);

  auto* max_length_tensor = context->Input<Tensor>(1);
  max_length = max_length_tensor ? static_cast<int>(*max_length_tensor->Data<int32_t>()) : kMaxSequenceLength;
  ORT_ENFORCE(max_length > sequence_length,
              "max_length (", max_length, ") shall be greater than input sequence length (", sequence_length, ")");
  ORT_ENFORCE(max_length <= kMaxSequenceLength,
              "max_length (", max_length, ") shall be no more than ", kMaxSequenceLength);

  auto* min_length_tensor = context->Input<Tensor>(2);
  min_length = min_length_tensor ? static_cast<int>(*min_length_tensor->Data<int32_t>()) : 0;

  num_beams = static_cast<int>(1);

  auto* repetition_penalty_tensor = context->Input<Tensor>(3);
  repetition_penalty = repetition_penalty_tensor ? static_cast<float>(*repetition_penalty_tensor->Data<float>()) : 1.0f;
  ORT_ENFORCE(repetition_penalty > 0.0f, "repetition_penalty shall be greater than 0, got ", repetition_penalty);
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
