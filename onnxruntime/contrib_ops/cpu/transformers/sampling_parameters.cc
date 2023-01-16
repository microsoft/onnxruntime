// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cpu/transformers/sampling_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

void SamplingParameters::ParseFromAttributes(const OpKernelInfo& info) {
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", 0));
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  decoder_start_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("decoder_start_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
  temperature = info.GetAttrOrDefault<float>("temperature", 1.0f);
  top_p = info.GetAttrOrDefault<float>("top_p", 0.0f);
  filter_value = info.GetAttrOrDefault<float>("filter_value", -std::numeric_limits<float>::infinity());
  min_tokens_to_keep = static_cast<int>(info.GetAttrOrDefault<int64_t>("min_tokens_to_keep", 0));
  presence_penalty = info.GetAttrOrDefault<float>("presence_penalty", 0.0f);
  custom_sampling = static_cast<int>(info.GetAttrOrDefault<int64_t>("custom", 0));
  vocab_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("vocab_size", -1));
}

void SamplingParameters::ParseFromInputs(OpKernelContext* context) {
  this->GreedySearchParameters::ParseFromInputs(context);

  auto* seed_tensor = context->Input<Tensor>(8);
  seed = seed_tensor ? static_cast<int>(*seed_tensor->Data<int32_t>()) : 0;
  ORT_ENFORCE(seed >= 0, "Seed must be >= 0");
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
