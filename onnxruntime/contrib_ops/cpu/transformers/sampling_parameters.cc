// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cpu/transformers/sampling_parameters.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

constexpr int kMaxSequenceLength = 4096;

void SamplingParameters::ParseFromAttributes(const OpKernelInfo& info) {
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", 0));
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  decoder_start_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("decoder_start_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
  temperature = info.GetAttrOrDefault<float>("temperature", 1.0f);
  top_p = info.GetAttrOrDefault<float>("top_p", 0.0f);
  filter_value = info.GetAttrOrDefault<float>("filter_value", -std::numeric_limits<float>::infinity());
  presence_penalty = info.GetAttrOrDefault<float>("presence_penalty", 0.0f);
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
