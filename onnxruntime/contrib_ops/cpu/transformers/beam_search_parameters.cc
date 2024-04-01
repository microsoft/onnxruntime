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
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", IGenerationParameters::kModelTypeGpt));
  early_stopping = info.GetAttrOrDefault<int64_t>("early_stopping", 0) == 1;
  eos_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("eos_token_id", -1));
  pad_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("pad_token_id", -1));
  decoder_start_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("decoder_start_token_id", -1));
  no_repeat_ngram_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_repeat_ngram_size", 0));
  vocab_size = static_cast<int>(info.GetAttrOrDefault<int64_t>("vocab_size", -1));
}

void BeamSearchParameters::ParseFromInputs(OpKernelContext* context) {
  ORT_ENFORCE(context != nullptr);
  const Tensor* input_ids = context->Input<Tensor>(0);
  const auto& dims = input_ids->Shape().GetDims();
  int initial_decode_sequence_length = 0;
  if (this->model_type == IGenerationParameters::kModelTypeWhisper) {
    ORT_ENFORCE(dims.size() == 3, "input_features shall have 3 dimensions. Got ", dims.size());
    const Tensor* decoder_input_ids = context->Input<Tensor>(10);
    if (decoder_input_ids == nullptr) {
      initial_decode_sequence_length = 1;
    } else {
      const auto& decoder_dims = decoder_input_ids->Shape().GetDims();
      initial_decode_sequence_length = static_cast<int>(decoder_dims[1]);
      ORT_ENFORCE(decoder_dims.size() == 2, "decoder_input_ids shall have 2 dimensions. Got ", decoder_dims.size());
    }
  } else {
    ORT_ENFORCE(dims.size() == 2, "input_ids shall have 2 dimensions. Got ", dims.size());
  }
  batch_size = static_cast<int>(dims[0]);

  extra_decoding_ids = gsl::span<int32_t>();
  if (this->model_type == IGenerationParameters::kModelTypeWhisper && extra_decoding_ids_input_id > 0) {
    const Tensor* extra_decoder_tensor = context->Input<Tensor>(extra_decoding_ids_input_id);
    if (extra_decoder_tensor != nullptr) {
      const auto& extra_decoder_tensor_dims = extra_decoder_tensor->Shape().GetDims();
      ORT_ENFORCE(extra_decoder_tensor_dims.size() == 2,
                  "extra_decoder_tensor shall have 2 dimensions. Got ",
                  extra_decoder_tensor_dims.size());
      ORT_ENFORCE(extra_decoder_tensor_dims[0] == batch_size,
                  "extra_decoder_tensor first dim not same as batch_size. Got ",
                  extra_decoder_tensor_dims[0], ", expecting ", batch_size);
      if (extra_decoder_tensor->Shape().Size() > 0) {
        extra_decoding_ids = gsl::span<const int32_t>(extra_decoder_tensor->Data<int32_t>(), (size_t)extra_decoder_tensor->Shape().Size());
      }
    }
  }

  if (this->model_type == IGenerationParameters::kModelTypeGpt) {
    sequence_length = static_cast<int>(dims[1]);
  } else if (this->model_type == IGenerationParameters::kModelTypeWhisper) {
    sequence_length = initial_decode_sequence_length;
  } else {
    // For T5, output sequence starts with decoder_start_token_id, so its sequence length is 1
    sequence_length = 1;
  }

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
  if (length_penalty_tensor) {
    if (length_penalty_tensor->IsDataType<float>()) {
      length_penalty = *length_penalty_tensor->Data<float>();
    } else {
      length_penalty = static_cast<float>(*length_penalty_tensor->Data<MLFloat16>());
    }
  } else {
    length_penalty = 1.0f;
  }

  auto* repetition_penalty_tensor = context->Input<Tensor>(6);
  if (repetition_penalty_tensor) {
    if (repetition_penalty_tensor->IsDataType<float>()) {
      repetition_penalty = *repetition_penalty_tensor->Data<float>();
    } else {
      repetition_penalty = static_cast<float>(*repetition_penalty_tensor->Data<MLFloat16>());
    }
  } else {
    repetition_penalty = 1.0f;
  }
  ORT_ENFORCE(repetition_penalty > 0.0f, "repetition_penalty shall be greater than 0, got ", repetition_penalty);

  auto* logits_processor_tensor = context->Input<Tensor>(11);
  logits_processor = logits_processor_tensor ? static_cast<int>(*logits_processor_tensor->Data<int32_t>()) : 0;
  ORT_ENFORCE(logits_processor >= 0,
              "logits_processor shall be a non-negative integer, got ", logits_processor);

  if (this->model_type == IGenerationParameters::kModelTypeWhisper) {
    auto* temperature_tensor = context->Input<Tensor>(14);
    if (temperature_tensor) {
      if (temperature_tensor->IsDataType<float>()) {
        temperature = *temperature_tensor->Data<float>();
      } else {
        temperature = static_cast<float>(*temperature_tensor->Data<MLFloat16>());
      }
    } else {
      temperature = 1.0f;
    }
  }
}
void BeamSearchParameters::SetSubgraphParameters(int vocabulary_size, int heads, int hidden_size_per_head, int layers) {
  // Override vocab_size using the inferred shape from the decoder subgraph ONLY IF
  // the vocab_size hasn't been explicitly specified by the user (as an attribute of BeamSearch)
  if (vocab_size == -1 || vocab_size == 0) {
    vocab_size = vocabulary_size;
  }
  num_heads = heads;
  head_size = hidden_size_per_head;
  num_layers = layers;
}

void WhisperBeamSearchParameters::ParseFromAttributes(const OpKernelInfo& info) {
  BeamSearchParameters::ParseFromAttributes(info);
  model_type = static_cast<int>(info.GetAttrOrDefault<int64_t>("model_type", IGenerationParameters::kModelTypeWhisper));
  ORT_ENFORCE(model_type == IGenerationParameters::kModelTypeWhisper);

  // Token ids are defined below in the order that they appear in the tokenizer
  translate_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("translate_token_id", -1LL));
  transcribe_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("transcribe_token_id", -1LL));
  start_of_lm_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("start_of_lm_token_id", -1LL));
  no_speech_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_speech_token_id", -1LL));
  no_timestamps_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("no_timestamps_token_id", -1LL));
  beginning_timestamp_token_id = static_cast<int>(info.GetAttrOrDefault<int64_t>("beginning_timestamp_token_id", -1LL));
  cross_qk_layer_head_input_id = 12;
  extra_decoding_ids_input_id = 13;
  cross_qk_output_id = 3;
  no_speech_probs_output_id = 4;
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
