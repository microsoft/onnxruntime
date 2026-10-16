// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/generate/token_generation.h"

#include <algorithm>
#include <cmath>

#include "core/common/safeint.h"
#include "core/framework/stateful_execution_utils.h"

namespace onnxruntime {
namespace {

Status AllocateIds(const TensorShape& shape, AllocatorPtr allocator, OrtValue& out) {
  ORT_RETURN_IF_NOT(allocator && allocator->Info().device.Type() == OrtDevice::CPU,
                    "Token-generation buffers require a CPU allocator");
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), shape, std::move(allocator), out);
  ORT_RETURN_IF(shape.Size() != 0 && out.Get<Tensor>().DataRaw() == nullptr,
                "CPU token-generation buffer allocation failed");
  return Status::OK();
}

template <typename T>
Status SelectToken(const Tensor& tensor, size_t offset, size_t width, int64_t& token) {
  const T* data = tensor.Data<T>();
  const size_t count = SafeInt<size_t>(tensor.Shape().Size());
  for (size_t i = 0; i < count; ++i) {
    ORT_RETURN_IF(std::isnan(static_cast<float>(data[i])), "Greedy sampling does not accept NaN logits");
  }
  const T* begin = data + offset;
  const auto less = [](const T& a, const T& b) { return static_cast<float>(a) < static_cast<float>(b); };
  const T* best = std::max_element(begin, begin + width, less);
  token = SafeInt<int64_t>(best - begin);
  return Status::OK();
}

Status CheckInputPort(const NodeArg* argument, bool single_token) {
  ORT_RETURN_IF_NOT(argument, "Unknown token-generation input port");
  const auto* type = argument->TypeAsProto();
  ORT_RETURN_IF_NOT(type && type->has_tensor_type() &&
                        type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64,
                    "Token-generation inputs require int64 tensors: ", argument->Name());
  const auto* shape = argument->Shape();
  if (shape) {
    ORT_RETURN_IF(shape->dim_size() != 2, "Token-generation inputs require rank two: ", argument->Name());
    ORT_RETURN_IF(shape->dim(0).has_dim_value() && shape->dim(0).dim_value() != 1,
                  "Token generation supports one sequence per context");
    ORT_RETURN_IF(shape->dim(1).has_dim_value() &&
                      (single_token ? shape->dim(1).dim_value() != 1 : shape->dim(1).dim_value() <= 0),
                  "Incompatible token-generation input sequence dimension: ", argument->Name());
  }
  return Status::OK();
}

}  // namespace

Status Sampler::Greedy(const OrtValue& logits, std::optional<int64_t> position, int64_t& token) {
  ORT_RETURN_IF_NOT(logits.IsTensor(), "Greedy sampling requires a logits tensor");
  const Tensor& tensor = logits.Get<Tensor>();
  ORT_RETURN_IF_NOT(tensor.Location().device.Type() == OrtDevice::CPU,
                    "CPU greedy sampling requires CPU logits; native device sampling is unavailable");
#ifdef ENABLE_STRIDED_TENSORS
  ORT_RETURN_IF_NOT(tensor.IsContiguous(), "Greedy sampling requires contiguous logits");
#endif
  ORT_RETURN_IF_NOT(tensor.IsDataType<float>() || tensor.IsDataType<MLFloat16>(),
                    "Greedy sampling supports float32 and float16 logits");
  const auto& shape = tensor.Shape();
  ORT_RETURN_IF((shape.NumDimensions() != 2 && shape.NumDimensions() != 3) || shape[0] != 1,
                "Greedy logits must have shape [1, vocabulary] or [1, sequence, vocabulary]");
  const int64_t sequence = shape.NumDimensions() == 3 ? shape[1] : 1;
  const int64_t width = shape[shape.NumDimensions() - 1];
  ORT_RETURN_IF(sequence <= 0 || width <= 0, "Greedy logits sequence and vocabulary must be nonempty");
  const int64_t selected = position.value_or(sequence - 1);
  ORT_RETURN_IF(selected < 0 || selected >= sequence, "Greedy logits position is out of range");
  ORT_UNUSED_PARAMETER(tensor.SizeInBytes());
  ORT_RETURN_IF_NOT(tensor.DataRaw(), "Greedy logits buffer is null");
  const size_t offset = SafeInt<size_t>(selected) * width;
  if (tensor.IsDataType<float>()) {
    return SelectToken<float>(tensor, offset, SafeInt<size_t>(width), token);
  }
  return SelectToken<MLFloat16>(tensor, offset, SafeInt<size_t>(width), token);
}

InlinedVector<std::string> TokenGeneration::ManagedInputs(const TokenGenerationOptions& options) {
  InlinedVector<std::string> names;
  if (options.phase == TokenGenerationPhase::Decode) {
    names.push_back(options.input_ids);
  }
  for (const auto* name : {&options.attention_mask, &options.position_ids}) {
    if (!name->empty()) {
      names.push_back(*name);
    }
  }
  return names;
}

Status TokenGeneration::Validate(const TokenGenerationOptions& options,
                                 gsl::span<const NodeArg* const> inputs, gsl::span<const NodeArg* const> outputs) {
  using stateful_execution::FindArgument;
  ORT_RETURN_IF(options.phase != TokenGenerationPhase::Prefill && options.phase != TokenGenerationPhase::Decode,
                "Unknown token-generation phase");
  ORT_RETURN_IF(options.input_ids.empty() || options.logits_output.empty(), "Token-generation ports must be explicit");
  const bool decode = options.phase == TokenGenerationPhase::Decode;
  ORT_RETURN_IF_ERROR(CheckInputPort(FindArgument(inputs, options.input_ids), decode));
  InlinedHashSet<std::string> names{options.input_ids};
  for (const auto* name : {&options.attention_mask, &options.position_ids}) {
    if (!name->empty()) {
      ORT_RETURN_IF_NOT(names.insert(*name).second, "Duplicate managed token-generation input");
      ORT_RETURN_IF_ERROR(CheckInputPort(FindArgument(inputs, *name), decode && name == &options.position_ids));
    }
  }
  const auto* logits = FindArgument(outputs, options.logits_output);
  ORT_RETURN_IF_NOT(logits, "Unknown generation logits output");
  const auto* type = logits->TypeAsProto();
  ORT_RETURN_IF_NOT(type && type->has_tensor_type() &&
                        (type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
                         type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16),
                    "Generation logits require float32 or float16");
  const auto* shape = logits->Shape();
  if (shape) {
    ORT_RETURN_IF((shape->dim_size() != 2 && shape->dim_size() != 3) ||
                      (shape->dim(0).has_dim_value() && shape->dim(0).dim_value() != 1),
                  "Generation logits require a single sequence and rank two or three");
    for (int i = 1; i < shape->dim_size(); ++i) {
      ORT_RETURN_IF(shape->dim(i).has_dim_value() && shape->dim(i).dim_value() <= 0,
                    "Generation logits dimensions must be nonempty");
    }
    if (options.logits_position && (shape->dim_size() == 2 || shape->dim(1).has_dim_value())) {
      const int64_t sequence = shape->dim_size() == 2 ? 1 : shape->dim(1).dim_value();
      ORT_RETURN_IF(*options.logits_position >= sequence, "Generation logits position exceeds the known sequence");
    }
  }
  ORT_RETURN_IF(options.logits_position.has_value() && *options.logits_position < 0,
                "Generation logits position must be nonnegative");
  return Status::OK();
}

Status TokenGeneration::ValidatePrompt(const OrtValue& prompt) {
  ORT_RETURN_IF_NOT(prompt.IsTensor(), "Prefill prompt must be a tensor");
  const Tensor& tensor = prompt.Get<Tensor>();
#ifdef ENABLE_STRIDED_TENSORS
  ORT_RETURN_IF_NOT(tensor.IsContiguous(), "Prefill prompt must be contiguous");
#endif
  ORT_RETURN_IF_NOT(tensor.IsDataType<int64_t>() && tensor.Location().device.Type() == OrtDevice::CPU &&
                        tensor.Shape().NumDimensions() == 2 && tensor.Shape()[0] == 1 && tensor.Shape()[1] > 0,
                    "Prefill requires a nonempty CPU int64 prompt shaped [1, sequence]");
  ORT_UNUSED_PARAMETER(tensor.SizeInBytes());
  ORT_RETURN_IF_NOT(tensor.DataRaw(), "Prefill prompt buffer is null");
  const int64_t* ids = tensor.Data<int64_t>();
  for (int64_t i = 0; i < tensor.Shape()[1]; ++i) {
    ORT_RETURN_IF(ids[i] < 0, "Prompt token IDs must be nonnegative");
  }
  return Status::OK();
}

Status TokenGeneration::PrepareInputs(const TokenGenerationOptions& options, const TokenGenerationState& state,
                                      const OrtValue* prompt, AllocatorPtr allocator,
                                      InlinedHashMap<std::string, OrtValue>& inputs, int64_t& input_count) {
  int64_t count = 1;
  InlinedHashMap<std::string, OrtValue> prepared;
  ORT_RETURN_IF(options.phase != TokenGenerationPhase::Prefill && options.phase != TokenGenerationPhase::Decode,
                "Token-generation phase must be explicit");
  ORT_RETURN_IF(state.processed_tokens < 0, "Processed token count must be nonnegative");
  if (options.phase == TokenGenerationPhase::Prefill) {
    ORT_RETURN_IF_NOT(prompt, "Prefill requires a prompt");
    ORT_RETURN_IF_ERROR(ValidatePrompt(*prompt));
    ORT_RETURN_IF(state.prefill_complete || !state.token_ids.empty() || state.processed_tokens != 0,
                  "Prefill generation state must be empty");
    count = prompt->Get<Tensor>().Shape()[1];
  } else {
    ORT_RETURN_IF(!state.prefill_complete || state.token_ids.empty(),
                  "Decode requires a completed prefill in the same execution context");
    ORT_RETURN_IF(state.token_ids.back() < 0, "Pending token ID must be nonnegative");
    OrtValue ids;
    ORT_RETURN_IF_ERROR(AllocateIds(TensorShape{1, 1}, allocator, ids));
    *ids.GetMutable<Tensor>()->MutableData<int64_t>() = state.token_ids.back();
    prepared.emplace(options.input_ids, std::move(ids));
  }
  const int64_t total = SafeInt<int64_t>(state.processed_tokens) + count;
  if (!options.attention_mask.empty()) {
    OrtValue mask;
    ORT_RETURN_IF_ERROR(AllocateIds(TensorShape{1, total}, allocator, mask));
    auto* data = mask.GetMutable<Tensor>()->MutableData<int64_t>();
    std::fill_n(data, static_cast<size_t>(SafeInt<size_t>(total)), int64_t{1});
    prepared.emplace(options.attention_mask, std::move(mask));
  }
  if (!options.position_ids.empty()) {
    OrtValue positions;
    ORT_RETURN_IF_ERROR(AllocateIds(TensorShape{1, count}, allocator, positions));
    auto* data = positions.GetMutable<Tensor>()->MutableData<int64_t>();
    for (int64_t i = 0; i < count; ++i) {
      data[i] = SafeInt<int64_t>(state.processed_tokens) + i;
    }
    prepared.emplace(options.position_ids, std::move(positions));
  }
  inputs = std::move(prepared);
  input_count = count;
  return Status::OK();
}

OrtValue TokenGeneration::OutputTokens(const TokenGenerationState& state, AllocatorPtr allocator) {
  OrtValue result;
  ORT_THROW_IF_ERROR(AllocateIds(TensorShape{SafeInt<int64_t>(state.token_ids.size())}, std::move(allocator), result));
  if (!state.token_ids.empty()) {
    std::copy(state.token_ids.begin(), state.token_ids.end(), result.GetMutable<Tensor>()->MutableData<int64_t>());
  }
  return result;
}

}  // namespace onnxruntime
