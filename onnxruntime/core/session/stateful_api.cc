// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

#include "core/framework/error_code_helper.h"

#if !defined(ORT_MINIMAL_BUILD)
#include <algorithm>
#include <array>
#include <cstring>
#include <variant>

#include "core/common/safeint.h"
#include "core/session/allocator_adapters.h"
#include "core/session/inference_session.h"
#include "core/session/session_chain.h"

using namespace onnxruntime;

struct OrtSessionLoopOptions {
  std::unique_ptr<SessionLoopOptions> value;
};
struct OrtSessionLoop {
  std::shared_ptr<const SessionLoop> value;
};
struct OrtSessionChainOptions {
  std::unique_ptr<SessionChainOptions> value;
};
struct OrtSessionChain {
  std::shared_ptr<const SessionChain> value;
};
struct OrtStatefulExecution {
  std::variant<std::unique_ptr<SessionLoopExecution>, std::unique_ptr<SessionChainExecution>> value;
  InlinedVector<std::string> loop_output_names;
  bool requires_reset = false;
};
struct OrtStatefulStepResult {
  struct Collection {
    InlinedVector<std::pair<std::string, OrtValue>> owned;
    InlinedVector<OrtStatefulNamedValue> borrowed;
    void Finalize() {
      for (const auto& [name, value] : owned) {
        borrowed.push_back({name.c_str(), &value});
      }
    }
  };
  OrtStatefulStepInfo info{};
  std::string component_name;
  InlinedVector<OrtStatefulExitKind> reasons;
  std::array<Collection, 3> collections;
};
struct OrtKVCacheOptions {
  std::unique_ptr<InlinedVector<KVCacheSpec>> value;
  StatefulProviderBinding provider;
};
struct OrtKVCacheManager {
  std::unique_ptr<KVCacheManager> value;
};
struct OrtKVCacheView {
  KVCacheView value;
};
struct OrtSamplerOptions {
  std::optional<int64_t> position;
  std::shared_ptr<StatefulDispatch> dispatch = std::make_shared<StatefulDispatch>();
};

namespace OrtStatefulAPI {
namespace {

Status Require(bool valid, std::string_view message) {
  if (!valid) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, message);
  }
  return Status::OK();
}

template <typename T>
struct V1Size;

// Appending fields must not increase the minimum size accepted from V1 callers.
#define V1_PREFIX(Type, Last)                                \
  template <>                                                \
  struct V1Size<Type> {                                      \
    static constexpr size_t value =                          \
        offsetof(Type, Last) + sizeof(decltype(Type::Last)); \
  };
V1_PREFIX(OrtStatefulLoopConfig, generation)
V1_PREFIX(OrtStatefulChainConfig, num_outputs)
V1_PREFIX(OrtStatefulInputs, num_initial_state)
V1_PREFIX(OrtStatefulKVCacheConfig, num_caches)
V1_PREFIX(OrtStatefulSamplerConfig, logits_position)
V1_PREFIX(OrtStatefulGenerationConfig, logits_position)
V1_PREFIX(OrtStatefulStepInfo, generated_token)
V1_PREFIX(OrtStatefulKVCacheStats, shared_updates)
V1_PREFIX(OrtStatefulProviderCapabilities, device_id)
V1_PREFIX(OrtStatefulDispatchInfo, failures)
V1_PREFIX(OrtStatefulStagingInfo, cpu_snapshot_copy_bytes)
#undef V1_PREFIX

template <typename T>
Status Sized(const T* config) {
  return Require(config && config->struct_size >= V1Size<T>::value,
                 "Configuration is NULL or its struct_size is too small");
}

template <typename T>
Status CopyInfo(T value, size_t size, T* out) {
  ORT_RETURN_IF_ERROR(Require(out && size >= V1Size<T>::value, "Output buffer is NULL or too small"));
  value.struct_size = std::min(size, sizeof(value));
  std::memcpy(out, &value, value.struct_size);
  return Status::OK();
}

template <typename T>
Status Range(const T* values, size_t count) {
  ORT_RETURN_IF_ERROR(Require(values || count == 0, "A nonempty array requires a non-NULL pointer"));
  ORT_UNUSED_PARAMETER(SafeInt<size_t>(count) * sizeof(T));
  return Status::OK();
}

template <typename T>
Status Enum(T value, int maximum) {
  return Require(static_cast<int64_t>(value) >= 0 && static_cast<int64_t>(value) <= maximum, "Unknown enum value");
}

Status Name(const char* name, std::string& out) {
  ORT_RETURN_IF_ERROR(Require(name && *name, "Names must be nonempty UTF-8 strings"));
  out = name;
  return Status::OK();
}

Status Names(const char* const* names, size_t count, InlinedVector<std::string>& out) {
  ORT_RETURN_IF_ERROR(Range(names, count));
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(Name(names[i], name));
    out.push_back(std::move(name));
  }
  return Status::OK();
}

Status CacheSpec(const OrtStatefulKVCacheSpec& config, KVCacheSpec& out) {
  ORT_RETURN_IF_ERROR(Name(config.name, out.name));
  ORT_RETURN_IF_ERROR(Range(config.shape, config.rank));
  out.element_type = static_cast<int32_t>(config.element_type);
  for (size_t i = 0; i < config.rank; ++i) {
    out.shape.push_back(config.shape[i]);
  }
  out.sequence_axis = config.sequence_axis;
  out.initial_capacity = config.initial_capacity;
  return KVCacheManager::ValidateSpecs(gsl::span<const KVCacheSpec>(&out, 1));
}

Status CacheBindings(const OrtStatefulKVCacheBinding* bindings, size_t count,
                     InlinedVector<LoopKVCacheBinding, 1>& out) {
  ORT_RETURN_IF_ERROR(Range(bindings, count));
  for (size_t i = 0; i < count; ++i) {
    const auto& input = bindings[i];
    LoopKVCacheBinding value;
    ORT_RETURN_IF_ERROR(CacheSpec(input.cache, value.cache));
    ORT_RETURN_IF_ERROR(Name(input.input_name, value.input_name));
    ORT_RETURN_IF_ERROR(Name(input.output_name, value.output_name));
    ORT_RETURN_IF_ERROR(Enum(input.update_kind, 2));
    value.update_kind = static_cast<KVCacheUpdateKind>(input.update_kind);
    out.push_back(std::move(value));
  }
  return Status::OK();
}

Status LoopConfig(const OrtStatefulLoopConfig* config, SessionLoopOptions& out) {
  ORT_RETURN_IF_ERROR(Sized(config));
  ORT_RETURN_IF_ERROR(Names(config->finite_input_names, config->num_finite_inputs, out.finite_input_names));
  ORT_RETURN_IF_ERROR(Names(config->output_names, config->num_outputs, out.output_names));
  ORT_RETURN_IF_ERROR(Range(config->feedback, config->num_feedback));
  for (size_t i = 0; i < config->num_feedback; ++i) {
    LoopFeedbackBinding value;
    ORT_RETURN_IF_ERROR(Name(config->feedback[i].input_name, value.input_name));
    ORT_RETURN_IF_ERROR(Name(config->feedback[i].output_name, value.output_name));
    out.feedback.push_back(std::move(value));
  }
  ORT_RETURN_IF_ERROR(Range(config->exit_conditions, config->num_exit_conditions));
  for (size_t i = 0; i < config->num_exit_conditions; ++i) {
    const auto& input = config->exit_conditions[i];
    ORT_RETURN_IF_ERROR(Enum(input.kind, 4));
    ORT_RETURN_IF_ERROR(Range(input.eos_token_ids, input.num_eos_token_ids));
    LoopExitCondition value{static_cast<LoopExitKind>(input.kind), input.limit, {}, {}};
    if (input.output_name) {
      value.output_name = input.output_name;
    }
    for (size_t j = 0; j < input.num_eos_token_ids; ++j) {
      value.eos_token_ids.push_back(input.eos_token_ids[j]);
    }
    ORT_RETURN_IF_ERROR(Require((input.kind == ORT_STATEFUL_MAX_ITERATIONS ||
                                 input.kind == ORT_STATEFUL_MAX_GENERATED_TOKENS)
                                    ? input.limit >= 0
                                    : input.limit == 0,
                                "Invalid or unused exit limit"));
    ORT_RETURN_IF_ERROR(Require(input.kind == ORT_STATEFUL_BOOLEAN_OUTPUT || value.output_name.empty(),
                                "Only a boolean exit names a model output"));
    ORT_RETURN_IF_ERROR(Require(input.kind == ORT_STATEFUL_EOS || value.eos_token_ids.empty(),
                                "Only an EOS exit supplies token IDs"));
    out.exit_conditions.push_back(std::move(value));
  }
  if (config->generated_token_output) {
    out.generated_token_output = config->generated_token_output;
  }
  if (config->nonempty_input) {
    out.nonempty_input = config->nonempty_input;
  }
  ORT_RETURN_IF_ERROR(CacheBindings(config->kv_caches, config->num_kv_caches, out.kv_caches));
  if (config->generation) {
    const auto* input = config->generation;
    ORT_RETURN_IF_ERROR(Sized(input));
    ORT_RETURN_IF_ERROR(Require(input->phase == ORT_STATEFUL_PREFILL || input->phase == ORT_STATEFUL_DECODE,
                                "Generation phase must be explicit"));
    ORT_RETURN_IF_ERROR(Require(input->has_logits_position == 0 || input->has_logits_position == 1,
                                "has_logits_position must be 0 or 1"));
    TokenGenerationOptions value;
    value.phase = input->phase == ORT_STATEFUL_PREFILL ? TokenGenerationPhase::Prefill : TokenGenerationPhase::Decode;
    ORT_RETURN_IF_ERROR(Name(input->input_ids, value.input_ids));
    ORT_RETURN_IF_ERROR(Name(input->logits_output, value.logits_output));
    if (input->attention_mask) {
      value.attention_mask = input->attention_mask;
    }
    if (input->position_ids) {
      value.position_ids = input->position_ids;
    }
    if (input->has_logits_position) {
      ORT_RETURN_IF_ERROR(Require(input->logits_position >= 0, "Logits position must be nonnegative"));
      value.logits_position = input->logits_position;
    }
    out.generation = std::move(value);
  }
  return Status::OK();
}

Status ChainConfig(const OrtStatefulChainConfig* config, SessionChainOptions& out) {
  ORT_RETURN_IF_ERROR(Sized(config));
  ORT_RETURN_IF_ERROR(Range(config->components, config->num_components));
  ORT_RETURN_IF_ERROR(Range(config->outputs, config->num_outputs));
  for (size_t i = 0; i < config->num_components; ++i) {
    const auto& input = config->components[i];
    SessionChainComponent value;
    ORT_RETURN_IF_ERROR(Name(input.name, value.name));
    ORT_RETURN_IF_ERROR(Require((input.session != nullptr) != (input.loop != nullptr),
                                "Exactly one Session or SessionLoop is required"));
    if (input.session) {
      value.execution = reinterpret_cast<InferenceSession*>(input.session);
    } else {
      value.execution = input.loop->value;
    }
    ORT_RETURN_IF_ERROR(Range(input.inputs, input.num_inputs));
    for (size_t j = 0; j < input.num_inputs; ++j) {
      ChainInputBinding binding;
      ORT_RETURN_IF_ERROR(Name(input.inputs[j].input_name, binding.input_name));
      const auto& source = input.inputs[j].source;
      ORT_RETURN_IF_ERROR(Enum(source.kind, 3));
      binding.source.kind = static_cast<ChainSourceKind>(source.kind);
      ORT_RETURN_IF_ERROR(Name(source.name, binding.source.name));
      binding.source.component_index = source.component_index;
      value.inputs.push_back(std::move(binding));
    }
    ORT_RETURN_IF_ERROR(Range(input.state_outputs, input.num_state_outputs));
    for (size_t j = 0; j < input.num_state_outputs; ++j) {
      ChainStateBinding binding;
      ORT_RETURN_IF_ERROR(Name(input.state_outputs[j].state_name, binding.state_name));
      ORT_RETURN_IF_ERROR(Name(input.state_outputs[j].output_name, binding.output_name));
      value.state_outputs.push_back(std::move(binding));
    }
    ORT_RETURN_IF_ERROR(CacheBindings(input.kv_caches, input.num_kv_caches, value.kv_caches));
    out.components.push_back(std::move(value));
  }
  for (size_t i = 0; i < config->num_outputs; ++i) {
    const auto& input = config->outputs[i];
    ORT_RETURN_IF_ERROR(Enum(input.kind, 1));
    ChainOutputBinding value;
    ORT_RETURN_IF_ERROR(Name(input.name, value.name));
    value.component_index = input.component_index;
    value.output_name = input.output_name ? input.output_name : "";
    value.kind = static_cast<ChainOutputKind>(input.kind);
    out.outputs.push_back(std::move(value));
  }
  return Status::OK();
}

Status CacheConfig(const OrtStatefulKVCacheConfig* config, InlinedVector<KVCacheSpec>& out) {
  ORT_RETURN_IF_ERROR(Sized(config));
  ORT_RETURN_IF_ERROR(Range(config->caches, config->num_caches));
  ORT_RETURN_IF_ERROR(Require(config->num_caches != 0, "Cache options require at least one layout"));
  for (size_t i = 0; i < config->num_caches; ++i) {
    KVCacheSpec spec;
    ORT_RETURN_IF_ERROR(CacheSpec(config->caches[i], spec));
    out.push_back(std::move(spec));
  }
  return KVCacheManager::ValidateSpecs(out);
}

Status SamplerConfig(const OrtStatefulSamplerConfig* config, std::optional<int64_t>& out) {
  ORT_RETURN_IF_ERROR(Sized(config));
  ORT_RETURN_IF_ERROR(Require(config->algorithm == ORT_STATEFUL_GREEDY, "Only greedy sampling is supported"));
  ORT_RETURN_IF_ERROR(Require(config->has_logits_position == 0 || config->has_logits_position == 1,
                              "has_logits_position must be 0 or 1"));
  if (config->has_logits_position) {
    ORT_RETURN_IF_ERROR(Require(config->logits_position >= 0, "Logits position must be nonnegative"));
    out = config->logits_position;
  }
  return Status::OK();
}

Status NamedInputs(const OrtStatefulNamedValue* values, size_t count, LoopFeeds& out) {
  ORT_RETURN_IF_ERROR(Range(values, count));
  for (size_t i = 0; i < count; ++i) {
    std::string name;
    ORT_RETURN_IF_ERROR(Name(values[i].name, name));
    ORT_RETURN_IF_ERROR(Require(values[i].value != nullptr, "Input OrtValue must be non-NULL"));
    ORT_RETURN_IF_ERROR(Require(out.emplace(std::move(name), *values[i].value).second, "Duplicate input name"));
  }
  return Status::OK();
}

Status LoopInputs(const OrtStatefulInputs* inputs, SessionLoopInputs& out) {
  ORT_RETURN_IF_ERROR(Sized(inputs));
  ORT_RETURN_IF_ERROR(Require(inputs->num_sequences == 0 && inputs->num_initial_state == 0,
                              "Loop inputs cannot contain chain sequences/state"));
  ORT_RETURN_IF_ERROR(NamedInputs(inputs->inputs, inputs->num_inputs, out.inputs));
  ORT_RETURN_IF_ERROR(Range(inputs->batches, inputs->num_batches));
  for (size_t i = 0; i < inputs->num_batches; ++i) {
    LoopFeeds batch;
    ORT_RETURN_IF_ERROR(NamedInputs(inputs->batches[i].values, inputs->batches[i].num_values, batch));
    out.batches.push_back(std::move(batch));
  }
  return Status::OK();
}

Status ChainInputs(const OrtStatefulInputs* inputs, SessionChainInputs& out) {
  ORT_RETURN_IF_ERROR(Sized(inputs));
  ORT_RETURN_IF_ERROR(Require(inputs->num_batches == 0, "Chain inputs cannot contain loop batches"));
  ORT_RETURN_IF_ERROR(NamedInputs(inputs->inputs, inputs->num_inputs, out.inputs));
  ORT_RETURN_IF_ERROR(NamedInputs(inputs->initial_state, inputs->num_initial_state, out.initial_state));
  ORT_RETURN_IF_ERROR(Range(inputs->sequences, inputs->num_sequences));
  for (size_t i = 0; i < inputs->num_sequences; ++i) {
    const auto& source = inputs->sequences[i];
    std::string name;
    ORT_RETURN_IF_ERROR(Name(source.name, name));
    ORT_RETURN_IF_ERROR(Range(source.values, source.num_values));
    InlinedVector<OrtValue> sequence;
    for (size_t j = 0; j < source.num_values; ++j) {
      ORT_RETURN_IF_ERROR(Require(source.values[j] != nullptr, "Sequence OrtValue must be non-NULL"));
      sequence.push_back(*source.values[j]);
    }
    ORT_RETURN_IF_ERROR(Require(out.sequences.emplace(std::move(name), std::move(sequence)).second,
                                "Duplicate sequence name"));
  }
  return Status::OK();
}

void CopyValues(const LoopFeeds& values, OrtStatefulStepResult::Collection& out) {
  for (const auto& [name, value] : values) {
    out.owned.emplace_back(name, value);
  }
  std::sort(out.owned.begin(), out.owned.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
}

void LoopInfo(const SessionLoopStepResult& source, OrtStatefulStepResult& result) {
  result.info.executed = source.executed;
  result.info.completed = source.completed;
  result.info.iterations = source.iterations;
  result.info.generated_tokens = source.generated_tokens;
  result.info.has_generated_token = source.generated_token.has_value();
  result.info.generated_token = source.generated_token.value_or(0);
  for (const auto reason : source.exit_reasons) {
    result.reasons.push_back(static_cast<OrtStatefulExitKind>(reason));
  }
}

Status Advance(OrtStatefulExecution& execution, bool run, std::unique_ptr<OrtStatefulStepResult>& out) {
  ORT_RETURN_IF_ERROR(Require(!execution.requires_reset, "Execution requires Reset"));
  execution.requires_reset = true;
  auto result = std::make_unique<OrtStatefulStepResult>();
  if (auto* loop = std::get_if<std::unique_ptr<SessionLoopExecution>>(&execution.value)) {
    SessionLoopStepResult value;
    ORT_RETURN_IF_ERROR(run ? (*loop)->Run(value) : (*loop)->Step(value));
    result->info.kind = ORT_STATEFUL_LOOP_RESULT;
    LoopInfo(value, *result);
    result->info.invocations = value.iterations;
    for (size_t i = 0; i < value.outputs.size(); ++i) {
      result->collections[0].owned.emplace_back(execution.loop_output_names[i], value.outputs[i]);
      result->collections[1].owned.emplace_back(execution.loop_output_names[i], value.outputs[i]);
    }
  } else {
    SessionChainStepResult value;
    auto& chain = std::get<std::unique_ptr<SessionChainExecution>>(execution.value);
    ORT_RETURN_IF_ERROR(run ? chain->Run(value) : chain->Step(value));
    result->info.kind = ORT_STATEFUL_CHAIN_RESULT;
    LoopInfo(value.component_result, *result);
    result->info.executed = value.executed;
    result->info.completed = value.completed;
    result->info.invocations = value.invocations;
    result->info.generated_tokens = value.generated_tokens;
    result->info.components_completed = value.components_completed;
    result->component_name = std::move(value.component_name);
    CopyValues(value.outputs, result->collections[0]);
    CopyValues(value.component_outputs, result->collections[1]);
  }
  for (auto& collection : result->collections) {
    collection.Finalize();
  }
  out = std::move(result);
  execution.requires_reset = false;
  return Status::OK();
}

}  // namespace

#define CHECK_POINTERS(condition) \
  ORT_API_RETURN_IF(!(condition), ORT_INVALID_ARGUMENT, "Required handle/configuration/output is NULL")

#define DEFINE_OPTIONS(Name, Config, Native, Parse, Reset)                         \
  ORT_API_STATUS_IMPL(Create##Name, const Config* config, Ort##Name** out) {       \
    API_IMPL_BEGIN                                                                 \
    CHECK_POINTERS(out);                                                           \
    auto value = std::make_unique<Native>();                                       \
    ORT_API_RETURN_IF_STATUS_NOT_OK(Parse(config, *value));                        \
    auto options = std::make_unique<Ort##Name>();                                  \
    options->value = std::move(value);                                             \
    *out = options.release();                                                      \
    return nullptr;                                                                \
    API_IMPL_END                                                                   \
  }                                                                                \
  ORT_API_STATUS_IMPL(Name##SetConfig, Ort##Name* options, const Config* config) { \
    API_IMPL_BEGIN                                                                 \
    CHECK_POINTERS(options);                                                       \
    auto value = std::make_unique<Native>();                                       \
    ORT_API_RETURN_IF_STATUS_NOT_OK(Parse(config, *value));                        \
    options->value = std::move(value);                                             \
    Reset;                                                                         \
    return nullptr;                                                                \
    API_IMPL_END                                                                   \
  }

DEFINE_OPTIONS(SessionLoopOptions, OrtStatefulLoopConfig, SessionLoopOptions, LoopConfig, {})
DEFINE_OPTIONS(SessionChainOptions, OrtStatefulChainConfig, SessionChainOptions, ChainConfig, {})
DEFINE_OPTIONS(KVCacheOptions, OrtStatefulKVCacheConfig, InlinedVector<KVCacheSpec>,
               CacheConfig, options->provider = {})

ORT_API_STATUS_IMPL(CreateSessionLoop,
                    OrtSession* session,
                    const OrtSessionLoopOptions* options,
                    OrtSessionLoop** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(session && options && out);
  auto value = std::make_unique<OrtSessionLoop>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(SessionLoop::Create(*reinterpret_cast<InferenceSession*>(session),
                                                      *options->value, value->value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateSessionChain, const OrtSessionChainOptions* options, OrtSessionChain** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options && out);
  auto value = std::make_unique<OrtSessionChain>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(SessionChain::Create(*options->value, value->value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateLoopExecution, const OrtSessionLoop* loop, const OrtStatefulInputs* inputs,
                    OrtStatefulExecution** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(loop && out);
  SessionLoopInputs feeds;
  ORT_API_RETURN_IF_STATUS_NOT_OK(LoopInputs(inputs, feeds));
  auto result = std::make_unique<OrtStatefulExecution>();
  result->loop_output_names = loop->value->Options().output_names;
  std::unique_ptr<SessionLoopExecution> value;
  ORT_API_RETURN_IF_STATUS_NOT_OK(loop->value->CreateExecution(std::move(feeds), value));
  result->value = std::move(value);
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateChainExecution, const OrtSessionChain* chain, const OrtStatefulInputs* inputs,
                    OrtStatefulExecution** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(chain && out);
  SessionChainInputs feeds;
  ORT_API_RETURN_IF_STATUS_NOT_OK(ChainInputs(inputs, feeds));
  auto result = std::make_unique<OrtStatefulExecution>();
  std::unique_ptr<SessionChainExecution> value;
  ORT_API_RETURN_IF_STATUS_NOT_OK(chain->value->CreateExecution(std::move(feeds), value));
  result->value = std::move(value);
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionStep, OrtStatefulExecution* execution, OrtStatefulStepResult** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  std::unique_ptr<OrtStatefulStepResult> result;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Advance(*execution, false, result));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionRun, OrtStatefulExecution* execution, OrtStatefulStepResult** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  std::unique_ptr<OrtStatefulStepResult> result;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Advance(*execution, true, result));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionReset, OrtStatefulExecution* execution) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution);
  execution->requires_reset = true;
  ORT_API_RETURN_IF_STATUS_NOT_OK(std::visit([](auto& value) { return value->Reset(); }, execution->value));
  execution->requires_reset = false;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionCancel, OrtStatefulExecution* execution) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution);
  std::visit([](auto& value) { value->Cancel(); }, execution->value);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionRequiresReset, const OrtStatefulExecution* execution, int* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  const bool invalid = execution->requires_reset ||
                       std::visit([](const auto& value) { return value->RequiresReset(); }, execution->value);
  *out = invalid;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionGetState, const OrtStatefulExecution* execution, OrtStatefulStepResult** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  int invalid = 0;
  ORT_API_RETURN_IF_ERROR(StatefulExecutionRequiresReset(execution, &invalid));
  ORT_API_RETURN_IF(invalid, ORT_FAIL, "State requires Reset after a failed execution");
  LoopFeeds state;
  if (const auto* loop = std::get_if<std::unique_ptr<SessionLoopExecution>>(&execution->value)) {
    ORT_API_RETURN_IF_STATUS_NOT_OK((*loop)->GetState(state));
  } else {
    state = std::get<std::unique_ptr<SessionChainExecution>>(execution->value)->State();
  }
  auto result = std::make_unique<OrtStatefulStepResult>();
  result->info.kind = ORT_STATEFUL_STATE_SNAPSHOT;
  CopyValues(state, result->collections[2]);
  result->collections[2].Finalize();
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulStepResultGetInfo, const OrtStatefulStepResult* result, size_t info_size,
                    OrtStatefulStepInfo* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(result && out);
  ORT_API_RETURN_IF(info_size < V1Size<OrtStatefulStepInfo>::value, ORT_INVALID_ARGUMENT, "Buffer is too small");
  auto info = result->info;
  info.struct_size = std::min(info_size, sizeof(info));
  info.component_name = result->component_name.c_str();
  info.exit_reasons = result->reasons.empty() ? nullptr : result->reasons.data();
  info.num_exit_reasons = result->reasons.size();
  std::memcpy(out, &info, info.struct_size);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulStepResultGetOutputs, const OrtStatefulStepResult* result, OrtStatefulOutputGroup group,
                    const OrtStatefulNamedValue** values, size_t* count) {
  API_IMPL_BEGIN
  CHECK_POINTERS(result && values && count);
  ORT_API_RETURN_IF(group < ORT_STATEFUL_SELECTED_OUTPUTS || group > ORT_STATEFUL_STATE_VALUES,
                    ORT_INVALID_ARGUMENT, "Unknown output collection");
  const auto& collection = result->collections[static_cast<size_t>(group)].borrowed;
  *values = collection.empty() ? nullptr : collection.data();
  *count = collection.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateKVCacheManager, const OrtKVCacheOptions* options, OrtAllocator* allocator,
                    OrtKVCacheManager** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options && allocator && out);
  ORT_API_RETURN_IF(!allocator->Alloc || !allocator->Free || !allocator->Info,
                    ORT_INVALID_ARGUMENT, "Allocator callbacks must be non-NULL");
  ORT_API_RETURN_IF(!allocator->Info(allocator), ORT_INVALID_ARGUMENT, "Allocator memory information is NULL");
  auto result = std::make_unique<OrtKVCacheManager>();
  auto adapter = std::make_shared<IAllocatorImplWrappingOrtAllocator>(allocator);
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      KVCacheManager::Create(*options->value, std::move(adapter), result->value, options->provider));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerGetView, const OrtKVCacheManager* manager, const char* name, OrtKVCacheView** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && name && out);
  auto result = std::make_unique<OrtKVCacheView>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(manager->value->Get(name, result->value));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheViewGetInfo, const OrtKVCacheView* view, const OrtValue** value,
                    int64_t* length, int64_t* capacity) {
  API_IMPL_BEGIN
  CHECK_POINTERS(view && value && length && capacity);
  *value = &view->value.value;
  *length = view->value.length;
  *capacity = view->value.capacity;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerUpdate, OrtKVCacheManager* manager, const char* name, const OrtValue* value,
                    OrtKVCacheUpdateKind kind) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && name && value);
  ORT_API_RETURN_IF_STATUS_NOT_OK(Enum(kind, 2));
  ORT_API_RETURN_IF_STATUS_NOT_OK(manager->value->Update(name, *value, static_cast<KVCacheUpdateKind>(kind)));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerReserve, OrtKVCacheManager* manager, const char* name, int64_t capacity) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && name);
  ORT_API_RETURN_IF_STATUS_NOT_OK(manager->value->Reserve(name, capacity));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerReset, OrtKVCacheManager* manager) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager);
  ORT_API_RETURN_IF_STATUS_NOT_OK(manager->value->Reset());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerGetStats, const OrtKVCacheManager* manager, size_t stats_size,
                    OrtStatefulKVCacheStats* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && out);
  ORT_API_RETURN_IF(stats_size < V1Size<OrtStatefulKVCacheStats>::value, ORT_INVALID_ARGUMENT, "Buffer is too small");
  const auto& stats = manager->value->Stats();
  const OrtStatefulKVCacheStats result{std::min(stats_size, sizeof(OrtStatefulKVCacheStats)), stats.allocations,
                                       stats.bytes_copied, stats.shared_updates};
  std::memcpy(out, &result, result.struct_size);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(CreateSamplerOptions, const OrtStatefulSamplerConfig* config, OrtSamplerOptions** out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(out);
  auto result = std::make_unique<OrtSamplerOptions>();
  ORT_API_RETURN_IF_STATUS_NOT_OK(SamplerConfig(config, result->position));
  *out = result.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SamplerOptionsSetConfig, OrtSamplerOptions* options, const OrtStatefulSamplerConfig* config) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options);
  std::optional<int64_t> position;
  ORT_API_RETURN_IF_STATUS_NOT_OK(SamplerConfig(config, position));
  auto dispatch = std::make_shared<StatefulDispatch>();
  options->position = position;
  options->dispatch = std::move(dispatch);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SamplerSample, const OrtSamplerOptions* options, const OrtValue* logits, int64_t* token) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options && logits && token);
  int64_t value = 0;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Sampler::Sample(*logits, options->position, *options->dispatch, value));
  *token = value;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SessionLoopOptionsSetProvider, OrtSessionLoopOptions* options,
                    OrtStatefulProviderFeature feature, const OrtStatefulProviderSelection* selection) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(Enum(feature, 1));
  StatefulProviderSelection parsed;
  ORT_API_RETURN_IF_STATUS_NOT_OK(StatefulProviderSelection::Parse(selection, parsed));
  (feature == ORT_STATEFUL_FEATURE_CACHE ? options->value->cache_provider : options->value->sampler_provider) =
      std::move(parsed);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SessionChainOptionsSetComponentProvider, OrtSessionChainOptions* options, size_t component,
                    OrtStatefulProviderFeature feature, const OrtStatefulProviderSelection* selection) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(Enum(feature, 1));
  ORT_API_RETURN_IF(component >= options->value->components.size(), ORT_INVALID_ARGUMENT, "Invalid component index");
  StatefulProviderSelection parsed;
  ORT_API_RETURN_IF_STATUS_NOT_OK(StatefulProviderSelection::Parse(selection, parsed));
  auto& value = options->value->components[component];
  (feature == ORT_STATEFUL_FEATURE_CACHE ? value.cache_provider : value.sampler_provider) = std::move(parsed);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheOptionsSetProvider, OrtKVCacheOptions* options, OrtSession* session,
                    const OrtStatefulProviderSelection* selection) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options);
  StatefulProviderSelection parsed;
  ORT_API_RETURN_IF_STATUS_NOT_OK(StatefulProviderSelection::Parse(selection, parsed));
  StatefulProviderBinding binding;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      StatefulProviderBinding::Resolve(reinterpret_cast<InferenceSession*>(session), parsed, binding));
  options->provider = std::move(binding);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SamplerOptionsSetProvider, OrtSamplerOptions* options, OrtSession* session,
                    const OrtStatefulProviderSelection* selection) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options);
  StatefulProviderSelection parsed;
  ORT_API_RETURN_IF_STATUS_NOT_OK(StatefulProviderSelection::Parse(selection, parsed));
  StatefulProviderBinding binding;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      StatefulProviderBinding::Resolve(reinterpret_cast<InferenceSession*>(session), parsed, binding));
  auto dispatch = std::make_shared<StatefulDispatch>(std::move(binding));
  bool provider = false;
  const auto types = dispatch->Binding().Descriptor().capabilities.sampler_element_types;
  const int element_type = (types & (uint64_t{1} << ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) != 0
                               ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
                               : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  ORT_API_RETURN_IF_STATUS_NOT_OK(
      dispatch->Select(ORT_STATEFUL_SAMPLE_GREEDY, element_type, true, provider));
  options->dispatch = std::move(dispatch);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(QueryStatefulProviderCapabilities, const OrtSession* session, const char* name,
                    size_t size, OrtStatefulProviderCapabilities* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(session && name && out);
  OrtStatefulProviderCapabilities caps{};
  auto& native = *const_cast<InferenceSession*>(reinterpret_cast<const InferenceSession*>(session));
  ORT_API_RETURN_IF_STATUS_NOT_OK(StatefulProviderBinding::Query(native, name, caps));
  ORT_API_RETURN_IF_STATUS_NOT_OK(CopyInfo(caps, size, out));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionGetDispatchInfo, const OrtStatefulExecution* execution,
                    OrtStatefulProviderFeature feature, size_t component, OrtStatefulProviderOperation operation,
                    size_t size, OrtStatefulDispatchInfo* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  OrtStatefulDispatchInfo info{};
  ORT_API_RETURN_IF_STATUS_NOT_OK(std::visit([&](const auto& value) {
    return value->GetDispatchInfo(feature, component, operation, info);
  },
                                             execution->value));
  ORT_API_RETURN_IF_STATUS_NOT_OK(CopyInfo(info, size, out));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerGetDispatchInfo, const OrtKVCacheManager* manager,
                    OrtStatefulProviderOperation operation, size_t size, OrtStatefulDispatchInfo* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && out);
  ORT_API_RETURN_IF(operation < ORT_STATEFUL_CACHE_INITIALIZE || operation > ORT_STATEFUL_CACHE_RESET,
                    ORT_INVALID_ARGUMENT, "Invalid cache operation");
  OrtStatefulDispatchInfo info{};
  ORT_API_RETURN_IF_STATUS_NOT_OK(manager->value->Dispatch().GetInfo(operation, info));
  ORT_API_RETURN_IF_STATUS_NOT_OK(CopyInfo(info, size, out));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(SamplerOptionsGetDispatchInfo, const OrtSamplerOptions* options,
                    size_t size, OrtStatefulDispatchInfo* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(options && out);
  OrtStatefulDispatchInfo info{};
  ORT_API_RETURN_IF_STATUS_NOT_OK(options->dispatch->GetInfo(ORT_STATEFUL_SAMPLE_GREEDY, info));
  ORT_API_RETURN_IF_STATUS_NOT_OK(CopyInfo(info, size, out));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(KVCacheManagerRequiresReset, const OrtKVCacheManager* manager, int* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(manager && out);
  *out = manager->value->RequiresReset();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(StatefulExecutionGetStagingInfo, const OrtStatefulExecution* execution,
                    size_t component, size_t size, OrtStatefulStagingInfo* out) {
  API_IMPL_BEGIN
  CHECK_POINTERS(execution && out);
  OrtStatefulStagingInfo info{};
  ORT_API_RETURN_IF_STATUS_NOT_OK(std::visit([&](const auto& value) {
    return value->GetStagingInfo(component, info);
  },
                                             execution->value));
  ORT_API_RETURN_IF_STATUS_NOT_OK(CopyInfo(info, size, out));
  return nullptr;
  API_IMPL_END
}

#define DEFINE_RELEASE(Name) \
  ORT_API(void, Release##Name, Ort##Name* input) { delete input; }
DEFINE_RELEASE(SessionLoopOptions)
DEFINE_RELEASE(SessionLoop)
DEFINE_RELEASE(SessionChainOptions)
DEFINE_RELEASE(SessionChain)
DEFINE_RELEASE(StatefulExecution)
DEFINE_RELEASE(StatefulStepResult)
DEFINE_RELEASE(KVCacheOptions)
DEFINE_RELEASE(KVCacheManager)
DEFINE_RELEASE(KVCacheView)
DEFINE_RELEASE(SamplerOptions)

static constexpr OrtStatefulApi stateful_api = {
    &CreateSessionLoopOptions,
    &SessionLoopOptionsSetConfig,
    &ReleaseSessionLoopOptions,
    &CreateSessionLoop,
    &ReleaseSessionLoop,
    &CreateSessionChainOptions,
    &SessionChainOptionsSetConfig,
    &ReleaseSessionChainOptions,
    &CreateSessionChain,
    &ReleaseSessionChain,
    &CreateLoopExecution,
    &CreateChainExecution,
    &ReleaseStatefulExecution,
    &StatefulExecutionStep,
    &StatefulExecutionRun,
    &StatefulExecutionReset,
    &StatefulExecutionCancel,
    &StatefulExecutionRequiresReset,
    &StatefulExecutionGetState,
    &ReleaseStatefulStepResult,
    &StatefulStepResultGetInfo,
    &StatefulStepResultGetOutputs,
    &CreateKVCacheOptions,
    &KVCacheOptionsSetConfig,
    &ReleaseKVCacheOptions,
    &CreateKVCacheManager,
    &ReleaseKVCacheManager,
    &KVCacheManagerGetView,
    &ReleaseKVCacheView,
    &KVCacheViewGetInfo,
    &KVCacheManagerUpdate,
    &KVCacheManagerReserve,
    &KVCacheManagerReset,
    &KVCacheManagerGetStats,
    &CreateSamplerOptions,
    &SamplerOptionsSetConfig,
    &ReleaseSamplerOptions,
    &SamplerSample,
    &SessionLoopOptionsSetProvider,
    &SessionChainOptionsSetComponentProvider,
    &KVCacheOptionsSetProvider,
    &SamplerOptionsSetProvider,
    &QueryStatefulProviderCapabilities,
    &StatefulExecutionGetDispatchInfo,
    &KVCacheManagerGetDispatchInfo,
    &SamplerOptionsGetDispatchInfo,
    &KVCacheManagerRequiresReset,
    &StatefulExecutionGetStagingInfo,
};

#undef CHECK_POINTERS
#undef DEFINE_OPTIONS
#undef DEFINE_RELEASE
}  // namespace OrtStatefulAPI
#endif  // !ORT_MINIMAL_BUILD

ORT_API(const OrtStatefulApi*, OrtApis::GetStatefulApi) {
#if !defined(ORT_MINIMAL_BUILD)
  return &OrtStatefulAPI::stateful_api;
#else
  return nullptr;
#endif
}
