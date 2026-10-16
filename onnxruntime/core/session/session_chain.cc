// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/session_chain.h"

#include <optional>

#include "core/common/safeint.h"
#include "core/session/inference_session.h"
#include "core/session/IOBinding.h"
#include "core/session/stateful_execution_utils.h"

namespace onnxruntime {
using namespace stateful_execution;

namespace {

Status CheckHostInput(const NodeArg& argument, const OrtValue& value) {
  ORT_RETURN_IF_ERROR(CheckInput(argument, value));
  ORT_RETURN_IF_NOT(value.Get<Tensor>().Location().device.UsesCpuMemory(),
                    "SessionChain currently requires host-accessible inputs: ", argument.Name());
  return Status::OK();
}

bool IsFinite(const SessionLoop& loop, std::string_view input_name) {
  const auto& names = loop.Options().finite_input_names;
  return std::find(names.begin(), names.end(), input_name) != names.end();
}

}  // namespace

SessionChain::SessionChain(SessionChainOptions options) : options_(std::move(options)) {}

Status SessionChain::Create(SessionChainOptions options, std::shared_ptr<const SessionChain>& out) {
  auto chain = std::shared_ptr<SessionChain>(new SessionChain(std::move(options)));
  ORT_RETURN_IF_ERROR(chain->Initialize());
  out = std::move(chain);
  return Status::OK();
}

Status SessionChain::Initialize() {
  ORT_RETURN_IF(options_.components.empty() || options_.outputs.empty(),
                "SessionChain requires components and selected output ports");
  InlinedHashSet<std::string> names;
  bool seen_decode = false;
  std::optional<int64_t> generation_budget;
  InlinedHashSet<std::string> generation_caches;
  for (const auto& component : options_.components) {
    ORT_RETURN_IF(component.name.empty() || !names.insert(component.name).second,
                  "Empty or duplicate SessionChain component name");
    InferenceSession* session = nullptr;
    SessionLoopOptions options;
    if (const auto* plain = std::get_if<InferenceSession*>(&component.execution)) {
      session = *plain;
      options.exit_conditions = {{LoopExitKind::MaxIterations, 1, {}, {}}};
    } else {
      const auto& loop = std::get<std::shared_ptr<const SessionLoop>>(component.execution);
      ORT_RETURN_IF_NOT(loop, "Null SessionLoop component: ", component.name);
      session = &loop->session_;
      options = loop->Options();
    }
    ORT_RETURN_IF_NOT(session, "Null Session component: ", component.name);
    options.kv_caches.insert(options.kv_caches.end(), component.kv_caches.begin(), component.kv_caches.end());
    const auto [status, outputs] = session->GetModelOutputs();
    ORT_RETURN_IF_ERROR(status);
    options.output_names.clear();
    for (const auto* output : *outputs) {
      options.output_names.push_back(output->Name());
    }
    std::shared_ptr<const SessionLoop> loop;
    ORT_RETURN_IF_ERROR(SessionLoop::Create(*session, std::move(options), loop));
    if (loop->Options().generation) {
      const auto phase = loop->Options().generation->phase;
      std::optional<int64_t> budget;
      for (const auto& condition : loop->Options().exit_conditions) {
        if (condition.kind == LoopExitKind::MaxGeneratedTokens) {
          budget = condition.limit;
        }
      }
      if (phase == TokenGenerationPhase::Prefill) {
        ORT_RETURN_IF(has_generation_, "A chain initially supports one prefill/decode generation pipeline");
        has_generation_ = true;
        generation_budget = budget;
        for (const auto& item : loop->Options().kv_caches) {
          generation_caches.insert(item.cache.name);
        }
      } else {
        ORT_RETURN_IF(!has_generation_ || seen_decode, "Decode must follow one prefill in the same chain");
        ORT_RETURN_IF(generation_budget != budget,
                      "Prefill and decode must declare the same explicit generated-token budget");
        ORT_RETURN_IF(generation_caches.size() != loop->Options().kv_caches.size(),
                      "Prefill/decode managed cache sets differ");
        for (const auto& item : loop->Options().kv_caches) {
          ORT_RETURN_IF_NOT(generation_caches.contains(item.cache.name), "Prefill/decode managed cache names differ");
        }
        seen_decode = true;
      }
    }
    for (const auto& item : loop->Options().kv_caches) {
      auto found = std::find_if(cache_specs_.begin(), cache_specs_.end(),
                                [&](const auto& spec) { return spec.name == item.cache.name; });
      if (found == cache_specs_.end()) {
        cache_specs_.push_back(item.cache);
      } else {
        ORT_RETURN_IF(found->element_type != item.cache.element_type || found->shape != item.cache.shape ||
                          found->sequence_axis != item.cache.sequence_axis ||
                          found->initial_capacity != item.cache.initial_capacity,
                      "Inconsistent shared KV cache layout: ", item.cache.name);
      }
    }
    loops_.push_back(std::move(loop));
    sessions_.push_back(session);
  }

  InlinedHashMap<std::string, const NodeArg*> state_types;
  for (size_t index = 0; index < options_.components.size(); ++index) {
    const auto& component = options_.components[index];
    const auto& arguments = *sessions_[index]->GetModelInputs().second;
    const auto managed = loops_[index]->Options().generation
                             ? TokenGeneration::ManagedInputs(*loops_[index]->Options().generation)
                             : InlinedVector<std::string>{};
    const size_t managed_count = loops_[index]->Options().kv_caches.size() + managed.size();
    ORT_RETURN_IF(component.inputs.size() + managed_count != arguments.size(),
                  "Component input binding count mismatch: ", component.name);
    InlinedHashSet<std::string> destinations;
    for (const auto& binding : component.inputs) {
      const auto* target = FindArgument(arguments, binding.input_name);
      ORT_RETURN_IF_NOT(target && destinations.insert(binding.input_name).second,
                        "Unknown or duplicate component input: ", component.name, ".", binding.input_name);
      ORT_RETURN_IF(std::find(managed.begin(), managed.end(), binding.input_name) != managed.end(),
                    "Managed generation input cannot also have a chain binding");
      ORT_RETURN_IF(std::any_of(loops_[index]->Options().kv_caches.begin(), loops_[index]->Options().kv_caches.end(),
                                [&](const auto& item) { return item.input_name == binding.input_name; }),
                    "Managed KV cache input cannot also have a chain input binding: ", binding.input_name);
      const auto& source = binding.source;
      ORT_RETURN_IF(source.name.empty(), "Empty component input source: ", component.name);
      const NodeArg* source_type = nullptr;
      switch (source.kind) {
        case ChainSourceKind::ExternalInput:
          external_names_.insert(source.name);
          break;
        case ChainSourceKind::ExternalSequence:
          ORT_RETURN_IF_NOT(IsFinite(*loops_[index], binding.input_name),
                            "External sequences require a finite loop input: ", component.name);
          sequence_names_.insert(source.name);
          break;
        case ChainSourceKind::ComponentOutput:
          ORT_RETURN_IF(source.component_index >= index,
                        "Binding must reference an earlier component: ", component.name);
          source_type = FindArgument(*sessions_[source.component_index]->GetModelOutputs().second, source.name);
          ORT_RETURN_IF_NOT(source_type, "Unknown upstream output: ", source.name);
          break;
        case ChainSourceKind::State: {
          auto cache = std::find_if(cache_specs_.begin(), cache_specs_.end(),
                                    [&](const auto& spec) { return spec.name == source.name; });
          if (cache != cache_specs_.end()) {
            ORT_RETURN_IF_ERROR(KVCacheManager::CheckBinding(*cache, *target, true));
            break;
          }
          auto [it, inserted] = state_types.emplace(source.name, target);
          if (inserted) {
            initial_state_names_.insert(source.name);
          }
          source_type = it->second;
          break;
        }
        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown SessionChain source kind");
      }
      if (source_type) {
        ORT_RETURN_IF_ERROR(CheckCompatibleTypes(*source_type, *target));
      }
    }
    InlinedHashSet<std::string> state_destinations;
    for (const auto& binding : component.state_outputs) {
      const auto* output = FindArgument(*sessions_[index]->GetModelOutputs().second, binding.output_name);
      ORT_RETURN_IF_NOT(output && !binding.state_name.empty() && state_destinations.insert(binding.state_name).second,
                        "Invalid or duplicate state output: ", component.name, ".", binding.state_name);
      auto cache = std::find_if(cache_specs_.begin(), cache_specs_.end(),
                                [&](const auto& spec) { return spec.name == binding.state_name; });
      if (cache != cache_specs_.end()) {
        ORT_RETURN_IF_ERROR(KVCacheManager::CheckBinding(*cache, *output, false));
        ORT_RETURN_IF(std::any_of(loops_[index]->Options().kv_caches.begin(), loops_[index]->Options().kv_caches.end(),
                                  [&](const auto& item) { return item.cache.name == binding.state_name; }),
                      "Managed KV output cannot also be written through a state binding: ", binding.state_name);
        continue;
      }
      auto it = state_types.find(binding.state_name);
      if (it != state_types.end()) {
        ORT_RETURN_IF_ERROR(CheckCompatibleTypes(*output, *it->second));
      }
      state_types.insert_or_assign(binding.state_name, output);
    }
  }
  InlinedHashSet<std::string> outputs;
  for (const auto& binding : options_.outputs) {
    ORT_RETURN_IF(binding.name.empty() || !outputs.insert(binding.name).second,
                  "Invalid or duplicate chain output port: ", binding.name);
    if (binding.kind == ChainOutputKind::GeneratedTokenIds) {
      ORT_RETURN_IF(!has_generation_ || !binding.output_name.empty(),
                    "Generated-token output ports require native generation and no model output name");
      continue;
    }
    ORT_RETURN_IF(binding.kind != ChainOutputKind::ModelOutput || binding.component_index >= sessions_.size(),
                  "Invalid chain output kind or component index");
    ORT_RETURN_IF_NOT(FindArgument(*sessions_[binding.component_index]->GetModelOutputs().second, binding.output_name),
                      "Unknown chain output: ", binding.output_name);
  }
  return Status::OK();
}

Status SessionChain::ValidateInputs(const SessionChainInputs& inputs) const {
  ORT_RETURN_IF(inputs.inputs.size() != external_names_.size() || inputs.sequences.size() != sequence_names_.size() ||
                    inputs.initial_state.size() != initial_state_names_.size(),
                "SessionChain external input, sequence, or initial state count mismatch");
  InlinedHashSet<std::string> produced_state;
  for (const auto& spec : cache_specs_) {
    produced_state.insert(spec.name);
  }
  for (size_t index = 0; index < options_.components.size(); ++index) {
    std::optional<size_t> batch_count;
    for (const auto& binding : options_.components[index].inputs) {
      const auto& source = binding.source;
      const auto& target = *FindArgument(*sessions_[index]->GetModelInputs().second, binding.input_name);
      const OrtValue* value = nullptr;
      size_t count = 1;
      if (source.kind == ChainSourceKind::ExternalSequence) {
        auto it = inputs.sequences.find(source.name);
        ORT_RETURN_IF(it == inputs.sequences.end(), "Missing external sequence: ", source.name);
        count = it->second.size();
        if (loops_[index]->Options().generation &&
            loops_[index]->Options().generation->phase == TokenGenerationPhase::Prefill &&
            binding.input_name == loops_[index]->Options().generation->input_ids) {
          ORT_RETURN_IF(count != 1, "Prefill requires one complete prompt batch");
        }
        for (const auto& item : it->second) {
          ORT_RETURN_IF_ERROR(CheckHostInput(target, item));
          if (loops_[index]->Options().generation &&
              binding.input_name == loops_[index]->Options().generation->input_ids) {
            ORT_RETURN_IF_ERROR(TokenGeneration::ValidatePrompt(item));
          }
        }
      } else if (source.kind == ChainSourceKind::ExternalInput) {
        auto it = inputs.inputs.find(source.name);
        ORT_RETURN_IF(it == inputs.inputs.end(), "Missing external input: ", source.name);
        value = &it->second;
      } else if (source.kind == ChainSourceKind::State && !produced_state.contains(source.name)) {
        auto it = inputs.initial_state.find(source.name);
        ORT_RETURN_IF(it == inputs.initial_state.end(), "Missing initial state: ", source.name);
        value = &it->second;
      }
      if (value) {
        ORT_RETURN_IF_ERROR(CheckHostInput(target, *value));
        if (loops_[index]->Options().generation &&
            binding.input_name == loops_[index]->Options().generation->input_ids) {
          ORT_RETURN_IF_ERROR(TokenGeneration::ValidatePrompt(*value));
        }
      }
      if (IsFinite(*loops_[index], binding.input_name)) {
        ORT_RETURN_IF(batch_count.has_value() && *batch_count != count,
                      "Finite input sequence lengths differ: ", options_.components[index].name);
        batch_count = count;
      }
    }
    for (const auto& binding : options_.components[index].state_outputs) {
      produced_state.insert(binding.state_name);
    }
  }
  return Status::OK();
}

Status SessionChain::CreateExecution(SessionChainInputs inputs, std::unique_ptr<SessionChainExecution>& out) const {
  ORT_RETURN_IF_ERROR(ValidateInputs(inputs));
  std::shared_ptr<KVCacheManager> cache;
  if (!cache_specs_.empty()) {
    std::unique_ptr<IOBinding> binding;
    ORT_RETURN_IF_ERROR(sessions_[0]->NewIOBinding(&binding));
    std::unique_ptr<KVCacheManager> manager;
    ORT_RETURN_IF_ERROR(KVCacheManager::Create(cache_specs_, binding->GetCPUAllocator(kCpuExecutionProvider), manager));
    cache = std::move(manager);
  }
  out = std::unique_ptr<SessionChainExecution>(
      new SessionChainExecution(shared_from_this(), std::move(inputs), std::move(cache),
                                has_generation_ ? std::make_shared<TokenGenerationState>() : nullptr));
  return Status::OK();
}

SessionChainExecution::SessionChainExecution(std::shared_ptr<const SessionChain> chain, SessionChainInputs inputs,
                                             std::shared_ptr<KVCacheManager> cache,
                                             std::shared_ptr<TokenGenerationState> generation)
    : chain_(std::move(chain)),
      initial_inputs_(std::move(inputs)),
      cache_(std::move(cache)),
      generation_(std::move(generation)) {
  Reset();
}

Status SessionChainExecution::Resolve(const ChainValueSource& source, const OrtValue*& out) const {
  const LoopFeeds* values = &initial_inputs_.inputs;
  if (source.kind == ChainSourceKind::ComponentOutput) {
    values = &component_outputs_[source.component_index];
  } else if (source.kind == ChainSourceKind::State) {
    values = &state_;
  }
  auto it = values->find(source.name);
  ORT_RETURN_IF(it == values->end(), "Chain value was not produced: ", source.name,
                source.kind == ChainSourceKind::ComponentOutput
                    ? MakeString(" from component '", chain_->options_.components[source.component_index].name, "'")
                    : "");
  out = &it->second;
  return Status::OK();
}

Status SessionChainExecution::PrepareComponentInputs(SessionLoopInputs& inputs) const {
  const auto& component = chain_->options_.components[component_index_];
  const auto& loop = *chain_->loops_[component_index_];
  size_t batch_count = loop.Options().finite_input_names.empty() ? 0 : 1;
  for (const auto& binding : component.inputs) {
    if (binding.source.kind == ChainSourceKind::ExternalSequence) {
      batch_count = initial_inputs_.sequences.at(binding.source.name).size();
      break;
    }
  }
  inputs.batches.reserve(batch_count);
  for (size_t i = 0; i < batch_count; ++i) {
    inputs.batches.emplace_back();
  }
  for (const auto& binding : component.inputs) {
    if (binding.source.kind == ChainSourceKind::ExternalSequence) {
      const auto& sequence = initial_inputs_.sequences.at(binding.source.name);
      for (size_t i = 0; i < sequence.size(); ++i) {
        inputs.batches[i].emplace(binding.input_name, sequence[i]);
      }
    } else {
      const OrtValue* value = nullptr;
      ORT_RETURN_IF_ERROR(Resolve(binding.source, value));
      if (IsFinite(loop, binding.input_name)) {
        for (auto& batch : inputs.batches) {
          batch.emplace(binding.input_name, *value);
        }
      } else {
        inputs.inputs.emplace(binding.input_name, *value);
      }
    }
  }
  return Status::OK();
}

Status SessionChainExecution::StepComponent(SessionChainStepResult& result) {
  const auto& component = chain_->options_.components[component_index_];
  const auto& loop = chain_->loops_[component_index_];
  if (!active_) {
    SessionLoopInputs inputs;
    ORT_RETURN_IF_ERROR(PrepareComponentInputs(inputs));
    ORT_RETURN_IF_ERROR(loop->CreateExecutionWithCache(std::move(inputs), cache_, generation_, active_));
  }
  ORT_RETURN_IF_ERROR(active_->Step(result.component_result));
  ORT_RETURN_IF(cancelled_.load(), "SessionChain cancelled during a component; call Reset before reuse");
  result.component_name = component.name;
  result.executed = result.component_result.executed;
  if (result.executed) {
    auto& outputs = component_outputs_[component_index_];
    for (size_t i = 0; i < loop->Options().output_names.size(); ++i) {
      outputs.insert_or_assign(loop->Options().output_names[i], result.component_result.outputs[i]);
    }
    result.component_outputs = outputs;
    for (const auto& binding : component.state_outputs) {
      if (cache_ && std::any_of(chain_->cache_specs_.begin(), chain_->cache_specs_.end(),
                                [&](const auto& spec) { return spec.name == binding.state_name; })) {
        ORT_RETURN_IF_ERROR(
            cache_->Update(binding.state_name, outputs.at(binding.output_name), KVCacheUpdateKind::Share));
      }
      state_.insert_or_assign(binding.state_name, outputs.at(binding.output_name));
    }
    for (const auto& item : loop->Options().kv_caches) {
      KVCacheView view;
      ORT_RETURN_IF_ERROR(cache_->Get(item.cache.name, view));
      state_.insert_or_assign(item.cache.name, view.value);
    }
    invocations_ = SafeInt<int64_t>(invocations_) + 1;
  }
  if (result.component_result.completed) {
    active_.reset();
    ++component_index_;
  }
  return Status::OK();
}

Status SessionChainExecution::Step(SessionChainStepResult& out) {
  ORT_RETURN_IF(requires_reset_, "SessionChain execution requires Reset after a failure or cancellation");
  if (cancelled_.load()) {
    requires_reset_ = true;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SessionChain cancelled; call Reset before reuse");
  }
  SessionChainStepResult result;
  requires_reset_ = true;
  while (component_index_ < chain_->loops_.size()) {
    const auto index = component_index_;
    const auto status = StepComponent(result);
    if (!status.IsOK()) {
      return Status(status.Category(), status.Code(),
                    MakeString("SessionChain component '", chain_->options_.components[index].name,
                               "': ", status.ErrorMessage()));
    }
    if (result.executed) {
      break;
    }
  }
  result.completed = component_index_ == chain_->loops_.size();
  result.components_completed = component_index_;
  result.invocations = invocations_;
  result.generated_tokens = generation_ ? static_cast<int64_t>(SafeInt<int64_t>(generation_->token_ids.size())) : 0;
  if (result.completed && (result.executed || !result.component_name.empty())) {
    for (const auto& binding : chain_->options_.outputs) {
      if (binding.kind == ChainOutputKind::GeneratedTokenIds) {
        std::unique_ptr<IOBinding> io;
        ORT_RETURN_IF_ERROR(chain_->sessions_[0]->NewIOBinding(&io));
        result.outputs.emplace(binding.name, TokenGeneration::OutputTokens(
                                                 *generation_, io->GetCPUAllocator(kCpuExecutionProvider)));
        continue;
      }
      const OrtValue* value = nullptr;
      ORT_RETURN_IF_ERROR(
          Resolve({ChainSourceKind::ComponentOutput, binding.output_name, binding.component_index}, value));
      result.outputs.emplace(binding.name, *value);
    }
  }
  requires_reset_ = false;
  out = std::move(result);
  return Status::OK();
}

Status SessionChainExecution::Run(SessionChainStepResult& out) {
  SessionChainStepResult result;
  do {
    ORT_RETURN_IF_ERROR(Step(result));
  } while (!result.completed);
  out = std::move(result);
  return Status::OK();
}

void SessionChainExecution::Reset() {
  active_.reset();
  if (generation_) {
    generation_->Reset();
  }
  component_index_ = 0;
  invocations_ = 0;
  component_outputs_.clear();
  component_outputs_.reserve(chain_->loops_.size());
  for (size_t i = 0; i < chain_->loops_.size(); ++i) {
    component_outputs_.emplace_back();
  }
  state_ = initial_inputs_.initial_state;
  if (cache_) {
    cache_->Reset();
    for (const auto& spec : chain_->cache_specs_) {
      KVCacheView view;
      ORT_THROW_IF_ERROR(cache_->Get(spec.name, view));
      state_.emplace(spec.name, std::move(view.value));
    }
  }
  cancelled_.store(false);
  requires_reset_ = false;
}

}  // namespace onnxruntime
