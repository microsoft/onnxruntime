// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/session_chain.h"

#include <optional>

#include "core/common/safeint.h"
#include "core/session/inference_session.h"
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
    const auto [status, outputs] = session->GetModelOutputs();
    ORT_RETURN_IF_ERROR(status);
    options.output_names.clear();
    for (const auto* output : *outputs) {
      options.output_names.push_back(output->Name());
    }
    std::shared_ptr<const SessionLoop> loop;
    ORT_RETURN_IF_ERROR(SessionLoop::Create(*session, std::move(options), loop));
    loops_.push_back(std::move(loop));
    sessions_.push_back(session);
  }

  InlinedHashMap<std::string, const NodeArg*> state_types;
  for (size_t index = 0; index < options_.components.size(); ++index) {
    const auto& component = options_.components[index];
    const auto& arguments = *sessions_[index]->GetModelInputs().second;
    ORT_RETURN_IF(component.inputs.size() != arguments.size(),
                  "Component input binding count mismatch: ", component.name);
    InlinedHashSet<std::string> destinations;
    for (const auto& binding : component.inputs) {
      const auto* target = FindArgument(arguments, binding.input_name);
      ORT_RETURN_IF_NOT(target && destinations.insert(binding.input_name).second,
                        "Unknown or duplicate component input: ", component.name, ".", binding.input_name);
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
      auto it = state_types.find(binding.state_name);
      if (it != state_types.end()) {
        ORT_RETURN_IF_ERROR(CheckCompatibleTypes(*output, *it->second));
      }
      state_types.insert_or_assign(binding.state_name, output);
    }
  }
  InlinedHashSet<std::string> outputs;
  for (const auto& binding : options_.outputs) {
    ORT_RETURN_IF(binding.name.empty() || !outputs.insert(binding.name).second ||
                      binding.component_index >= sessions_.size(),
                  "Invalid or duplicate chain output port: ", binding.name);
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
        for (const auto& item : it->second) {
          ORT_RETURN_IF_ERROR(CheckHostInput(target, item));
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
  out = std::unique_ptr<SessionChainExecution>(new SessionChainExecution(shared_from_this(), std::move(inputs)));
  return Status::OK();
}

SessionChainExecution::SessionChainExecution(std::shared_ptr<const SessionChain> chain, SessionChainInputs inputs)
    : chain_(std::move(chain)), initial_inputs_(std::move(inputs)) {
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
    ORT_RETURN_IF_ERROR(loop->CreateExecution(std::move(inputs), active_));
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
      state_.insert_or_assign(binding.state_name, outputs.at(binding.output_name));
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
  if (result.completed && (result.executed || !result.component_name.empty())) {
    for (const auto& binding : chain_->options_.outputs) {
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
  component_index_ = 0;
  invocations_ = 0;
  component_outputs_.clear();
  component_outputs_.reserve(chain_->loops_.size());
  for (size_t i = 0; i < chain_->loops_.size(); ++i) {
    component_outputs_.emplace_back();
  }
  state_ = initial_inputs_.initial_state;
  cancelled_.store(false);
  requires_reset_ = false;
}

}  // namespace onnxruntime
