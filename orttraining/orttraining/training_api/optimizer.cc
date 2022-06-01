// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

// Currently all parameters are in a single group, so we hardcode group0 here.
const std::string GROUP_ZERO_NAME = "group0";

// TODO: don't hard code the state names, should get the state names according to the optimizer types.
// TODO: Conolidate with frontend tooling
const std::vector<std::string> MOMENT_SUFFIXES{".exp_avg", ".exp_avg_sq"};
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

}  // namespace

Status Optimizer::GenerateMomentumNamedStates() {
  auto& param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;
  auto& optim_sess_state = optim_sess_->GetSessionState();
  for (auto& pair : named_parameters_) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : MOMENT_STATE_NAMES) {
        OrtValue param_state;
        ORT_ENFORCE(utils::OrtValueLike(optim_sess_state, pair.second->Data(), param_state).IsOK(),
                    "Error generating moment state for ", pair.first);
        cur_param_optimizer_states.momentum_named_states.insert({state_name, std::move(param_state)});
      }
    }
  }
  return Status::OK();
}

// Constructs the ortvalue inputs to be fed to the graph
// at each step
Status Optimizer::ConstructInputs() {
  if (optimizer_type_ == OptimizerType::AdamW) {
    auto& param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;

    std::string param_name;
    std::vector<std::string> param_names, grad_names, moment1_names, moment2_names, user_inputs;
    for (size_t i = 2; i < input_names_.size(); i++) {
      std::string& name = input_names_[i];
      auto it = named_parameters_.find(name);
      if (it != named_parameters_.end()) {  // is param
        param_names.push_back(name);
        inputs_.push_back(it->second->Data());
      } else if (utils::GetParamNameFromGradient(name, param_name)) {
        grad_names.emplace_back(name);
        // assert param_name is valid.
        auto it = named_parameters_.find(param_name);
        ORT_ENFORCE(it != named_parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
        inputs_.push_back(it->second->Gradient());
      } else if (utils::GetParamNameFromSuffix(name, MOMENT_SUFFIXES[0], param_name)) {
        moment1_names.push_back(name);
        auto it = named_parameters_.find(param_name);
        ORT_ENFORCE(it != named_parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
        inputs_.push_back(param_named_optimizer_states.at(param_name).momentum_named_states.at(MOMENT_STATE_NAMES[0]));
      } else if (utils::GetParamNameFromSuffix(name, MOMENT_SUFFIXES[1], param_name)) {
        moment2_names.push_back(name);
        auto it = named_parameters_.find(param_name);
        ORT_ENFORCE(it != named_parameters_.end(), "Unknown param: ", param_name, " for field: ", name);
        inputs_.push_back(param_named_optimizer_states.at(param_name).momentum_named_states.at(MOMENT_STATE_NAMES[1]));
      } else {
        ORT_ENFORCE("This is an invalid graph. Optimizer graph contains unknown user input:", name);
      }
      ORT_ENFORCE(inputs_.back().IsAllocated() && inputs_.back().IsTensor(), "Uninitialized tensor data for ", name);
    }
  }
  // Add other optimizer reordering logic here
  return Status::OK();
}

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
                     const onnxruntime::SessionOptions& session_options,
                     const Environment& env) : named_parameters_(named_parameters) {
  optim_sess_ = std::move(std::make_unique<InferenceSession>(session_options, env));

  ORT_THROW_IF_ERROR(optim_sess_->Load(optim_path_or_bytes));
  ORT_THROW_IF_ERROR(optim_sess_->Initialize());

  utils::GetGraphInputOutputNames(optim_sess_, input_names_, output_names_);
  ORT_ENFORCE(input_names_[0] == "learning_rate");  // TODO: make this better
  ORT_ENFORCE(input_names_[1] == "step");           // TODO: make this better

  if (optimizer_type_ == OptimizerType::AdamW) {
    ORT_THROW_IF_ERROR(GenerateMomentumNamedStates());
  } else {
    ORT_THROW("Unsupported optimizer type");
  }
  ORT_THROW_IF_ERROR(ConstructInputs());
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  utils::WrapInOrtValue<float>(optimizer_state_.learning_rate, &learning_rate_input);
  utils::WrapInOrtValue<int64_t>(optimizer_state_.step, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // extract step output and update
  if (utils::GetValue<int64_t>(outputs[0]) == 1LL)
    optimizer_state_.step++;

  return Status::OK();
}

Status Optimizer::GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;

  // To support multiple groups, Optimizer constructor need accept informations for groupping.
  grouped_optimizer_states.insert({GROUP_ZERO_NAME, std::make_shared<GroupOptimizerState>(optimizer_state_)});

  // Pass the optimizer session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

Status Optimizer::LoadStateDict(OptimizerCheckpointState& optimizer_checkpoint_states) {
  auto& group_optimizer_state =
      optimizer_checkpoint_states.group_named_optimizer_states[GROUP_ZERO_NAME];
  optimizer_state_.initial_lr = group_optimizer_state->initial_lr;
  optimizer_state_.step = group_optimizer_state->step;

  // TODO(pengwa): restore the momentums state from checkpoint.
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
