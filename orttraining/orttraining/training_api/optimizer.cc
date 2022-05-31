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

// TODO: don't hard code the state names, should get the state names according to the optimizer types.
// TODO: Conolidate with frontend tooling
const std::vector<std::string> MOMENT_SUFFIXES{".exp_avg", ".exp_avg_sq"};
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

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
    }
  }
  // Add other optimizer reordering logic here
  return Status::OK();
}

Optimizer::Optimizer(const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
                     InferenceSession* optim_session) : named_parameters_(named_parameters) {
  optim_sess_ = optim_session;

  utils::GetGraphInputOutputNames(optim_sess_, input_names_, output_names_);
  ORT_ENFORCE(input_names_[0] == "learning_rate");  // TODO: make this better
  ORT_ENFORCE(input_names_[1] == "step");           // TODO: make this better

  if (optimizer_type_ == OptimizerType::AdamW) {
    ORT_THROW_IF_ERROR(GenerateMomentumNamedStates());
  }
  ORT_THROW_IF_ERROR(ConstructInputs());
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  utils::WarpInOrtValue<float>(optimizer_state_.learning_rate, &learning_rate_input);
  utils::WarpInOrtValue<int64_t>(optimizer_state_.step, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // extract step output and update
  // TODO: need to remove hardcoding
  optimizer_state_.step = utils::GetValue<int64_t>(outputs[0]);

  return Status::OK();
}

Status Optimizer::GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;

  // Currently all parameters are in a single group, so we hardcode group0 here.
  // To support multiple groups, Optimizer constructor need accept informations for groupping.
  const std::string group_zero_name = "group0";
  grouped_optimizer_states.insert({group_zero_name, std::make_shared<GroupOptimizerState>(optimizer_state_)});

  // Pass the optimizer session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
