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
constexpr char GROUP_ZERO_NAME[] = "group0";

// TODO: don't hard code the state names, should get the state names according to the optimizer types.
// TODO: Conolidate with frontend tooling
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

constexpr char LearningRateName[] = "learning_rate";
constexpr char StepName[] = "step";
constexpr char ParamsName[] = "params";
constexpr char FirstOrderMomentsName[] = "first_order_moments";
constexpr char SecondOrderMomentsName[] = "second_order_moments";

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

    std::vector<Tensor> params, first_order_moments, second_order_moments;
    // TODO: Change to tensor seq implementation once clip grad norm op
    // that accepts tensor seq as input for gradients is complete.
    std::vector<OrtValue> grads;

    // Input names 0-4 are reserved for lr, step, params, first order moments, second order moments
    // input names 5 onwards are all the gradient names.
    // Collect all the inputs based on the gradient names order.
    for (size_t i = 5; i < input_names_.size(); i++) {
      std::string param_name;
      if (utils::GetParamNameFromGradient(input_names_[i], param_name)) {
        const auto named_parameter_it = named_parameters_.find(param_name);
        ORT_ENFORCE(named_parameter_it != named_parameters_.end(),
                    "Unknown param: ", param_name, " for field: ", input_names_[i]);

        // Collect the gradients as ortvalues
        grads.push_back(named_parameter_it->second->Gradient());

        // Collect parameters and prepare for tensorseq creation
        auto* param_tensor = named_parameter_it->second->Data().GetMutable<Tensor>();
        params.emplace_back(
            Tensor(param_tensor->DataType(), param_tensor->Shape(),
                   param_tensor->MutableDataRaw(), param_tensor->Location()));

        // Collect first order moments and prepare for tensorseq creation
        auto* first_order_moment_tensor = param_named_optimizer_states.at(param_name)
                                              .momentum_named_states.at(MOMENT_STATE_NAMES[0])
                                              .GetMutable<Tensor>();
        first_order_moments.emplace_back(
            Tensor(first_order_moment_tensor->DataType(), first_order_moment_tensor->Shape(),
                   first_order_moment_tensor->MutableDataRaw(), first_order_moment_tensor->Location()));

        // Collect second order moments and prepare for tensorseq creation
        auto* second_order_moment_tensor = param_named_optimizer_states.at(param_name)
                                               .momentum_named_states.at(MOMENT_STATE_NAMES[1])
                                               .GetMutable<Tensor>();
        second_order_moments.emplace_back(
            Tensor(second_order_moment_tensor->DataType(), second_order_moment_tensor->Shape(),
                   second_order_moment_tensor->MutableDataRaw(), second_order_moment_tensor->Location()));
      } else {
        ORT_ENFORCE(
            false, "This is an invalid graph. Optimizer graph contains unknown user input:", input_names_[i]);
      }
    }

    const auto tensorseq_inserter = [](auto& tensors, auto* inputs) {
      ORT_ENFORCE(!tensors.empty(), "Tensors cannot be empty while building a tensor sequence.");

      auto tensor_seq = std::make_unique<TensorSeq>(tensors.front().DataType());
      tensor_seq->SetElements(std::move(tensors));
      inputs->emplace_back(
          OrtValue(tensor_seq.release(), DataTypeImpl::GetType<TensorSeq>(),
                   DataTypeImpl::GetType<TensorSeq>()->GetDeleteFunc()));
    };

    // Add the params and moments as tensorseq ortvalues to inputs
    tensorseq_inserter(params, &inputs_);
    tensorseq_inserter(first_order_moments, &inputs_);
    tensorseq_inserter(second_order_moments, &inputs_);

    // Add the gradients as ortvalues to inputs
    inputs_.insert(inputs_.end(),
                   std::make_move_iterator(grads.begin()),
                   std::make_move_iterator(grads.end()));
  }
  // Add other optimizer reordering logic here
  return Status::OK();
}

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
                     const onnxruntime::SessionOptions& session_options,
                     const Environment& env,
                     const std::vector<std::shared_ptr<IExecutionProvider>>& providers)
    : named_parameters_(named_parameters) {
  optim_sess_ = std::move(std::make_unique<InferenceSession>(session_options, env));
  for (const auto& execution_provider : providers) {
    ORT_THROW_IF_ERROR(optim_sess_->RegisterExecutionProvider(execution_provider));
  }

  ORT_THROW_IF_ERROR(optim_sess_->Load(optim_path_or_bytes));
  ORT_THROW_IF_ERROR(optim_sess_->Initialize());

  utils::GetGraphInputOutputNames(optim_sess_, input_names_, output_names_);
  ORT_ENFORCE(input_names_[0] == LearningRateName);  // TODO: make this better
  ORT_ENFORCE(input_names_[1] == StepName);          // TODO: make this better
  ORT_ENFORCE(input_names_[2] == ParamsName);        // TODO: make this better

  if (optimizer_type_ == OptimizerType::AdamW) {
    ORT_ENFORCE(input_names_[3] == FirstOrderMomentsName);   // TODO: make this better
    ORT_ENFORCE(input_names_[4] == SecondOrderMomentsName);  // TODO: make this better

    ORT_THROW_IF_ERROR(GenerateMomentumNamedStates());
  } else {
    ORT_THROW("Unsupported optimizer type");
  }
  ORT_THROW_IF_ERROR(ConstructInputs());
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  utils::WrapInOrtValue<float>(optimizer_state_.learning_rate, &learning_rate_input);
  // Use step count + 1 before running optimizer step.
  // This is necessary since bias correction uses the step
  // as a power. Using power of 0 is wrong.
  utils::WrapInOrtValue<int64_t>(optimizer_state_.step + 1, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // extract step output and update
  if (utils::GetValue<int64_t>(outputs[0]) == 1LL) {
    optimizer_state_.step++;
  }

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

Status Optimizer::LoadStateDict(const OptimizerCheckpointState& optimizer_checkpoint_states) {
  auto group_optimizer_state_it =
      optimizer_checkpoint_states.group_named_optimizer_states.find(GROUP_ZERO_NAME);
  ORT_ENFORCE(group_optimizer_state_it != optimizer_checkpoint_states.group_named_optimizer_states.cend(),
              "Group 0 not found in the optimizer checkpoint states.");
  optimizer_state_.initial_lr = group_optimizer_state_it->second->initial_lr;
  optimizer_state_.step = group_optimizer_state_it->second->step;

  // TODO(pengwa): restore the momentums state from checkpoint.
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
