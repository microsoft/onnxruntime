// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/training_session.h"

namespace onnxruntime {
namespace training {
namespace api {

TrainingSession::TrainingSession(const Environment& session_env,
                                 const SessionOptions& session_options,
                                 const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                                 const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                                 const ModelIdentifiers& model_identifiers)
    : named_parameters_{parameters},
      module_{std::make_unique<Module>(model_identifiers.train_model, named_parameters_,
                                       session_options, session_env, providers, model_identifiers.eval_model)},
      optimizer_{model_identifiers.optim_model.has_value()
                     ? std::make_unique<Optimizer>(
                           model_identifiers.optim_model.value(), named_parameters_,
                           session_options, session_env, providers)
                     : std::unique_ptr<Optimizer>()} {}

Status TrainingSession::RegisterScheduler(
    const std::function<std::unique_ptr<LRSchedulerBase>(std::shared_ptr<Optimizer>)>& get_scheduler,
    std::optional<float> initial_lr) {
  scheduler_ = std::move(get_scheduler(optimizer_));
  ORT_RETURN_IF_NOT(scheduler_, "The provided instance of the learning rate scheduler is a nullptr.");

  if (initial_lr.has_value()) {
    ORT_RETURN_IF_ERROR(optimizer_->SetInitialLearningRate(initial_lr.value()));
  }

  return Status::OK();
}

size_t TrainingSession::GetTrainModeOutputCount() const noexcept {
  return module_->GetTrainModeOutputCount();
}

size_t TrainingSession::GetEvalModeOutputCount() const noexcept {
  return module_->GetEvalModeOutputCount();
}

Status TrainingSession::TrainStep(const RunOptions&,
                                  const std::vector<OrtValue>& inputs,
                                  std::vector<OrtValue>& fetches) {
  return module_->TrainStep(inputs, fetches);
}

Status TrainingSession::EvalStep(const RunOptions&,
                                 const std::vector<OrtValue>& inputs,
                                 std::vector<OrtValue>& fetches) const {
  return module_->EvalStep(inputs, fetches);
}

Status TrainingSession::ResetGrad() {
  return module_->ResetGrad();
}

Status TrainingSession::OptimizerStep(const RunOptions&) {
  return optimizer_->Step();
}

Status TrainingSession::CreateCheckpointState(CheckpointState& chkpt_state, bool save_optimizer_state) const {
  ORT_RETURN_IF_ERROR(module_->GetStateDict(chkpt_state.module_checkpoint_state));
  if (save_optimizer_state) {
    ORT_RETURN_IF_ERROR(optimizer_->GetStateDict(chkpt_state.optimizer_checkpoint_state));
  }

  return Status::OK();
}

Status TrainingSession::SetLearningRate(float learning_rate) noexcept {
  ORT_RETURN_IF_ERROR(optimizer_->SetLearningRate(learning_rate));

  return Status::OK();
}

Status TrainingSession::SchedulerStep() noexcept {
  ORT_RETURN_IF_NOT(scheduler_, "No learning rate schedler was registered. Please register a valid learning rate scheduler");
  return scheduler_->Step();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
