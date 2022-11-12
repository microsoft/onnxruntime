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
  ORT_RETURN_IF_NOT(optimizer_, "No optimizer session initialized.");
  scheduler_ = get_scheduler(optimizer_);
  ORT_RETURN_IF_NOT(scheduler_, "The provided instance of the learning rate scheduler is a nullptr.");

  if (initial_lr.has_value()) {
    ORT_RETURN_IF_ERROR(optimizer_->SetInitialLearningRate(initial_lr.value()));
  }

  return Status::OK();
}

size_t TrainingSession::GetTrainingModelOutputCount() const noexcept {
  return module_->GetTrainingModelOutputCount();
}

size_t TrainingSession::GetEvalModelOutputCount() const noexcept {
  return module_->GetEvalModelOutputCount();
}

std::string TrainingSession::GetTrainingModelOutputName(size_t index) const noexcept {
  return module_->GetTrainingModelOutputName(index);
}

std::string TrainingSession::GetEvalModelOutputName(size_t index) const noexcept {
  return module_->GetEvalModelOutputName(index);
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
  ORT_RETURN_IF_NOT(optimizer_, "No optimizer session initialized.");
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
  ORT_RETURN_IF_NOT(optimizer_, "No optimizer session initialized.");
  ORT_RETURN_IF_ERROR(optimizer_->SetLearningRate(learning_rate));

  return Status::OK();
}

float TrainingSession::GetLearningRate() const {
  ORT_ENFORCE(optimizer_, "No optimizer session initialized.");
  return optimizer_->GetLearningRate();
}

Status TrainingSession::SchedulerStep() noexcept {
  ORT_RETURN_IF_NOT(scheduler_, "No learning rate scheduler was registered. Please register a valid learning rate scheduler");
  return scheduler_->Step();
}

size_t TrainingSession::GetParametersSize(const bool trainable_only) const {
  return module_->GetParametersSize(trainable_only);
}

Status TrainingSession::CopyParametersToBuffer(OrtValue& parameters_buffer, const bool trainable_only) {
  return module_->CopyParametersToBuffer(parameters_buffer, trainable_only);
}

Status TrainingSession::CopyBufferToParameters(OrtValue& parameters_buffer, const bool trainable_only) {
  return module_->CopyBufferToParameters(parameters_buffer, trainable_only);
}

Status TrainingSession::ExportModelForInferencing(const std::string& inference_model_path,
                                                  gsl::span<const std::string> graph_output_names) const {
  return module_->ExportModelForInferencing(inference_model_path, graph_output_names);
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
