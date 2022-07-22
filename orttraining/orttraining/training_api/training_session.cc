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

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
