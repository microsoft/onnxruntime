// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/include/training_session.h"

namespace onnxruntime {
namespace training {
namespace api {

TrainingSession::TrainingSession(const Environment& session_env,
                                 const SessionOptions& session_options,
                                 const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                                 const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters)
    : environment_(session_env),
      session_options_{session_options},
      providers_{providers},
      named_parameters_{parameters} {}

Status TrainingSession::Initialize(const std::string& train_model_uri, const std::optional<std::string>& eval_model_uri,
                                   const std::optional<std::string>& optim_model_uri) {
  module_ = std::make_unique<Module>(train_model_uri, named_parameters_, session_options_,
                                     environment_, providers_, eval_model_uri);

  if (optim_model_uri.has_value()) {
    optimizer_ = std::make_unique<Optimizer>(optim_model_uri.value(), named_parameters_,
                                             session_options_, environment_, providers_);
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
                                 std::vector<OrtValue>& fetches) {
  return module_->EvalStep(inputs, fetches);
}

Status TrainingSession::ResetGrad() {
  return module_->ResetGrad();
}

Status TrainingSession::OptimizerStep(const RunOptions&) {
  return optimizer_->Step();
}

Status TrainingSession::CreateCheckpointState(CheckpointState& chkpt_state, bool save_optimizer_state) {
  ORT_RETURN_IF_ERROR(module_->GetStateDict(chkpt_state.module_checkpoint_state));
  if (save_optimizer_state) {
    ORT_RETURN_IF_ERROR(optimizer_->GetStateDict(chkpt_state.optimizer_checkpoint_state));
  }

  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
