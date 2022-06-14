// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "module.h"
#include "optimizer.h"
#include "checkpoint.h"

namespace onnxruntime {
namespace training {
namespace api {
using namespace common;

// Wrapper on top of module and optimizer classes and is the only class exposed via capis
class TrainingSession {
 public:
  TrainingSession(const Environment& session_env,
                  const SessionOptions& session_options,
                  const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                  const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters);

  Status Initialize(const std::string& train_model_uri,
                    const std::optional<std::string>& eval_model_uri,
                    const std::optional<std::string>& optim_model_uri);

  size_t GetTrainModeOutputCount() const noexcept;

  size_t GetEvalModeOutputCount() const noexcept;

  Status TrainStep(const RunOptions& run_options,
                   const std::vector<OrtValue>& inputs,
                   std::vector<OrtValue>& fetches);

  Status EvalStep(const RunOptions& run_options,
                  const std::vector<OrtValue>& inputs,
                  std::vector<OrtValue>& fetches);

  Status ResetGrad();

  Status OptimizerStep(const RunOptions& run_options);

  Status CreateCheckpointState(CheckpointState& chkpt_state, bool save_optimizer_state);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TrainingSession);

  const Environment& environment_;
  SessionOptions session_options_;
  std::vector<std::shared_ptr<IExecutionProvider>> providers_;
  const std::unordered_map<std::string, std::shared_ptr<Parameter>> named_parameters_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<Optimizer> optimizer_;
};
}  // namespace api
}  // namespace training
}  // namespace onnxruntime
