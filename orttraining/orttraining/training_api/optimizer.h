// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/module.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief States belonging to one specific trainable Parameter.
 *   Momentum states for each Parameter.
 *   For Adam optimizer, it looks like:
 *     { "moment_0": OrtValue, "moment_1": OrtValue,}.
 */
struct ParameterOptimizerState {
  std::unordered_map<std::string, OrtValue> momentum_named_states;
};

/**
 * @brief States belong to one specific group of trainable Parameters.
 */
struct GroupOptimizerState {
  int64_t step = 0;
  float initial_lr = 0.001f;        // Default value used in torch AdamW
  float learning_rate{initial_lr};  // Adaptive learning rate as training proceeds.
  std::unordered_map<std::string, ParameterOptimizerState> param_named_optimizer_states;
};

/**
 * @brief States belong to all groups of trainable Parameters.
 * Besides, also maintain a pointer of DataTransferManager* that is owned by InferenceSession.
 * This is used to do Tensor copy in the file saving stage.
 */
struct OptimizerCheckpointState {
 public:
  std::unordered_map<std::string, std::shared_ptr<GroupOptimizerState>> group_named_optimizer_states;
  const DataTransferManager* optimizer_session_data_transfer_mgr;
};

enum class OptimizerType {
  AdamW,
  // More optimizers can be added later as:
  // Lamb,
};

struct Optimizer {
  friend struct LRSchedulerBase;

 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input's ValueInfoProto if the parameter doesn't have it already.
  Optimizer(const std::string& optim_path_or_bytes,
            const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
            const onnxruntime::SessionOptions& session_options,
            const Environment& env,
            const std::vector<std::shared_ptr<IExecutionProvider>>& providers);

  Status Step();

  Status GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_states);

  Status LoadStateDict(const OptimizerCheckpointState& optimizer_checkpoint_states);

  Status SetLearningRate(float lr) {
    optimizer_state_.learning_rate = lr;
    return Status::OK();
  }

  float GetLearningRate() const noexcept {
    return optimizer_state_.learning_rate;
  }

  Status SetInitialLearningRate(float initial_lr) {
    optimizer_state_.initial_lr = initial_lr;
    optimizer_state_.learning_rate = initial_lr;
    return Status::OK();
  }

 private:
  int64_t GetStep() const {
    return optimizer_state_.step;
  }

  // Generates optimizer momentum states for applicable optimizer types
  Status GenerateMomentumNamedStates();
  // Constructs the ortvalue inputs to be fed to the graph
  // at each step
  Status ConstructInputs();

  // TODO: load this info from checkpoint
  OptimizerType optimizer_type_ = OptimizerType::AdamW;
  std::unique_ptr<onnxruntime::InferenceSession> optim_sess_;
  const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters_;
  GroupOptimizerState optimizer_state_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<OrtValue> inputs_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
