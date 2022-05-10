// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/module.h"

namespace onnxruntime {
namespace training {
namespace api {

/**
 * @brief States belong to one specific trainable Parameter.
 *   Momentum states for each Parameter.
 *   For Adam optimizer, it looks like:
 *     {
 *       "moment_0": shared_ptr<OrtValue>,
 *       "moment_1": shared_ptr<OrtValue>,
 *     }.
 */
struct ParameterOptimizerState {
  std::unordered_map<std::string, std::shared_ptr<OrtValue>> momentum_named_states;
};

/**
 * @brief States belong to one specific group of trainable Parameters.
 */
struct GroupOptimizerState {
  int64_t step;
  float learning_rate;
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

struct Optimizer {
 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input's ValueInfoProto if the parameter doesn't have it already.
  Optimizer(const std::string& optim_path_or_bytes,
            const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters);

  // Reset and release the gradient buffer of all trainable params
  Status ResetGrad() {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  // Optimizer Step.
  Status Step() {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

  Status GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_states);

 protected:
  int64_t GetStep() const {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return 0;
  }
  Status SetLearningRate(float /*lr*/) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 private:
  std::unique_ptr<onnxruntime::InferenceSession> optim_sess_;
  std::vector<std::shared_ptr<Parameter>> parameters_;
  GroupOptimizerState optimizer_state_;
};

class LearningRateScheduler {
 public:
  LearningRateScheduler(const Optimizer& optim)
      : optim_(optim) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  virtual ~LearningRateScheduler() = default;

  // Modify the current learning rate based on current step
  virtual Status Step(/*int64_t step*/) = 0;

  const Optimizer& optim_;
};

class LinearScheduler : public LearningRateScheduler {
 public:
  explicit LinearScheduler(const Optimizer& optim, float start_factor, float end_factor, int64_t total_iters)
      : LearningRateScheduler(optim),
        start_factor_(start_factor),
        end_factor_(end_factor),
        total_iters_(total_iters) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }

  // Fetch the step, calculate next value and set lr in optimizer
  Status Step(/*int64_t step*/) override {
    ORT_NOT_IMPLEMENTED("Not implemented.");
    return Status::OK();
  }

 private:
  float start_factor_;
  float end_factor_;
  int64_t total_iters_;
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
