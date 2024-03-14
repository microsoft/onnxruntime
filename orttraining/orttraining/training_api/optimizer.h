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
typedef InlinedHashMap<std::string, OrtValue> ParameterOptimizerState;

/**
 * @brief States belong to one specific group of trainable Parameters.
 */
struct GroupOptimizerState {
  int64_t step = 0;
  float initial_lr = 0.001f;  // Default value used in torch AdamW

  // Adaptive learning rate as training proceeds. Be noted, learning_rate can be
  // restored by lr scheduler from given step and initial_lr, though, we still save/load this in checkpoint.
  float learning_rate{initial_lr};
  InlinedHashMap<std::string, ParameterOptimizerState> param_named_optimizer_states;
};

/**
 * @brief States belong to all groups of trainable Parameters.
 * Besides, also maintain a pointer of DataTransferManager* that is owned by InferenceSession.
 * This is used to do Tensor copy in the file saving stage.
 */
struct OptimizerCheckpointState {
 public:
  InlinedHashMap<std::string, std::shared_ptr<GroupOptimizerState>> group_named_optimizer_states;
  const DataTransferManager* optimizer_session_data_transfer_mgr;
};

struct OptimizerAlgorithmBase {
  OptimizerAlgorithmBase(const InlinedVector<std::string>& momentum_keys,
                         const InlinedVector<std::string>& optimizer_states_inputs)
      : momentum_keys(momentum_keys), optimizer_states_inputs(optimizer_states_inputs) {}
  InlinedVector<std::string> momentum_keys;
  InlinedVector<std::string> optimizer_states_inputs;
};

struct AdamWOptimizerAlgorithm : public OptimizerAlgorithmBase {
  AdamWOptimizerAlgorithm() : OptimizerAlgorithmBase({"momentum0", "momentum1"},
                                                     {"first_order_moments", "second_order_moments"}) {}
};

struct SGDOptimizerV2Algorithm : public OptimizerAlgorithmBase {
  SGDOptimizerV2Algorithm() : OptimizerAlgorithmBase({"momentum0"},
                                                     {"first_order_moments"}) {}
};

struct OptimizerAlorithmFactory {
  static std::unique_ptr<OptimizerAlgorithmBase> CreateInstance(const GraphViewer& graph_viewer,
                                                                int32_t& group_count);
};

struct CheckpointState;

/**
 * @brief Optimizer class for running gradient updates.
 *
 * This class is responsible for running gradient updates on the parameters.
 * > It does NOT own the parameters, and will not modify the "named_parameters" in `CheckpointState`
 *   passed from the constructor.
 *   A tensor sequence is created based on the "named_parameters" to construct parameter input (of type tensorseq).
 * > If 'optimizer_checkpoint_states' is provided in the constructor as part of `CheckpointState`.
 *   Optimizer will reuse the data buffer from the passed in 'optimizer_checkpoint_states'.
 *   >> If the device of momentums in 'optimizer_checkpoint_states' is not
 *     matching its parameter device, a copy will be done during the 'LoadStateDict',
 *     but reserving using the original OrtValue with the copied data buffer;
 *   >> Otherwise, it generates the optimizer state initialized as all zeros and owns them.
 * > If 'optimizer_checkpoint_states' is not provided in the constructor as part of `CheckpointState`.
 *   The optimizer states are initialized as all zeros on the same device of corresponding parameters.
 *
 * Currently, we only support load checkpoints from the constructor;
 * no public API to load state dict after Optimizer instance is created.
 */
struct Optimizer {
  friend struct LRSchedulerBase;

 public:
  // Initialize an optimizer module from an ORT inference session with loaded
  // training ONNX model For each parameter, initialize the OptimizerState based
  // on the graph input's ValueInfoProto if the parameter doesn't have it already.
  Optimizer(const ModelIdentifiers& model_identifiers,
            CheckpointState* state,
            const onnxruntime::SessionOptions& session_options,
            const Environment& env,
            const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
            gsl::span<OrtCustomOpDomain* const> op_domains = gsl::span<OrtCustomOpDomain* const>());

  Status Step();

  Status SetLearningRate(float lr) {
    optimizer_state_->learning_rate = lr;
    return Status::OK();
  }

  float GetLearningRate() const noexcept {
    return optimizer_state_->learning_rate;
  }

  Status SetInitialLearningRate(float initial_lr) {
    optimizer_state_->initial_lr = initial_lr;
    optimizer_state_->learning_rate = initial_lr;
    return Status::OK();
  }

  // Constructs the optimizer state and prepares the model inputs.
  // This is called once during the construction of the Optimizer if the model state is available.
  // In case the optimizer was instantiated with a nominal checkpoint, this function must be
  // called when the model state is available.
  // The optimizer checks if the optimizer state needs to be constructed in the train step function.
  // However, this is exposed as a public function in case the user wants to construct the optimizer
  // state before the train step function is called.
  Status ConstructOptimizerStateAndInputs();

 private:
  void Initialize(const ModelIdentifiers& model_identifiers,
                  const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                  gsl::span<OrtCustomOpDomain* const> op_domains);

  int64_t GetStep() const {
    return optimizer_state_->step;
  }

  // Generates optimizer momentum states for parameters that require grad.
  Status GenerateMomentumNamedStates(OptimizerCheckpointState& optimizer_checkpoint_states);
  // Constructs the ortvalue inputs to be fed to the graph at each step.
  Status ConstructInputs();

  /**
   * @brief Load states from optimizer_checkpoint_states into current optimizer state.
   *
   * Be noted Optimizer will reuse the data buffer of passed in optimizer_checkpoint_states.
   * If the device of momentums in optimizer_checkpoint_states is not matching its parameter device,
   * an implicit copy will be done during the LoadStateDict, but reserving using the original OrtValue
   * with the copied data buffer.
   * @return Status
   */
  Status LoadStateDict(OptimizerCheckpointState& optimizer_checkpoint_states);

  std::unique_ptr<OptimizerAlgorithmBase> optimizer_algo_ptr_;
  std::unique_ptr<onnxruntime::InferenceSession> optim_sess_;

  CheckpointState* state_;  // Non owning pointer to the state
  std::shared_ptr<GroupOptimizerState> optimizer_state_;

  InlinedVector<std::string> input_names_;
  InlinedVector<std::string> output_names_;
  InlinedVector<OrtValue> inputs_;

  int32_t group_count_{0};

  bool delay_optimizer_state_contruction_{false};
};

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
