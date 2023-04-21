// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_training_c_api.h"
#include <optional>
#include <variant>

namespace Ort::detail {

#define ORT_DECLARE_TRAINING_RELEASE(NAME) \
  void OrtRelease(Ort##NAME* ptr);

// These release methods must be forward declared before including onnxruntime_cxx_api.h
// otherwise class Base won't be aware of them
ORT_DECLARE_TRAINING_RELEASE(CheckpointState);
ORT_DECLARE_TRAINING_RELEASE(TrainingSession);

}  // namespace Ort::detail

#include "onnxruntime_cxx_api.h"

namespace Ort {

inline const OrtTrainingApi& GetTrainingApi() { return *GetApi().GetTrainingApi(ORT_API_VERSION); }

namespace detail {

#define ORT_DEFINE_TRAINING_RELEASE(NAME) \
  inline void OrtRelease(Ort##NAME* ptr) { GetTrainingApi().Release##NAME(ptr); }

ORT_DEFINE_TRAINING_RELEASE(CheckpointState);
ORT_DEFINE_TRAINING_RELEASE(TrainingSession);

#undef ORT_DECLARE_TRAINING_RELEASE
#undef ORT_DEFINE_TRAINING_RELEASE

}  // namespace detail

using Property = std::variant<int64_t, float, std::string>;

/** \brief Class that holds the state of the training session state
 *
 * Wraps OrtCheckpointState
 *
 */
class CheckpointState : public detail::Base<OrtCheckpointState> {
 private:
  CheckpointState(OrtCheckpointState* checkpoint_state) { p_ = checkpoint_state; }

 public:
  CheckpointState() = delete;

  /** \brief Loads the checkpoint at provided path and returns the checkpoint state
   *
   * Wraps OrtTrainingApi::LoadCheckpoint
   *
   * \param[in] path_to_checkpoint Path to the checkpoint file to load
   * \return CheckpointState object which holds the state of the training session parameters.
   */
  static CheckpointState LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint);

  /** \brief Saves the state of the training session to a checkpoint file provided by the given path.
   *
   * Wraps OrtTrainingApi::SaveCheckpoint
   *
   * \param[in] checkpoint_state Training session checkpoint state to save to the checkpoint file
   * \param[in] path_to_checkpoint Path to the checkpoint file to load
   */
  static void SaveCheckpoint(const CheckpointState& checkpoint_state,
                             const std::basic_string<ORTCHAR_T>& path_to_checkpoint,
                             const bool include_optimizer_state = false);

  /** \brief Adds the given property to the state.
   *
   * Wraps OrtTrainingApi::AddProperty
   *
   * \param[in] property_name Name of the property to add to the state.
   * \param[in] property_value Value of the property to add to the state.
   */
  void AddProperty(const std::string& property_name, const Property& property_value);

  /** \brief Gets the property associated with the given name from the state.
   *
   * Wraps OrtTrainingApi::GetProperty
   *
   * \param[in] property_name Name of the property to get from the state.
   * \return Property value associated with the property name.
   */
  Property GetProperty(const std::string& property_name);
};

/** \brief Trainer class that provides training, evaluation and optimizer methods for
 *         executing ONNX models.
 *
 * Wraps OrtTrainingSession
 *
 */
class TrainingSession : public detail::Base<OrtTrainingSession> {
 private:
  size_t training_model_output_count_, eval_model_output_count_;

 public:
  TrainingSession(const Env& env, const SessionOptions& session_options, CheckpointState& checkpoint_state,
                  const std::basic_string<ORTCHAR_T>& train_model_path,
                  const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path = std::nullopt,
                  const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path = std::nullopt);

  /** \brief Run the train step returning results in an Ort allocated vector.
   *
   * Wraps OrtTrainingApi::TrainStep
   *
   * \param[in] input_values Array of Value objects in the order expected by the training model.
   * \return A std::vector of Value objects that represents the output of the forward pass.
   */
  std::vector<Value> TrainStep(const std::vector<Value>& input_values);

  /** \brief Lazily resets the gradients of the trainable parameters.
   *
   * Wraps OrtTrainingApi::LazyResetGrad
   *
   */
  void LazyResetGrad();

  /** \brief Run the evaluation step returning results in an Ort allocated vector.
   *
   * Wraps OrtTrainingApi::EvalStep
   *
   * \param[in] input_values Array of Value objects in the order expected by the eval model.
   * \return A std::vector of Value objects that represents the output of the eval pass.
   */
  std::vector<Value> EvalStep(const std::vector<Value>& input_values);

  /** \brief Set the learning rate to be used by the optimizer for parameter updates.
   *
   * Wraps OrtTrainingApi::SetLearningRate
   *
   * \param[in] learning_rate float value representing the constant learning rate to be used.
   */
  void SetLearningRate(float learning_rate);

  /** \brief Get the current learning rate that is being used by the optimizer.
   *
   * Wraps OrtTrainingApi::GetLearningRate
   *
   * \return float representing the current learning rate.
   */
  float GetLearningRate() const;

  /** \brief Register the linear learning rate scheduler for the training session.
   *
   * Wraps OrtTrainingApi::RegisterLinearLRScheduler
   *
   * \param[in] warmup_step_count Number of steps in the warmup phase.
   * \param[in] total_step_count Total number of training steps.
   * \param[in] initial_lr Initial learning rate to use.
   */
  void RegisterLinearLRScheduler(int64_t warmup_step_count, int64_t total_step_count,
                                 float initial_lr);

  /** \brief Updates the learning rate based on the lr scheduler.
   *
   * Wraps OrtTrainingApi::SchedulerStep
   *
   */
  void SchedulerStep();

  /** \brief Runs the optimizer model and updates the model parameters.
   *
   * Wraps OrtTrainingApi::OptimizerStep
   *
   */
  void OptimizerStep();

  /** \brief Exports a model that can be used for inferencing with the inference session.
   *
   * Wraps OrtTrainingApi::ExportModelForInferencing
   *
   * \param[in] inference_model_path Path to a location where the inference ready onnx model should be saved to.
   * \param[in] graph_output_names Vector of output names that the inference model should have.
   */
  void ExportModelForInferencing(const std::basic_string<ORTCHAR_T>& inference_model_path,
                                 const std::vector<std::string>& graph_output_names);

  /** \brief Gets the graph input names.
   *
   * Wraps OrtTrainingApi::TrainingSessionGetTrainingModelInputName,
   * OrtTrainingApi::TrainingSessionGetEvalModelInputName,
   * OrtTrainingApi::TrainingSessionGetTrainingModelInputCount
   * OrtTrainingApi::TrainingSessionGetEvalModelInputCount
   *
   * \param[in] training Whether the training model input names are requested or eval model input names.
   * \return Graph input names for either the training model or the eval model.
   *
   */
  std::vector<std::string> InputNames(const bool training);

  /** \brief Gets the graph output names.
   *
   * Wraps OrtTrainingApi::TrainingSessionGetTrainingModelOutputName,
   * OrtTrainingApi::TrainingSessionGetEvalModelOutputName,
   * OrtTrainingApi::TrainingSessionGetTrainingModelOutputCount
   * OrtTrainingApi::TrainingSessionGetEvalModelOutputCount
   *
   * \param[in] training Whether the training model output names are requested or eval model output names.
   * \return Graph output names for either the training model or the eval model.
   */
  std::vector<std::string> OutputNames(const bool training);

  /** \brief Copies the training session model parameters to a contiguous buffer
   *
   * Wraps OrtTrainingApi::CopyParametersToBuffer
   *
   * \param[in] only_trainable Whether to only copy trainable parameters or to copy all parameters.
   * \return Contiguous buffer to the model parameters.
   */
  Value ToBuffer(const bool only_trainable);

  /** \brief Loads the training session model parameters from a contiguous buffer
   *
   * Wraps OrtTrainingApi::CopyBufferToParameters
   *
   * \param[in] buffer Contiguous buffer to load the parameters from.
   */
  void FromBuffer(Value& buffer);
};

/** \brief Sets the given seed for random number generation.
 *
 * Wraps OrtTrainingApi::SetSeed
 *
 * \param[in] seed Manual seed to use for random number generation.
 */
void SetSeed(const int64_t seed);

}  // namespace Ort

#include "onnxruntime_training_cxx_inline.h"
