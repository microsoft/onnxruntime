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

/// <summary>
/// This function returns the C training api struct with the pointers to the ort training C functions.
/// If using C++, please use the class instances instead of invoking the C functions directly.
/// </summary>
/// <returns>OrtTrainingApi struct with ort training C function pointers.</returns>
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

/**
 * \defgroup TrainingCpp Ort Training C++ API
 * @{
 */

/** \brief Holds the state of the training session.
 *
 * This class holds the entire training session state that includes model parameters, their gradients,
 * optimizer parameters, and user properties. The Ort::TrainingSession leverages the Ort::CheckpointState
 * by accessing and updating the contained training state.
 * \note Note that the training session created with a checkpoint state uses this state to store the entire
 * training state (including model parameters, its gradients, the optimizer states and the properties).
 * The Ort::TrainingSession does not hold a copy of the Ort::CheckpointState and as a result, it is required
 * that the checkpoint state outlive the lifetime of the training session.
 * \note Note that the checkpoint state can be either the complete checkpoint state or the nominal checkpoint
 * state depending on the version provided while loading the checkpoint.
 *
 */
class CheckpointState : public detail::Base<OrtCheckpointState> {
 private:
  CheckpointState(OrtCheckpointState* checkpoint_state) { p_ = checkpoint_state; }

 public:
  // Construct the checkpoint state by loading the checkpoint by calling LoadCheckpoint
  CheckpointState() = delete;

  /// \name Accessing The Training Session State
  /// @{

  /** \brief Load a checkpoint state from a file on disk into checkpoint_state.
   *
   * This function will parse a checkpoint file, pull relevant data and load the training
   * state and return an instance of Ort::CheckpointState. This checkpoint state can then be used to create the
   * training session by instantiating Ort::TrainingSession. By doing so, the training session will resume
   * training from the given checkpoint state.
   *
   * \param[in] path_to_checkpoint Path to the checkpoint file
   * \return Ort::CheckpointState object which holds the state of the training session parameters.
   *
   */
  static CheckpointState LoadCheckpoint(const std::basic_string<ORTCHAR_T>& path_to_checkpoint);

  /** \brief Load a checkpoint state from a buffer.
   *
   * This function will parse a checkpoint buffer, pull relevant data and load the training
   * state and return an instance of Ort::CheckpointState. This checkpoint state can then be used to create the
   * training session by instantiating Ort::TrainingSession. By doing so, the training session will resume
   * training from the given checkpoint state.
   *
   * \param[in] buffer Buffer containing the checkpoint data.
   * \return Ort::CheckpointState object which holds the state of the training session parameters.
   *
   */
  static CheckpointState LoadCheckpointFromBuffer(const std::vector<uint8_t>& buffer);

  /** \brief Save the given state to a checkpoint file on disk.
   *
   * This function serializes the provided checkpoint state to a file on disk.
   * This checkpoint can later be loaded by invoking Ort::CheckpointState::LoadCheckpoint to resume
   * training from this snapshot of the state.
   *
   * \param[in] checkpoint_state The checkpoint state to save.
   * \param[in] path_to_checkpoint Path to the checkpoint file.
   * \param[in] include_optimizer_state Flag to indicate whether to save the optimizer state or not.
   *
   */
  static void SaveCheckpoint(const CheckpointState& checkpoint_state,
                             const std::basic_string<ORTCHAR_T>& path_to_checkpoint,
                             const bool include_optimizer_state = false);

  /** \brief Adds or updates the given property to/in the checkpoint state.
   *
   * Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
   * state by the user by calling this function with the corresponding property name and value.
   * The given property name must be unique to be able to successfully add the property.
   *
   * \param[in] property_name Name of the property being added or updated.
   * \param[in] property_value Property value associated with the given name.
   *
   */
  void AddProperty(const std::string& property_name, const Property& property_value);

  /** \brief Gets the property value associated with the given name from the checkpoint state.
   *
   * Gets the property value from an existing entry in the checkpoint state. The property must
   * exist in the checkpoint state to be able to retrieve it successfully.
   *
   * \param[in] property_name Name of the property being retrieved.
   * \return Property value associated with the given property name.
   *
   */
  Property GetProperty(const std::string& property_name);

  /** \brief Updates the data associated with the model parameter in the checkpoint state for the given parameter name.
   *
   * This function updates a model parameter in the checkpoint state with the given parameter data.
   * The training session must be already created with the checkpoint state that contains the parameter
   * being updated. The given parameter is copied over to the registered device for the training session.
   * The parameter must exist in the checkpoint state to be able to update it successfully.
   *
   * \param[in] parameter_name Name of the parameter being updated.
   * \param[in] parameter The parameter data that should replace the existing parameter data.
   *
   */
  void UpdateParameter(const std::string& parameter_name, const Value& parameter);

  /** \brief Gets the data associated with the model parameter from the checkpoint state for the given parameter name.
   *
   * This function retrieves the model parameter data from the checkpoint state for the given parameter name.
   * The parameter is copied over to the provided OrtValue. The training session must be already created
   * with the checkpoint state that contains the parameter being retrieved.
   * The parameter must exist in the checkpoint state to be able to retrieve it successfully.
   *
   * \param[in] parameter_name Name of the parameter being retrieved.
   * \return The parameter data that is retrieved from the checkpoint state.
   *
   */
  Value GetParameter(const std::string& parameter_name);

  /// @}
};

/** \brief Trainer class that provides training, evaluation and optimizer methods for training an ONNX models.
 *
 * The training session requires four training artifacts
 * - The training onnx model
 * - The evaluation onnx model (optional)
 * - The optimizer onnx model
 * - The checkpoint file
 *
 * These artifacts can be generated using the `onnxruntime-training` python [utility](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).
 *
 */
class TrainingSession : public detail::Base<OrtTrainingSession> {
 private:
  size_t training_model_output_count_, eval_model_output_count_;

 public:
  /// \name Constructing the Training Session
  /// @{
  /** \brief Create a training session that can be used to begin or resume training.
   *
   * This constructor instantiates the training session based on the env and session options provided that can
   * begin or resume training from a given checkpoint state for the given onnx models.
   * The checkpoint state represents the parameters of the training session which will be moved
   * to the device specified by the user through the session options (if necessary).
   *
   * \param[in] env Env to be used for the training session.
   * \param[in] session_options SessionOptions that the user can customize for this training session.
   * \param[in] checkpoint_state Training states that the training session uses as a starting point for training.
   * \param[in] train_model_path Model to be used to perform training.
   * \param[in] eval_model_path Model to be used to perform evaluation.
   * \param[in] optimizer_model_path Model to be used to perform gradient descent.
   *
   */
  TrainingSession(const Env& env, const SessionOptions& session_options, CheckpointState& checkpoint_state,
                  const std::basic_string<ORTCHAR_T>& train_model_path,
                  const std::optional<std::basic_string<ORTCHAR_T>>& eval_model_path = std::nullopt,
                  const std::optional<std::basic_string<ORTCHAR_T>>& optimizer_model_path = std::nullopt);

  /** \brief Create a training session that can be used to begin or resume training.
   * This constructor allows the users to load the models from buffers instead of files.
   *
   * \param[in] env Env to be used for the training session.
   * \param[in] session_options SessionOptions that the user can customize for this training session.
   * \param[in] checkpoint_state Training states that the training session uses as a starting point for training.
   * \param[in] train_model_data Buffer containing training model data.
   * \param[in] eval_model_data Buffer containing evaluation model data.
   * \param[in] optim_model_data Buffer containing optimizer model (used for performing weight/parameter update).
   *
   */
  TrainingSession(const Env& env, const SessionOptions& session_options, CheckpointState& checkpoint_state,
                  const std::vector<uint8_t>& train_model_data, const std::vector<uint8_t>& eval_model_data = {},
                  const std::vector<uint8_t>& optim_model_data = {});
  /// @}

  /// \name Implementing The Training Loop
  /// @{
  /** \brief Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
   *
   * This function performs a training step that computes the outputs of the training model and the gradients
   * of the trainable parameters for the given inputs. The train step is performed based on the training model
   * that was provided to the training session.
   * The Ort::TrainingSession::TrainStep is equivalent of running forward propagation and backward propagation in a single
   * step.
   * The gradients computed are stored inside the training session state so they can be later consumed
   * by the Ort::TrainingSession::OptimizerStep function.
   * The gradients can be lazily reset by invoking the Ort::TrainingSession::LazyResetGrad function.
   *
   * \param[in] input_values The user inputs to the training model.
   * \return A std::vector of Ort::Value objects that represents the output of the forward pass of the training model.
   *
   *
   */
  std::vector<Value> TrainStep(const std::vector<Value>& input_values);

  /** \brief Reset the gradients of all trainable parameters to zero lazily.
   *
   * This function sets the internal state of the training session such that the gradients of the trainable
   * parameters in the OrtCheckpointState will be scheduled to be reset just before the new gradients are
   * computed on the next invocation of the next Ort::TrainingSession::TrainStep.
   *
   */
  void LazyResetGrad();

  /** \brief Computes the outputs for the eval model for the given inputs
   *
   * This function performs an eval step that computes the outputs of the eval model for the given inputs.
   * The eval step is performed based on the eval model that was provided to the training session.
   *
   * \param[in] input_values The user inputs to the eval model.
   * \return A std::vector of Ort::Value objects that represents the output of the eval pass.
   *
   */
  std::vector<Value> EvalStep(const std::vector<Value>& input_values);

  /** \brief Sets the learning rate for this training session.
   *
   * This function allows users to set the learning rate for the training session. The current
   * learning rate is maintained by the training session and can be overwritten by invoking
   * this function with the desired learning rate. This function should not be used when a valid
   * learning rate scheduler is registered. It should be used either to set the learning rate
   * derived from a custom learning rate scheduler or to set a constant learning rate to be used
   * throughout the training session.
   * \note Please note that this function does not set the initial learning rate that may be needed
   * by the predefined learning rate schedulers. To set the initial learning rate for learning
   * rate schedulers, please look at the function Ort::TrainingSession::RegisterLinearLRScheduler.
   *
   * \param[in] learning_rate Desired learning rate to be set.
   *
   */
  void SetLearningRate(float learning_rate);

  /** \brief Gets the current learning rate for this training session.
   *
   * This function allows users to get the learning rate for the training session. The current
   * learning rate is maintained by the training session, and users can query it for the purpose
   * of implementing their own learning rate schedulers.
   *
   * \return float representing the current learning rate.
   *
   */
  float GetLearningRate() const;

  /** \brief Registers a linear learning rate scheduler for the training session.
   *
   * Register a linear learning rate scheduler that decays the learning rate by linearly updated
   * multiplicative factor from the initial learning rate set on the training session to 0. The decay
   * is performed after the initial warm up phase where the learning rate is linearly incremented
   * from 0 to the initial learning rate provided.
   *
   * \param[in] warmup_step_count Warmup steps for LR warmup.
   * \param[in] total_step_count Total step count.
   * \param[in] initial_lr The initial learning rate to be used by the training session.
   *
   */
  void RegisterLinearLRScheduler(int64_t warmup_step_count, int64_t total_step_count,
                                 float initial_lr);

  /** \brief Update the learning rate based on the registered learing rate scheduler.
   *
   * Takes a scheduler step that updates the learning rate that is being used by the training session.
   * This function should typically be called before invoking the optimizer step for each round,
   * or as determined necessary to update the learning rate being used by the training session.
   * \note Please note that a valid predefined learning rate scheduler must be first registered to invoke this
   * function.
   *
   */
  void SchedulerStep();

  /** \brief Performs the weight updates for the trainable parameters using the optimizer model.
   *
   * This function performs the weight update step that updates the trainable parameters such that they
   * take a step in the direction of their gradients (gradient descent). The optimizer step is performed
   * based on the optimizer model that was provided to the training session.
   * The updated parameters are stored inside the training state so that they can be used by the next
   * Ort::TrainingSession::TrainStep function call.
   *
   */
  void OptimizerStep();

  /// @}

  /// \name Prepare For Inferencing
  /// @{

  /** \brief Export a model that can be used for inferencing.
   *
   * If the training session was provided with an eval model, the training session can generate
   * an inference model if it knows the inference graph outputs. The input inference graph outputs
   * are used to prune the eval model so that the inference model's outputs align with the provided outputs.
   * The exported model is saved at the path provided and can be used for inferencing with Ort::Session.
   * \note Note that the function re-loads the eval model from the path provided to Ort::TrainingSession
   * and expects that this path still be valid.
   *
   * \param[in] inference_model_path Path where the inference model should be serialized to.
   * \param[in] graph_output_names Names of the outputs that are needed in the inference model.
   *
   */
  void ExportModelForInferencing(const std::basic_string<ORTCHAR_T>& inference_model_path,
                                 const std::vector<std::string>& graph_output_names);

  /// @}

  /// \name Model IO Information
  /// @{
  /** \brief Retrieves the names of the user inputs for the training and eval models.
   *
   * This function returns the names of inputs of the training or eval model that can be associated
   * with the Ort::Value(s) provided to the Ort::TrainingSession::TrainStep or Ort::TrainingSession::EvalStep
   * function.
   *
   * \param[in] training Whether the training model input names are requested or eval model input names.
   * \return Graph input names for either the training model or the eval model.
   *
   */
  std::vector<std::string> InputNames(const bool training);

  /** \brief Retrieves the names of the user outputs for the training and eval models.
   *
   * This function returns the names of outputs of the training or eval model that can be associated
   * with the Ort::Value(s) returned by the Ort::TrainingSession::TrainStep or Ort::TrainingSession::EvalStep
   * function.
   *
   * \param[in] training Whether the training model output names are requested or eval model output names.
   * \return Graph output names for either the training model or the eval model.
   *
   */
  std::vector<std::string> OutputNames(const bool training);

  /// @}

  /// \name Accessing The Training Session State
  /// @{

  /** \brief Returns a contiguous buffer that holds a copy of all training state parameters
   *
   * \param[in] only_trainable Whether to only copy trainable parameters or to copy all parameters.
   * \return Contiguous buffer to the model parameters.
   *
   */
  Value ToBuffer(const bool only_trainable);

  /** \brief Loads the training session model parameters from a contiguous buffer
   *
   * In case the training session was created with a nominal checkpoint, invoking this function is required
   * to load the updated parameters onto the checkpoint to complete it.
   *
   * \param[in] buffer Contiguous buffer to load the parameters from.
   */
  void FromBuffer(Value& buffer);

  /// @}
};

/// \name Training Utilities
/// @{
/** \brief This function sets the seed for generating random numbers.
 *
 * Use this function to generate reproducible results. It should be noted that completely
 * reproducible results are not guaranteed.
 *
 * \param[in] seed Manual seed to use for random number generation.
 */
void SetSeed(const int64_t seed);
/// @}

/// @}

}  // namespace Ort

#include "onnxruntime_training_cxx_inline.h"
