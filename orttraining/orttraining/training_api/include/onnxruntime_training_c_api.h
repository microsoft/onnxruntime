// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the training c apis.

#pragma once
#include <stdbool.h>
#include "onnxruntime_c_api.h"

/** \page training_c_cpp_api Training C & C++ APIs
 *
 * Training C and C++ APIs are an extension of the \ref c_cpp_api "onnxruntime core C and C++ APIs" and should be used in conjunction with them.
 *
 * In order to train a model with onnxruntime, the following training artifacts must be generated:
 * - The training onnx model
 * - The checkpoint directory
 * - The optimizer onnx model
 * - The eval onnx model model (optional)
 *
 * These training artifacts can be generated as part of an offline step using the python [utilities](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md) made available in the `onnxruntime-training` python package.
 *
 * After these artifacts have been generated, the C and C++ utilities listed in this documentation can be leveraged to perform training.
 *
 * If any problem is encountered, please create an [issue](https://github.com/microsoft/onnxruntime/issues/new) with your scenario and requirements, and we will be sure to respond and follow up on the request.
 *
 * <h1>Training C API</h1>
 *
 * ::OrtTrainingApi - Training C API functions.
 *
 * This C structure contains functions that enable users to perform training with onnxruntime.
 *
 * _Sample Code_:
 *
 * ```c
 * #include <onnxruntime_training_api.h>
 *
 * OrtApi* g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
 * OrtTrainingApi* g_ort_training_api = g_ort_api->GetTrainingApi(ORT_API_VERSION);
 *
 * OrtEnv* env = NULL;
 * g_ort_api->CreateEnv(logging_level, logid, &env);
 * OrtSessionOptions* session_options = NULL;
 * g_ort_api->CreateSessionOptions(&session_options);
 *
 * OrtCheckpointState* state = NULL;
 * g_ort_training_api->LoadCheckpoint(path_to_checkpoint, &state);
 *
 * OrtTrainingSession* training_session = NULL;
 * g_ort_training_api->CreateTrainingSession(env, session_options, training_model_path,
 *                                           state, eval_model_path, optimizer_model_path,
 *                                           &training_session);
 * // Training loop
 * {
 *     g_ort_training_api->TrainStep(...);
 *     g_ort_training_api->OptimizerStep(...);
 *     g_ort_training_api->LazyResetGrad(...);
 * }
 *
 * g_ort_training_api->ExportModelForInferencing(training_session, inference_model_path, ...);
 * g_ort_training_api->SaveCheckpoint(state, path_to_checkpoint, false);
 *
 * g_ort_training_api->ReleaseTrainingSession(training_session);
 * g_ort_training_api->ReleaseCheckpointState(state);
 * ```
 *
 * > **Note**
 * > The ::OrtCheckpointState contains the entire training state that the ::OrtTrainingSession uses. As a result, the training session must always have access to the state. That is to say, the ::OrtCheckpointState instance must outlive the lifetime of the ::OrtTrainingSession instance.
 *
 * <h1>Training C++ API</h1>
 *
 * @ref TrainingCpp - Training C++ API classes and functions.
 *
 * These C++ classes and functions enable users to perform training with onnxruntime.
 *
 * _Sample Code_:
 *
 * ```cc
 * #include <onnxruntime_training_cxx_api.h>
 *
 * Ort::Env env;
 * Ort::SessionOptions session_options;
 *
 * auto state = Ort::CheckpointState::LoadCheckpoint(path_to_checkpoint);
 * auto training_session = Ort::TrainingSession(env, session_options, state, training_model_path,
 *                                              eval_model_path, optimizer_model_path);
 *
 * // Training Loop
 * {
 *     training_session.TrainStep(...);
 *     training_session.OptimizerStep(...);
 *     training_session.LazyResetGrad(...);
 * }
 *
 * training_session->ExportModelForInferencing(inference_model_path, ...);
 * Ort::CheckpointState::SaveCheckpoint(state, path_to_checkpoint, false);
 * ```
 * > **Note**
 * > The ::Ort::CheckpointState contains the entire training state that the ::Ort::TrainingSession uses. As a result, the training session must always have access to the state. That is to say, the ::Ort::CheckpointState instance must outlive the lifetime of the ::Ort::TrainingSession instance.
 */

/** @defgroup TrainingC Ort Training C API
 * @{
 */
ORT_RUNTIME_CLASS(TrainingSession);  // Type that enables performing training for the given user models.
ORT_RUNTIME_CLASS(CheckpointState);  // Type that holds the training states for the training session.

/** \brief Type of property to be added to or returned from the ::OrtCheckpointState.
 */
typedef enum OrtPropertyType {
  OrtIntProperty = 0,
  OrtFloatProperty = 1,
  OrtStringProperty = 2,
} OrtPropertyType;

/** \brief The Training C API that holds onnxruntime training function pointers
 *
 * All the Training C API functions are defined inside this structure as pointers to functions.
 * Call OrtApi::GetTrainingApi to get a pointer to this struct.
 *
 * \nosubgrouping
 */
struct OrtTrainingApi {
  /// \name Accessing The Training Session State
  /// @{

  /** \brief Load a checkpoint state from directory on disk into checkpoint_state.
   *
   * This function will parse a checkpoint directory, pull relevant files and load the training
   * state into the checkpoint_state. This checkpoint state can then be used to create the
   * training session by invoking OrtTrainingApi::CreateTrainingSession. By doing so, the training
   * session will resume training from the given checkpoint state.
   * \note Note that the training session created with a checkpoint state uses this state to store the entire
   * training state (including model parameters, its gradients, the optimizer states and the properties).
   * As a result, it is required that the checkpoint state outlive the lifetime of the training session.
   *
   * \param[in] checkpoint_path Path to the checkpoint directory
   * \param[out] checkpoint_state Checkpoint state that contains the states of the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path,
                  _Outptr_ OrtCheckpointState** checkpoint_state);

  /** \brief Save the given state to a checkpoint directory on disk.
   *
   * This function serializes the provided checkpoint state to a directory on disk.
   * This checkpoint can later be loaded by invoking OrtTrainingApi::LoadCheckpoint to resume
   * training from this snapshot of the state.
   *
   * \param[in] checkpoint_state The checkpoint state to save.
   * \param[in] checkpoint_path Path to the checkpoint directory.
   * \param[in] include_optimizer_state Flag to indicate whether to save the optimizer state or not.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SaveCheckpoint, _In_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* checkpoint_path,
                  const bool include_optimizer_state);

  /// @}

  /// \name Implementing The Training Loop
  /// @{
  /** \brief Create a training session that can be used to begin or resume training.
   *
   * This function creates a training session based on the env and session options provided that can
   * begin or resume training from a given checkpoint state for the given onnx models.
   * The checkpoint state represents the parameters of the training session which will be moved
   * to the device specified by the user through the session options (if necessary).
   * The training session requires four training artifacts
   * - The training onnx model
   * - The evaluation onnx model (optional)
   * - The optimizer onnx model
   * - The checkpoint directory
   *
   * These artifacts can be generated using the `onnxruntime-training` python [utility](https://github.com/microsoft/onnxruntime/blob/main/orttraining/orttraining/python/training/onnxblock/README.md).
   *
   * \param[in] env Environment to be used for the training session.
   * \param[in] options Session options that the user can customize for this training session.
   * \param[in] checkpoint_state Training states that the training session uses as a starting point for training.
   * \param[in] train_model_path Model to be used to perform training.
   * \param[in] eval_model_path Model to be used to perform evaluation.
   * \param[in] optimizer_model_path Model to be used to perform gradient descent.
   * \param[out] out Created training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CreateTrainingSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                  _Inout_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* train_model_path,
                  _In_ const ORTCHAR_T* eval_model_path, _In_ const ORTCHAR_T* optimizer_model_path,
                  _Outptr_ OrtTrainingSession** out);

  /// @}

  /// \name Model IO Information
  /// @{

  /** \brief Retrieves the number of user outputs in the training model.
   *
   * This function returns the number of outputs of the training model so that the user can
   * allocate space for the number of outputs when OrtTrainingApi::TrainStep is invoked.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] out Number of user outputs in the training model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the number of user outputs in the eval model.
   *
   * This function returns the number of outputs of the eval model so that the user can
   * allocate space for the number of outputs when OrtTrainingApi::EvalStep is invoked.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] out Number of user outputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the names of user outputs in the training model.
   *
   * This function returns the names of outputs of the training model that can be associated with the OrtValue(s)
   * returned by the OrtTrainingApi::TrainStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] index Index of the output name requested.
   * \param[in] allocator Allocator to use to allocate the memory for the name.
   * \param[out] output Name of the training model output at the given index.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelOutputName, _In_ const OrtTrainingSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

  /** \brief Retrieves the names of user outputs in the eval model.
   *
   * This function returns the names of outputs of the eval model that can be associated with the OrtValue(s) returned
   * by the OrtTrainingApi::EvalStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] index Index of the output name requested.
   * \param[in] allocator Allocator to use to allocate the memory for the name.
   * \param[out] output Name of the eval model output at the given index.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelOutputName, _In_ const OrtTrainingSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

  /// @}

  /// \name Implementing The Training Loop
  /// @{

  /** \brief Reset the gradients of all trainable parameters to zero lazily.
   *
   * This function sets the internal state of the training session such that the gradients of the trainable
   * parameters in the OrtCheckpointState will be scheduled to be reset just before the new gradients are
   * computed on the next invocation of the next OrtTrainingApi::TrainStep.
   *
   * \param[in] session The `this` pointer to the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(LazyResetGrad, _Inout_ OrtTrainingSession* session);

  /** \brief Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs
   *
   * This function performs a training step that computes the outputs of the training model and the gradients
   * of the trainable parameters for the given inputs. The train step is performed based on the training model
   * that was provided to the training session.
   * The OrtTrainingApi::TrainStep is equivalent of running forward propagation and backward propagation in a single
   * step.
   * The gradients computed are stored inside the training session state so they can be later consumed
   * by the OrtTrainingApi::OptimizerStep function.
   * The gradients can be lazily reset by invoking the OrtTrainingApi::LazyResetGrad function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] run_options Run options for this training step.
   * \param[in] inputs_len Number of user inputs to the training model.
   * \param[in] inputs The user inputs to the training model.
   * \param[in] outputs_len Number of user outputs expected from this training step.
   * \param[out] outputs User outputs computed by train step.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_ size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  _In_ size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

  /** \brief Computes the outputs for the eval model for the given inputs
   *
   * This function performs an eval step that computes the outputs of the eval model for the given inputs.
   * The eval step is performed based on the eval model that was provided to the training session.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] run_options Run options for this eval step.
   * \param[in] inputs_len Number of user inputs to the eval model.
   * \param[in] inputs The user inputs to the eval model.
   * \param[in] outputs_len Number of user outputs expected from this eval step.
   * \param[out] outputs User outputs computed by eval step.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(EvalStep, _In_ const OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_ size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  _In_ size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

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
   * rate schedulers, please look at the function OrtTrainingApi::RegisterLinearLRScheduler.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] learning_rate Desired learning rate to be set.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SetLearningRate, _Inout_ OrtTrainingSession* sess, _In_ float learning_rate);

  /** \brief Gets the current learning rate for this training session.
   *
   * This function allows users to get the learning rate for the training session. The current
   * learning rate is maintained by the training session, and users can query it for the purpose
   * of implementing their own learning rate schedulers.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] learning_rate Learning rate currently in use by the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetLearningRate, _Inout_ OrtTrainingSession* sess, _Out_ float* learning_rate);

  /** \brief Performs the weight updates for the trainable parameters using the optimizer model.
   *
   * This function performs the weight update step that updates the trainable parameters such that they
   * take a step in the direction of their gradients (gradient descent). The optimizer step is performed
   * based on the optimizer model that was provided to the training session.
   * The updated parameters are stored inside the training state so that they can be used by the next
   * OrtTrainingApi::TrainStep function call.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] run_options Run options for this optimizer step.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(OptimizerStep, _Inout_ OrtTrainingSession* sess,
                  _In_opt_ const OrtRunOptions* run_options);

  /** \brief Registers a linear learning rate scheduler for the training session.
   *
   * Register a linear learning rate scheduler that decays the learning rate by linearly updated
   * multiplicative factor from the initial learning rate set on the training session to 0. The decay
   * is performed after the initial warm up phase where the learning rate is linearly incremented
   * from 0 to the initial learning rate provided.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] warmup_step_count Warmup steps for LR warmup.
   * \param[in] total_step_count Total step count.
   * \param[in] initial_lr The initial learning rate to be used by the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(RegisterLinearLRScheduler, _Inout_ OrtTrainingSession* sess, _In_ const int64_t warmup_step_count,
                  _In_ const int64_t total_step_count, _In_ const float initial_lr);

  /** \brief Update the learning rate based on the registered learing rate scheduler.
   *
   * Takes a scheduler step that updates the learning rate that is being used by the training session.
   * This function should typically be called before invoking the optimizer step for each round,
   * or as determined necessary to update the learning rate being used by the training session.
   * \note Please note that a valid predefined learning rate scheduler must be first registered to invoke this
   * function.
   *
   * \param[in] sess The `this` pointer to the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SchedulerStep, _Inout_ OrtTrainingSession* sess);

  /// @}

  /// \name Accessing The Training Session State
  /// @{
  /** \brief Retrieves the size of all the parameters.
   *
   * Calculates the total number of primitive (datatype of the parameters) elements of all the parameters in the
   * training state.
   * When trainable_only argument is true, the size is calculated for trainable params only.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] out Size of all parameter elements.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetParametersSize, _Inout_ OrtTrainingSession* sess, _Out_ size_t* out, bool trainable_only);

  /** \brief Copy all parameters to a contiguous buffer held by the argument parameters_buffer
   *
   * The parameters_buffer has to be of the size given by GetParametersSize api call,
   * with matching setting for the argument trainable_only. All the target parameters must be of the same
   * datatype. The OrtValue must be pre-allocated onto
   * the desired device. This is a complementary function to OrtTrainingApi::CopyBufferToParameters.
   * Parameter ordering is preserved.
   * User is responsible for allocating and freeing the resources used by the parameters_buffer.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy onto.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CopyParametersToBuffer, _Inout_ OrtTrainingSession* sess,
                  _Inout_ OrtValue* parameters_buffer, bool trainable_only);

  /** \brief Copy parameter values from the given contiguous buffer held by parameters_buffer to the training state
   *
   * The parameters_buffer argument has to be of the size given by OrtTrainingApi::GetParametersSize api call,
   * with matching setting for trainable_only argument. All the target parameters must be of the same
   * datatype. This is a complementary function to OrtTrainingApi::CopyBufferToParameters
   * and can be used to load updated buffer values onto the training state.
   * Parameter ordering is preserved.
   * User is responsible for allocating and freeing the resources used by the parameters_buffer.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy from.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CopyBufferToParameters, _Inout_ OrtTrainingSession* sess,
                  _Inout_ OrtValue* parameters_buffer, bool trainable_only);

  /// @}

  /// \name Release Training Resources
  /// @{

  /** \brief Frees up the memory used up by the training session.
   *
   * This function frees up any memory that was allocated in the training session. The training
   * session can no longer be used after this call.
   *
   */
  ORT_CLASS_RELEASE(TrainingSession);

  /** \brief Frees up the memory used up by the checkpoint state.
   *
   * This function frees up any memory that was allocated in the checkpoint state. The checkpoint
   * state can no longer be used after this call.
   * \note Note that the checkpoint state must be released only after the training session has been released.
   *
   */
  ORT_CLASS_RELEASE(CheckpointState);

  /// @}

  /// \name Prepare For Inferencing
  /// @{
  /** \brief Export a model that can be used for inferencing.
   *
   * If the training session was provided with an eval model, the training session can generate
   * an inference model if it knows the inference graph outputs. The input inference graph outputs
   * are used to prune the eval model so that the inference model's outputs align with the provided outputs.
   * The exported model is saved at the path provided and can be used for inferencing with InferenceSession.
   * \note Note that the function re-loads the eval model from the path provided to OrtTrainingApi::CreateTrainingSession
   * and expects that this path still be valid.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] inference_model_path Path where the inference model should be serialized to.
   * \param[in] graph_outputs_len Size of the graph output names array.
   * \param[in] graph_output_names Names of the outputs that are needed in the inference model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(ExportModelForInferencing, _Inout_ OrtTrainingSession* sess,
                  _In_ const ORTCHAR_T* inference_model_path, size_t graph_outputs_len,
                  _In_reads_(graph_outputs_len) const char* const* graph_output_names);

  /// @}

  /// \name Training Utilities
  /// @{
  /** \brief Sets the seed used for random number generation in Onnxruntime.
   *
   * Use this function to generate reproducible results. It should be noted that completely reproducible
   * results are not guaranteed.
   *
   * \param[in] seed The seed to be set.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SetSeed, _In_ const int64_t seed);

  /// @}

  /// \name Model IO Information
  /// @{
  /** \brief Retrieves the number of user inputs in the training model.
   *
   * This function returns the number of inputs of the training model so that the user can accordingly
   * allocate the OrtValue(s) provided to the OrtTrainingApi::TrainStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] out Number of user inputs in the training model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the number of user inputs in the eval model.
   *
   * This function returns the number of inputs of the eval model so that the user can accordingly
   * allocate the OrtValue(s) provided to the OrtTrainingApi::EvalStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[out] out Number of user inputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the name of the user input at given index in the training model.
   *
   * This function returns the names of inputs of the training model that can be associated with the
   * OrtValue(s) provided to the OrtTrainingApi::TrainStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] index The index of the training model input name requested.
   * \param[in] allocator The allocator to use to allocate the memory for the requested name.
   * \param[out] output Name of the user input for the training model at the given index.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                  _In_ OrtAllocator* allocator, _Outptr_ char** output);

  /** \brief Retrieves the name of the user input at given index in the eval model.
   *
   * This function returns the names of inputs of the eval model that can be associated with the OrtValue(s) provided
   * to the OrtTrainingApi::EvalStep function.
   *
   * \param[in] sess The `this` pointer to the training session.
   * \param[in] index The index of the eval model input name requested.
   * \param[in] allocator The allocator to use to allocate the memory for the requested name.
   * \param[out] output Name of the user input for the eval model at the given index.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                  _In_ OrtAllocator* allocator, _Outptr_ char** output);

  /// @}

  /// \name Accessing The Training Session State
  /// @{

  /** \brief Adds the given property to the checkpoint state.
   *
   * Runtime properties such as epoch, training step, best score, and others can be added to the checkpoint
   * state by the user if they desire by calling this function with the appropriate property name and
   * value. The given property name must be unique to be able to successfully add the property.
   *
   * \param[in] checkpoint_state The checkpoint state which should hold the property.
   * \param[in] property_name Unique name of the property being added.
   * \param[in] property_type Type of the property associated with the given name.
   * \param[in] property_value Property value associated with the given name.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(AddProperty, _Inout_ OrtCheckpointState* checkpoint_state,
                  _In_ const char* property_name, _In_ enum OrtPropertyType property_type,
                  _In_ void* property_value);

  /** \brief Gets the property value associated with the given name from the checkpoint state.
   *
   * Gets the property value from an existing entry in the checkpoint state. The property must
   * exist in the checkpoint state to be able to retrieve it successfully.
   *
   * \param[in] checkpoint_state The checkpoint state that is currently holding the property.
   * \param[in] property_name Unique name of the property being retrieved.
   * \param[in] allocator Allocator used to allocate the memory for the property_value.
   * \param[out] property_type Type of the property associated with the given name.
   * \param[out] property_value Property value associated with the given name.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetProperty, _In_ const OrtCheckpointState* checkpoint_state,
                  _In_ const char* property_name, _Inout_ OrtAllocator* allocator,
                  _Out_ enum OrtPropertyType* property_type, _Outptr_ void** property_value);

  /// @}
};

typedef struct OrtTrainingApi OrtTrainingApi;

/// @}
