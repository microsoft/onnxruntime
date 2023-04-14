// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the training c apis.

#pragma once
#include <stdbool.h>
#include "onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(TrainingSession);  /// Type that enables performing training for the given user models.
ORT_RUNTIME_CLASS(CheckpointState);  /// Type that holds the training states for the training session.

typedef enum OrtPropertyType {
  OrtIntProperty = 0,
  OrtFloatProperty = 1,
  OrtStringProperty = 2,
} OrtPropertyType;

struct OrtTrainingApi {
  /** \brief Load a checkpoint state from directory on disk into checkpoint_state.
   *
   * This function will parse a checkpoint directory, pull relevant files and load the training
   * state into the checkpoint_state. This checkpoint state can then be used to create the
   * training session by invoking CreateTrainingSession. By doing so, the training session will resume
   * training from the given checkpoint state.
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
   * This checkpoint can later be loaded by invoking LoadCheckpoint to continue the training with the same state.
   *
   * \param[in] checkpoint_state The checkpoint state to save.
   * \param[in] checkpoint_path Path to the checkpoint directory.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SaveCheckpoint, _In_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* checkpoint_path);

  /** \brief Create a training session that can be used to begin or resume training.
   *
   * This function creates a training session based on the env and session options provided that can
   * begin or resume training from a given checkpoint state for the given onnx models.
   * The checkpoint state represents the parameters of the training session which will be moved
   * to the device specified by the user through the session options (if necessary).
   *
   * \param[in] env Environment to be used for the training session.
   * \param[in] options Session options that the user can customize for this training session.
   * \param[in] checkpoint_state Training states that the training session uses as a starting point for training.
   * \param[in] train_model_path Model to be used to perform training that can be generated using the offline tooling library.
   * \param[in] eval_model_path Model to be used to perform evaluation that can be generated using the offline tooling library.
   * \param[in] optimizer_model_path Model to be used to the optimizer step for weight updates. The model can be generated using the offline tooling library.
   * \param[out] out Created training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CreateTrainingSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                  _Inout_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* train_model_path,
                  _In_ const ORTCHAR_T* eval_model_path, _In_ const ORTCHAR_T* optimizer_model_path,
                  _Outptr_ OrtTrainingSession** out);

  /** \brief Retrieves the number of user outputs in the training model.
   *
   * This function returns the number of outputs of the training model so that the user can
   * allocate space for the number of outputs when TrainStep is invoked.
   *
   * \param[in] sess The training session which owns the training model.
   * \param[out] out Number of user outputs in the training model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the number of user outputs in the eval model.
   *
   * This function returns the number of outputs of the eval model so that the user can
   * allocate space for the number of outputs when EvalStep is invoked.
   *
   * \param[in] sess The training session which owns the eval model.
   * \param[out] out Number of user outputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  ORT_API2_STATUS(TrainingSessionGetTrainingModelOutputName, _In_ const OrtTrainingSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

  ORT_API2_STATUS(TrainingSessionGetEvalModelOutputName, _In_ const OrtTrainingSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

  /** \brief Reset the training model gradients to zero lazily.
   *
   * This function sets the internal state of the training session such that the training model gradients
   * will be scheduled to be reset just before the new gradients are computed on the next invocation
   * of TrainStep.
   *
   * \param[in] session The training session which owns the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(LazyResetGrad, _Inout_ OrtTrainingSession* session);

  /** \brief Computes the outputs and the gradients for the training model for the given inputs
   *
   * This function performs a training step that computes the outputs and the gradients of the training model
   * for the given inputs. The train step is performed based on the training model that was provided
   * to the training session.
   * The gradients computed are stored inside the training session so they can be later consumed
   * by the OptimizerStep function.
   *
   * \param[in] sess The training session which owns the eval model.
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
   * \param[in] sess The training session which owns the eval model.
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
   * derived from a custom learning rate scheduler or to set the learning rate constant to be used
   * throughout the training session.
   * Please note that this function does not set the initial learning rate that may be needed
   * by the predefined learning rate schedulers. To set the initial learning rate for learning
   * rate schedulers, please look at the function `RegisterLinearLRScheduler`.
   *
   * \param[in] sess The training session on which the learning rate needs to be set.
   * \param[in] learning_rate Desired learning rate to set.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SetLearningRate, _Inout_ OrtTrainingSession* sess, _In_ float learning_rate);

  /** \brief Gets the current learning rate for this training session.
   *
   * This function allows users to get the learning rate for the training session. The current
   * learning rate is maintained by the training session
   *
   * \param[in] sess The training session on which the learning rate needs to be set.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetLearningRate, _Inout_ OrtTrainingSession* sess, _Out_ float* learning_rate);

  /** \brief Performs the weight updates for the trainable parameters using the optimizer model.
   *
   * This function performs the weight update step that updates the trainable parameters such that they
   * take a step in the direction of their gradients. The optimizer step is performed based on the optimizer
   * model that was provided to the training session.
   * The updated parameters are stored inside the training session so that they can be used by the next
   * TrainStep function call.
   *
   * \param[in] sess The training session which owns the optimizer model.
   * \param[in] run_options Run options for this eval step.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(OptimizerStep, _Inout_ OrtTrainingSession* sess,
                  _In_opt_ const OrtRunOptions* run_options);

  /** \brief Registers a Linear learning rate scheduler for the training session.
   *
   * Register a linear learning rate scheduler that decays the learning rate by linearly updated
   * multiplicative factor from the initial learning rate set on the training session to 0. The decay
   * is performed after the initial warm up phase where the learning rate is linearly incremented
   * from 0 to the initial learning rate provided.
   *
   * \param[in] sess The training session that should use the linear learning rate scheduler.
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
   * Please note that a valid predefined learning rate scheduler must be first registered to invoke this
   * function.
   *
   * \param[in] sess The training session that has the registered learning rate scheduler.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SchedulerStep, _Inout_ OrtTrainingSession* sess);

  /** \brief Retrieves the size of all the parameters.
   *
   * Calculates the size of all the parameters for the training session.
   * When 'trainable_only' is true, the size is calculated for trainable params only.
   *
   * \param[in] sess The training session.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetParametersSize, _Inout_ OrtTrainingSession* sess,
                  _Out_ size_t* out, bool trainable_only);

  /** \brief Copy parameters onto contiguous buffer held by parameters_buffer
   *
   * The parameters_buffer has to be of the size given by GetParametersSize api call,
   * with matching setting for 'trainable_only'. All the target parameters must be of the same
   * datatype. The OrtValue must be pre-allocated onto
   * the desired device. This is a complementary function to 'CopyBufferToParameters'.
   * Parameter ordering is preserved.
   * User is responsible for allocating/freeing the 'parameters_buffer'.
   *
   * \param[in] sess The training session.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy onto.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CopyParametersToBuffer, _Inout_ OrtTrainingSession* sess,
                  _Inout_ OrtValue* parameters_buffer, bool trainable_only);

  /** \brief Copy parameter values from contiguous buffer held by parameters_buffer onto parameters
   *
   * The parameters_buffer has to be of the size given by GetParametersSize api call,
   * with matching setting for 'trainable_only'. All the target parameters must be of the same
   * datatype. This is a complementary function to 'CopyBufferToParameters'
   * and can be used to load updated buffer values onto the parameters.
   * Parameter ordering is preserved.
   * User is responsible for allocating/freeing the 'parameters_buffer'.
   *
   * \param[in] sess The training session.
   * \param[in] trainable_only Whether to skip non-trainable parameters
   * \param[out] parameters_buffer The pre-allocated OrtValue buffer to copy from.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(CopyBufferToParameters, _Inout_ OrtTrainingSession* sess,
                  _Inout_ OrtValue* parameters_buffer, bool trainable_only);

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
   *
   */
  ORT_CLASS_RELEASE(CheckpointState);

  /** \brief Export a model that can be used for inferencing.
   *
   * If the training session was provided with an eval model, the training session can generate
   * an inference model if it knows the inference graph outputs. The input inference graph outputs
   * are used to prune the eval model so that the output model's outputs align with the provided outputs.
   * The exported model is saved at the path provided and can be used for inferencing with InferenceSession.
   * Note that the function re-loads the eval model from the path provided to CreateTrainingSession and expects
   * that this path still be valid.
   *
   * \param[in] sess The training session.
   * \param[in] inference_model_path Path where the inference model should be serialized to.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(ExportModelForInferencing, _Inout_ OrtTrainingSession* sess,
                  _In_ const ORTCHAR_T* inference_model_path, size_t graph_outputs_len,
                  _In_reads_(graph_outputs_len) const char* const* graph_output_names);

  /** \brief Sets the seed used for random number generation in Onnxruntime.
   *
   * Use this to get deterministic results.
   *
   * \param[in] seed The seed to be set.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(SetSeed, _In_ const int64_t seed);

  /** \brief Retrieves the number of user inputs in the training model.
   *
   * \param[in] sess The training session which owns the training model.
   * \param[out] out Number of user inputs in the training model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the number of user inputs in the eval model.
   *
   * \param[in] sess The training session which owns the eval model.
   * \param[out] out Number of user inputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the name of the user input at given index in the training model.
   *
   * \param[in] sess The training session which owns the training model.
   * \param[out] out Number of user inputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetTrainingModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                  _In_ OrtAllocator* allocator, _Outptr_ char** output);

  /** \brief Retrieves the name of the user input at given index in the eval model.
   *
   * \param[in] sess The training session which owns the training model.
   * \param[out] out Number of user inputs in the eval model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(TrainingSessionGetEvalModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                  _In_ OrtAllocator* allocator, _Outptr_ char** output);

  /** \brief Gets the current state of the training session.
   *
   * The state contains information about the model parameters and their gradients.
   * It also contains information about the optimizer parameters should the user request it.
   * The returned state can be used to save a checkpoint or analyze the model parameters.
   *
   * \param[in] session The training session.
   * \param[in] include_optimizer_state Whether or not to include optimizer states in the returned state.
   * \param[out] checkpoint_state The current state of the training session.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   */
  ORT_API2_STATUS(GetState, _In_ const OrtTrainingSession* session, bool include_optimizer_state,
                  _Outptr_ OrtCheckpointState** checkpoint_state);

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
   * Property values are allocated on the heap. The user must free up the memory as needed by their application.
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
                  _Out_ enum OrtPropertyType* property_type, _Out_ void** property_value);
};

typedef struct OrtTrainingApi OrtTrainingApi;
