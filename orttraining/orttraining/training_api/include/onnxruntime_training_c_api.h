// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the training c apis.

#pragma once
#include "core/session/onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(TrainingSession);  /// Type that enables performing training for the given user models.
ORT_RUNTIME_CLASS(CheckpointState);  /// Type that holds the training states for the training session.

struct OrtTrainingApi {
  /** \brief Load a checkpoint state from directory on disk into checkpoint_state.
  *
  * This function will parse a checkpoint directory, pull relevant files and load the training
  * states into the checkpoint_state. This checkpoint state can then be used to create the
  * training session by invoking CreateTrainingSession. By doing so, the training session will resume
  * training from the given checkpoint.
  *
  * \param[in] checkpoint_path Path to the checkpoint directory
  * \param[out] checkpoint_state Checkpoint states that contains the states of the training session.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path,
                  _Outptr_ OrtCheckpointState** checkpoint_state);

  /** \brief Save the training session states to a checkpoint directory on disk.
  *
  * This function retrieves the training session states from the training session and serializes them
  * to a checkpoint directory on disk. This checkpoint can later be loaded by invoking LoadCheckpoint
  * to continue the training with the same states.
  *
  * \param[in] checkpoint_path Path to the checkpoint directory
  * \param[in] session The training session from where the checkpoint states are to be retrieved.
  * \param[in] save_optimizer_state Boolean flag indicating whether or not to save the optimizer states to the checkpoint.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(SaveCheckpoint, _In_ const ORTCHAR_T* checkpoint_path, _In_ const OrtTrainingSession* session,
                  bool save_optimizer_state);

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
  * \param[in] sess The training session which has working knowledge of the training model.
  * \param[out] out Number of user outputs in the training model.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(TrainingSessionGetTrainModeOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Retrieves the number of user outputs in the eval model.
  *
  * This function returns the number of outputs of the eval model so that the user can
  * allocate space for the number of outputs when EvalStep is invoked.
  *
  * \param[in] sess The training session which has working knowledge of the eval model.
  * \param[out] out Number of user outputs in the eval model.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(TrainingSessionGetEvalModeOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  /** \brief Reset the training model gradients to zero lazily.
  *
  * This function sets the internal state of the training session such that the training model gradients
  * will be reset just before the new gradients are computed on the next invocation of TrainStep.
  *
  * \param[in] session The training session which has working knowledge of the eval model.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(ResetGrad, _Inout_ OrtTrainingSession* session);

  /** \brief Computes the outputs and the gradients for the training model for the given inputs
  *
  * This function performs a training step that computes the outputs and the gradients of the training model
  * for the given inputs. The train step is performed based on the training model that was provided
  * to the training session.
  * The gradients computed are stored inside the training session so they can be later consumed
  * by the OptimizerStep function.
  *
  * \param[in] sess The training session which has working knowledge of the eval model.
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
                  size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

  /** \brief Computes the outputs for the eval model for the given inputs
  *
  * This function performs an eval step that computes the outputs of the eval model for the given inputs.
  * The eval step is performed based on the eval model that was provided to the training session.
  *
  * \param[in] sess The training session which has working knowledge of the eval model.
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
                  size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

  /** \brief Performs the weight updates for the trainable parameters using the optimizer model.
  *
  * This function performs the weight update step that updates the trainable parameters such that they
  * take a step in the direction of their gradients. The optimizer step is performed based on the optimizer
  * model that was provided to the training session.
  * The updated parameters are stored inside the training session so that they can be used by the next
  * TrainStep function call.
  *
  * \param[in] sess The training session which has working knowledge of the optimizer model.
  * \param[in] run_options Run options for this eval step.
  *
  * \snippet{doc} snippets.dox OrtStatus Return Value
  *
  */
  ORT_API2_STATUS(OptimizerStep, _Inout_ OrtTrainingSession* sess,
                  _In_opt_ const OrtRunOptions* run_options);

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
};

typedef struct OrtTrainingApi OrtTrainingApi;
