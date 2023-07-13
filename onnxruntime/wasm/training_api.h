// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This file contains declarations of exported functions as WebAssembly API.
// Unlike a normal C-API, the purpose of this API is to make emcc to generate correct exports for the WebAssembly. The
// macro "EMSCRIPTEN_KEEPALIVE" helps the compiler to mark the function as an exported funtion of the WebAssembly
// module. Users are expected to consume those functions from JavaScript side.

#pragma once

#include <emscripten.h>

#include <stddef.h>

struct OrtTrainingManager;
using orttraining_handle_t = OrtTrainingManager*;

struct OrtSessionOptions;
using ort_session_options_handle_t = OrtSessionOptions*;

struct OrtRunOptions;
using ort_run_options_handle_t = OrtRunOptions*;

struct OrtValue;
using ort_tensor_handle_t = OrtValue*;

extern "C" {
/**
 * Loads in a checkpoint & creates an instance of a checkpoint state.
 * @param checkpoint pointer to the checkpoint data buffer
 * @param checkpoint_size size of the checkpoint buffer in bytes
 * @returns a handle of the ORT checkpoint state.
 */
orttraining_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingLoadCheckpoint(void* checkpoint, size_t checkpoint_size);

/**
 * Release the CheckpointState in the given TrainingManager.
 * @param training_handle
 */
void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseCheckpoint(orttraining_handle_t training_handle);

/**
 * Saves the specified checkpoint state to the specified filepath.
 * @param checkpoint_state handle of specified checkpoint state to save
 * @param path_to_checkpoint filepath to the directory where the checkpoint should be saved to
 * @param include_optimizer_state (optional; defaults to false) indicates whether to save the optimizer state
 */
// void EMSCRIPTEN_KEEPALIVE OrtTrainingSaveCheckpoint(const orttraining_checkpoint_handle_t checkpoint_state,
//                                                     const char** path_to_checkpoint,
//                                                     const bool include_optimizer_state = false);

/**
 * Creates an instance of a training session that can be used to begin or resume training from a given checkpoint state
 * for the given onnx models.
 * @param options Session options that the user can customize for this training session.
 * @param checkpoint_state Training states that the training session uses as a starting point for training.
 * @param train_model pointer to a buffer containing the ONNX training model
 * @param train_size size of the train_model buffer in bytes
 * @param eval_model pointer to a buffer containing the ONNX evaluation model
 * @param eval_size size of the eval_model buffer in bytes
 * @param optimizer_model pointer to a buffer containing the ONNX optimizer model
 * @param optimizer_size size of the optimizer_model buffer in bytes
 * @return a handle of the ORT training session
 *
 */

orttraining_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCreateTrainingSession(const ort_session_options_handle_t options,
                                                                           orttraining_handle_t training_handle,
                                                                           void* train_model,
                                                                           size_t train_size,
                                                                           void* eval_model,
                                                                           size_t eval_size,
                                                                           void* optimizer_model,
                                                                           size_t optimizer_size);

/**
 * Resets the gradients of all trainable parameters to zero for the TrainingSession in the given training_handle
 * @param training_handle handle of the training session
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingLazyResetGrad(orttraining_handle_t training_handle);

/**
 * Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs with
 * run options.
 * @param session handle of the training session
 * @param options handle of the run options for this training step
 * @param inputs_len number of user inputs to the training model
 * @param inputs the user inputs to the training model
 * @param outputs_len number of user outputs expected from this training setp
 * @param outputs user outputs computed by train step
 * @return handler to the outputs
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStepWithOptions(orttraining_handle_t training_handle,
                                                                         const ort_run_options_handle_t options,
                                                                         const size_t inputs_len,
                                                                         const ort_tensor_handle_t* inputs,
                                                                         const size_t outputs_len,
                                                                         ort_tensor_handle_t* outputs);

/**
 * Computes the outputs of the training model and the gradients of the trainable parameters for the given inputs.
 * @param session handle of the training session
 * @param inputs_len number of user inputs to the training model
 * @param inputs the user inputs to the training model
 * @param outputs_len number of user outputs expected from this training setp
 * @param outputs user outputs computed by train step
 * @return handler to the outputs
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStep(orttraining_handle_t session,
                                                              const size_t inputs_len,
                                                              const ort_tensor_handle_t* inputs,
                                                              const size_t outputs_len,
                                                              ort_tensor_handle_t* outputs);

/**
 * Performs weight updates for the trainable parameters in the given training session using the optimizer model.
 * @param training_handle handle of the training manager
 * @param run_options optional parameter of run options for this training step
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingOptimizerStep(orttraining_handle_t training_handle,
                                                  const ort_run_options_handle_t run_options = nullptr);

ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingEvalStep(orttraining_handle_t training_handle,
                                                             const ort_run_options_handle_t options,
                                                             size_t inputs_len,
                                                             const ort_tensor_handle_t* inputs,
                                                             size_t outputs_len,
                                                             ort_tensor_handle_t* outputs);

// void EMSCRIPTEN_KEEPALIVE OrtTrainingExportModelForInferencing(orttraining_session_handle_t session,
//                                                                const char** inference_model_path,
//                                                                size_t graph_outputs_len,
//                                                                const char** graph_output_names);

/**
 * Retrieves the size of all parameters for the training state.
 * When the trainable_only argument is true, the size is calculated for trainable params only.
 *
 * @param training_handle the TrainingManager
 * @param trainable_only skips non-trainable parameters when true.
 */
size_t* EMSCRIPTEN_KEEPALIVE OrtTrainingGetParametersSize(orttraining_handle_t training_handle,
                                                          bool trainable_only);

/**
 * Copy all parameters to a contiguous buffer held by the argument parameters_buffer
 *
 * User is responsible for allocating and freeing resources used by the parameters_buffer.
 * Parameter ordering is preserved.
 *
 * @param training_handle handle of the training manager
 * @param parameters_buffer pre-allocated OrtValue buffer to copy onto. Must be same size as results of
 *                          GetParametersSize api call
 * @param trainable_only whether to skip non-trainable parameters
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCopyParametersToBuffer(orttraining_handle_t training_handle,
                                                                           ort_tensor_handle_t parameters_buffer,
                                                                           bool trainable_only);

/**
 * Copy parameters values from given contiguous buffer held by parameters_buffer to the training state.
 * Parameter ordering is preserved.
 * @param training_handle handle of the training manager
 * @param parameters_buffer OrtValue buffer to copy from. Must be same size as results of
 *                          GetParametersSize api call
 * @param trainable_only whether to skip non-trainable parameters
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCopyBufferToParameters(orttraining_handle_t training_handle,
                                                                           ort_tensor_handle_t parameters_buffer,
                                                                           bool trainable_only);

/**
 * Release the TrainingSession in the given TrainingManager.
 */
void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(orttraining_handle_t training_handle);

/**
 * Release the given TrainingManager.
 */
void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseHandle(orttraining_handle_t training_handle);
};
