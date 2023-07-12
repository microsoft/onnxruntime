// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_training_cxx_api.h"
#include "training_api.h"
#include "api.h"

struct OrtTrainingManager {
  OrtTrainingSession* trainingSession;
  OrtCheckpointState* checkpointState;
};

#define CHECK_TRAINING_STATUS(ORT_API_NAME, ...) \
  CheckStatus(Ort::GetTrainingApi().ORT_API_NAME(__VA_ARGS__))

OrtTrainingManager* OrtTrainingLoadCheckpoint(void* checkpoint, size_t checkpoint_size) {
  OrtTrainingManager* trainingManager;
  trainingManager->trainingSession = nullptr;
  trainingManager->checkpointState = nullptr;
  return (CHECK_TRAINING_STATUS(LoadCheckpointFromBuffer, checkpoint, checkpoint_size, &trainingManager->checkpointState) == ORT_OK)
             ? trainingManager
             : nullptr;
}

// void EMSCRIPTEN_KEEPALIVE OrtTrainingSaveCheckpoint(const orttraining_checkpoint_handle_t checkpoint_state,
//                                                     const ORTCHAR_T* path_to_checkpoint,
//                                                     const bool include_optimizer_state) {
//   Ort::GetTrainingApi().SaveCheckpoint(checkpoint_state, path_to_checkpoint, include_optimizer_state);
// }

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseCheckpoint(orttraining_handle_t training_handle) {
  Ort::GetTrainingApi().ReleaseCheckpointState(training_handle->checkpointState);
}

OrtTrainingManager* EMSCRIPTEN_KEEPALIVE OrtTrainingCreateTrainingSession(const ort_session_options_handle_t options,
                                                                          orttraining_handle_t training_handle,
                                                                          void* train_model,
                                                                          size_t train_size,
                                                                          void* eval_model,
                                                                          size_t eval_size,
                                                                          void* optimizer_model,
                                                                          size_t optimizer_size) {
  training_handle->trainingSession = nullptr;
  return (CHECK_TRAINING_STATUS(CreateTrainingSessionFromArray, OrtGlobals::g_env, options, training_handle->checkpointState,
                                train_model, train_size, eval_model, eval_size, optimizer_model,
                                optimizer_size,
                                &training_handle->trainingSession) == ORT_OK)
             ? training_handle
             : nullptr;
}

int EMSCRIPTEN_KEEPALIVE OrtTrainingLazyResetGrad(orttraining_handle_t training_handle) {
  return CHECK_TRAINING_STATUS(LazyResetGrad, training_handle->trainingSession);
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStepWithOptions(orttraining_handle_t training_handle,
                                                               const ort_run_options_handle_t options,
                                                               const size_t inputs_len,
                                                               const ort_tensor_handle_t* inputs,
                                                               const size_t outputs_len,
                                                               ort_tensor_handle_t* outputs) {
  return (CHECK_TRAINING_STATUS(TrainStep, training_handle->trainingSession, options, inputs_len, inputs, outputs_len, outputs) == ORT_OK)
             ? *outputs
             : nullptr;
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStep(orttraining_handle_t training_handle,
                                                    const size_t inputs_len,
                                                    const ort_tensor_handle_t* inputs,
                                                    const size_t outputs_len,
                                                    ort_tensor_handle_t* outputs) {
  OrtRunOptions* run_options = nullptr;
  return OrtTrainingTrainStepWithOptions(training_handle, run_options, inputs_len, inputs, outputs_len, outputs);
}

int EMSCRIPTEN_KEEPALIVE OrtTrainingOptimizerStep(orttraining_handle_t training_handle,
                                                   const ort_run_options_handle_t run_options) {
  return CHECK_TRAINING_STATUS(OptimizerStep, training_handle->trainingSession, run_options);
}

// void EMSCRIPTEN_KEEPALIVE OrtTrainingExportModelForInferencing(orttraining_session_handle_t session,
//                                                                const ORTCHAR_T* inference_model_path,
//                                                                size_t graph_outputs_len,
//                                                                const char* const* graph_output_names) {
//   Ort::GetTrainingApi().ExportModelForInferencing(session, inference_model_path, graph_outputs_len, graph_output_names);
// }

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(orttraining_handle_t training_handle) {
  Ort::GetTrainingApi().ReleaseTrainingSession(training_handle->trainingSession);
}

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(orttraining_handle_t training_handle) {
  // TODO: ask somebody how to free the struct :(
    if (training_handle->checkpointState)
      OrtTrainingReleaseCheckpoint(training_handle);
    if (training_handle->trainingSession)
      OrtTrainingReleaseSession(training_handle);

    delete training_handle;
}

size_t* EMSCRIPTEN_KEEPALIVE OrtTrainingGetParametersSize(orttraining_handle_t training_handle,
                                                            bool trainable_only) {
  size_t* param_size = nullptr;
  return (CHECK_TRAINING_STATUS(GetParametersSize, training_handle->trainingSession, param_size, trainable_only) == ORT_OK)
             ? param_size
             : nullptr;
}

ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCopyParametersToBuffer(orttraining_handle_t training_handle,
                                                                            ort_tensor_handle_t parameters_buffer,
                                                                            bool trainableOnly) {
  return (CHECK_TRAINING_STATUS(CopyParametersToBuffer, training_handle->trainingSession, parameters_buffer, trainable_only) == ORT_OK)
             ? parameters_buffer
             : nullptr;
}

ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCopyBufferToParameters(orttraining_handle_t training_handle,
                                                                            ort_tensor_handle_t parameters_buffer,
                                                                            bool trainableOnly) {
  return (CHECK_TRAINING_STATUS(CopyBufferToParameters, training_handle->trainingSession, parameters_buffer, trainable_only) == ORT_OK)
             ? parameters_buffer
             : nullptr;
}
