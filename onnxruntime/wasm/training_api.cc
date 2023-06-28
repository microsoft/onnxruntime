// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_training_cxx_api.h"
#include "training_api.h"

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

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseCheckpoint(orttraining_handle_t trainingHandle) {
  Ort::GetTrainingApi().ReleaseCheckpointState(trainingHandle->checkpointState);
}


OrtTrainingManager* EMSCRIPTEN_KEEPALIVE OrtTrainingCreateTrainingSession(const ort_session_options_handle_t options,
                                                                          orttraining_handle_t trainingHandle,
                                                                          void* train_model,
                                                                          size_t train_size,
                                                                          void* eval_model,
                                                                          size_t eval_size,
                                                                          void* optimizer_model,
                                                                          size_t optimizer_size) {
  trainingHandle->trainingSession = nullptr;
  return (CHECK_TRAINING_STATUS(CreateTrainingSessionFromArray, g_env, options, trainingHandle->checkpointState,
                                train_model, train_size, eval_model, eval_size, optimizer_model,
                                optimizer_size,
                                &trainingHandle->trainingSession) == ORT_OK)
                                ? trainingHandle
                                : nullptr;
}

void EMSCRIPTEN_KEEPALIVE OrtTrainingLazyResetGrad(orttraining_handle_t trainingHandle) {
  Ort::GetTrainingApi().LazyResetGrad(trainingHandle->trainingSession);
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStepWithOptions(orttraining_handle_t trainingHandle,
                                                               const ort_run_options_handle_t options,
                                                               const size_t inputs_len,
                                                               const ort_tensor_handle_t* inputs,
                                                               const size_t outputs_len,
                                                               ort_tensor_handle_t* outputs
                                                               ) {
  return (CHECK_TRAINING_STATUS(TrainStep, trainingHandle->trainingSession, options, inputs_len, inputs, outputs_len, outputs) == ORT_OK)
                                ? *outputs
                                : nullptr;
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStep(orttraining_handle_t trainingHandle,
                                                               const size_t inputs_len,
                                                               const ort_tensor_handle_t* inputs,
                                                               const size_t outputs_len,
                                                               ort_tensor_handle_t* outputs
                                                               ) {
  OrtRunOptions* run_options = nullptr;
  return OrtTrainingTrainStepWithOptions(trainingHandle, run_options, inputs_len, inputs, outputs_len, outputs);
}

void EMSCRIPTEN_KEEPALIVE OrtTrainingOptimizerStep(orttraining_handle_t trainingHandle,
                                                   const ort_run_options_handle_t run_options) {
  Ort::GetTrainingApi().OptimizerStep(trainingHandle->trainingSession, run_options);
}

// void EMSCRIPTEN_KEEPALIVE OrtTrainingExportModelForInferencing(orttraining_session_handle_t session,
//                                                                const ORTCHAR_T* inference_model_path,
//                                                                size_t graph_outputs_len,
//                                                                const char* const* graph_output_names) {
//   Ort::GetTrainingApi().ExportModelForInferencing(session, inference_model_path, graph_outputs_len, graph_output_names);
// }

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(orttraining_handle_t trainingHandle) {
  Ort::GetTrainingApi().ReleaseTrainingSession(trainingHandle->trainingSession);
}
