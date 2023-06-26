// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "training_api.h"
#include "api.h"

#include "onnxruntime_training_cxx_api.h"

#define CHECK_TRAINING_STATUS(ORT_API_NAME, ...) \
  CheckStatus(Ort::GetTrainingApi().ORT_API_NAME(__VA_ARGS__))

OrtCheckpointState* OrtTrainingLoadCheckpoint(void* checkpoint, size_t checkpoint_size) {
  OrtCheckpointState* checkpointState = nullptr;
  return (CHECK_TRAINING_STATUS(LoadCheckpointFromBuffer, checkpoint, checkpoint_size, &checkpointState) == ORT_OK)
             ? checkpointState
             : nullptr;
}


// void EMSCRIPTEN_KEEPALIVE OrtTrainingSaveCheckpoint(const orttraining_checkpoint_handle_t checkpoint_state,
//                                                     const ORTCHAR_T* path_to_checkpoint,
//                                                     const bool include_optimizer_state) {
//   Ort::GetTrainingApi().SaveCheckpoint(checkpoint_state, path_to_checkpoint, include_optimizer_state);
// }

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseCheckpoint(orttraining_checkpoint_handle_t checkpoint) {
  Ort::GetTrainingApi().ReleaseCheckpointState(checkpoint);
}


	OrtTrainingSession* EMSCRIPTEN_KEEPALIVE OrtTrainingCreateTrainingSession(const ort_session_options_handle_t options,
                                                                          orttraining_checkpoint_handle_t checkpoint_state,
                                                                          void* train_model,
                                                                          size_t train_size,
                                                                          void* eval_model,
                                                                          size_t eval_size,
                                                                          void* optimizer_model,
                                                                          size_t optimizer_size) {
  OrtTrainingSession* training_session = nullptr;
  return (CHECK_TRAINING_STATUS(CreateTrainingSessionFromArray, g_env, options, checkpoint_state,
                                train_model, train_size, eval_model, eval_size, optimizer_model,
                                optimizer_size,
                                &training_session) == ORT_OK)
                                ? training_session
                                : nullptr;
}

void EMSCRIPTEN_KEEPALIVE OrtTrainingLazyResetGrad(orttraining_session_handle_t session) {
  Ort::GetTrainingApi().LazyResetGrad(session);
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStepWithOptions(orttraining_session_handle_t session,
                                                               const ort_run_options_handle_t options,
                                                               const size_t inputs_len,
                                                               const ort_tensor_handle_t* inputs,
                                                               const size_t outputs_len,
                                                               ort_tensor_handle_t* outputs
                                                               ) {
  return (CHECK_TRAINING_STATUS(TrainStep, session, options, inputs_len, inputs, outputs_len, outputs) == ORT_OK)
                                ? *outputs
                                : nullptr;
}

OrtValue* EMSCRIPTEN_KEEPALIVE OrtTrainingTrainStep(orttraining_session_handle_t session,
                                                               const size_t inputs_len,
                                                               const ort_tensor_handle_t* inputs,
                                                               const size_t outputs_len,
                                                               ort_tensor_handle_t* outputs
                                                               ) {
  OrtRunOptions* run_options = nullptr;
  return OrtTrainingTrainStepWithOptions(session, run_options, inputs_len, inputs, outputs_len, outputs);
}

void EMSCRIPTEN_KEEPALIVE OrtTrainingOptimizerStep(orttraining_session_handle_t session,
                                                   const ort_run_options_handle_t run_options) {
  Ort::GetTrainingApi().OptimizerStep(session, run_options);
}

// void EMSCRIPTEN_KEEPALIVE OrtTrainingExportModelForInferencing(orttraining_session_handle_t session,
//                                                                const ORTCHAR_T* inference_model_path,
//                                                                size_t graph_outputs_len,
//                                                                const char* const* graph_output_names) {
//   Ort::GetTrainingApi().ExportModelForInferencing(session, inference_model_path, graph_outputs_len, graph_output_names);
// }

void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(orttraining_session_handle_t session) {
  Ort::GetTrainingApi().ReleaseTrainingSession(session);
}
