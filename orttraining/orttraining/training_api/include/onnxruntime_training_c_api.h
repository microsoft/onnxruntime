// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the training c apis.

#pragma once
#include "core/session/onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(TrainingSession);
ORT_RUNTIME_CLASS(CheckpointState);

struct OrtTrainingApi {
  ORT_API2_STATUS(LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path, _Outptr_ OrtCheckpointState** checkpoint_state);

  ORT_API2_STATUS(SaveCheckpoint, _In_ const ORTCHAR_T* checkpoint_path, _Inout_ OrtTrainingSession* session,
                  bool save_optimizer_state);

  ORT_API2_STATUS(CreateTrainingSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                  _Inout_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* train_model_path,
                  _In_ const ORTCHAR_T* eval_model_path, _In_ const ORTCHAR_T* optimizer_model_path,
                  _Outptr_ OrtTrainingSession** out);

  ORT_API2_STATUS(TrainingSessionGetTrainModeOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  ORT_API2_STATUS(TrainingSessionGetEvalModeOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

  ORT_API2_STATUS(ResetGrad, _Inout_ OrtTrainingSession* session);

  ORT_API2_STATUS(TrainStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

  ORT_API2_STATUS(EvalStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

  ORT_API2_STATUS(OptimizerStep, _Inout_ OrtTrainingSession* sess,
                  _In_opt_ const OrtRunOptions* run_options);

  ORT_CLASS_RELEASE(TrainingSession);
  ORT_CLASS_RELEASE(CheckpointState);
};

typedef struct OrtTrainingApi OrtTrainingApi;
