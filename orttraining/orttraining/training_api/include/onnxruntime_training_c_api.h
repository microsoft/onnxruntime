// This file contains c apis for on device training
// This file should never be included standalone
// It is included from within core/session/onnxruntime_c_api.h when
// on device training is enabled
// These apis can be moved to core/session/onnxruntime_c_api.h once they stabilize

// DO NOT UNCOMMENT
//#include "core/session/onnxruntime_c_api.h"

ORT_API2_STATUS(LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path, _Outptr_ OrtCheckpointState** checkpoint_state);


ORT_API2_STATUS(SaveCheckpoint, _In_ const ORTCHAR_T* checkpoint_path, _Inout_ OrtTrainingSession* session,
                  bool save_optimizer_state);

ORT_API2_STATUS(CreateTrainingSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                  _Outptr_ OrtTrainingSession** out);

ORT_API2_STATUS(InitializeTrainingSession, _Inout_ OrtTrainingSession* session,
                    _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const ORTCHAR_T* train_model_path, _In_ const ORTCHAR_T* eval_model_path,
                    _In_ const ORTCHAR_T* optimizer_model_path);

ORT_API2_STATUS(ResetGrad, _Inout_ OrtTrainingSession* session);

ORT_API2_STATUS(TrainStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(inputs_len) const OrtValue* const* inputs, size_t inputs_len,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

ORT_API2_STATUS(EvalStep, _Inout_ OrtTrainingSession* sess, _In_opt_ const OrtRunOptions* run_options,
                  _In_reads_(inputs_len) const OrtValue* const* inputs, size_t inputs_len,
                  size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

ORT_API2_STATUS(OptimizerStep, _Inout_ OrtTrainingSession* sess,
                    _In_opt_ const OrtRunOptions* run_options);

ORT_CLASS_RELEASE(TrainingSession);
ORT_CLASS_RELEASE(CheckpointState);
