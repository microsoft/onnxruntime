// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtTrainingApis {

ORT_API(const OrtTrainingApi*, GetTrainingApi, uint32_t version);

ORT_API_STATUS_IMPL(CreateTrainingSession, _In_ const OrtEnv* env, _In_ const OrtSessionOptions* options,
                    _Inout_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* train_model_path,
                    _In_ const ORTCHAR_T* eval_model_path, _In_ const ORTCHAR_T* optimizer_model_path,
                    _Outptr_result_maybenull_ OrtTrainingSession** out);

ORT_API_STATUS_IMPL(CreateTrainingSessionFromBuffer, _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const void* train_model_data, size_t train_data_length,
                    _In_ const void* eval_model_data, size_t eval_data_length,
                    _In_ const void* optim_model_data, size_t optim_data_length,
                    _Outptr_result_maybenull_ OrtTrainingSession** out);

ORT_API_STATUS_IMPL(TrainingSessionGetTrainingModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

ORT_API_STATUS_IMPL(TrainingSessionGetEvalModelOutputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

ORT_API_STATUS_IMPL(TrainingSessionGetTrainingModelOutputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

ORT_API_STATUS_IMPL(TrainingSessionGetEvalModelOutputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** output);

ORT_API_STATUS_IMPL(LazyResetGrad, _Inout_ OrtTrainingSession* session);

ORT_API_STATUS_IMPL(TrainStep, _Inout_ OrtTrainingSession* session, _In_opt_ const OrtRunOptions* run_options,
                    _In_ size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                    _In_ size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

ORT_API_STATUS_IMPL(EvalStep, _In_ const OrtTrainingSession* session, _In_opt_ const OrtRunOptions* run_options,
                    _In_ size_t inputs_len, _In_reads_(inputs_len) const OrtValue* const* inputs,
                    _In_ size_t outputs_len, _Inout_updates_all_(outputs_len) OrtValue** outputs);

ORT_API_STATUS_IMPL(SetLearningRate, _Inout_ OrtTrainingSession* sess, _In_ float learning_rate);

ORT_API_STATUS_IMPL(GetLearningRate, _Inout_ OrtTrainingSession* sess, _Out_ float* learning_rate);

ORT_API_STATUS_IMPL(OptimizerStep, _Inout_ OrtTrainingSession* session, _In_opt_ const OrtRunOptions* run_options);

ORT_API_STATUS_IMPL(RegisterLinearLRScheduler, _Inout_ OrtTrainingSession* sess, _In_ const int64_t warmup_step_count,
                    _In_ const int64_t total_step_count, _In_ const float initial_lr);

ORT_API_STATUS_IMPL(SchedulerStep, _Inout_ OrtTrainingSession* sess);

ORT_API_STATUS_IMPL(LoadCheckpoint, _In_ const ORTCHAR_T* checkpoint_path,
                    _Outptr_ OrtCheckpointState** checkpoint_state);

ORT_API_STATUS_IMPL(SaveCheckpoint, _In_ OrtCheckpointState* checkpoint_state, _In_ const ORTCHAR_T* checkpoint_path,
                    const bool include_optimizer_state);

ORT_API_STATUS_IMPL(GetParametersSize, _Inout_ OrtTrainingSession* sess,
                    _Out_ size_t* out, bool trainable_only);

ORT_API_STATUS_IMPL(CopyParametersToBuffer, _Inout_ OrtTrainingSession* sess,
                    _Inout_ OrtValue* parameters_buffer, bool trainable_only);

ORT_API_STATUS_IMPL(CopyBufferToParameters, _Inout_ OrtTrainingSession* sess,
                    _Inout_ OrtValue* parameters_buffer, bool trainable_only);

ORT_API(void, ReleaseCheckpointState, _Frees_ptr_opt_ OrtCheckpointState* checkpoint_state);

ORT_API(void, ReleaseTrainingSession, _Frees_ptr_opt_ OrtTrainingSession* session);

ORT_API_STATUS_IMPL(ExportModelForInferencing, _Inout_ OrtTrainingSession* sess,
                    _In_ const ORTCHAR_T* inference_model_path, size_t graph_outputs_len,
                    _In_reads_(graph_outputs_len) const char* const* graph_output_names);

ORT_API_STATUS_IMPL(SetSeed, _In_ const int64_t seed);

ORT_API_STATUS_IMPL(TrainingSessionGetTrainingModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

ORT_API_STATUS_IMPL(TrainingSessionGetEvalModelInputCount, _In_ const OrtTrainingSession* sess, _Out_ size_t* out);

ORT_API_STATUS_IMPL(TrainingSessionGetTrainingModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _In_ OrtAllocator* allocator, _Outptr_ char** output);

ORT_API_STATUS_IMPL(TrainingSessionGetEvalModelInputName, _In_ const OrtTrainingSession* sess, size_t index,
                    _In_ OrtAllocator* allocator, _Outptr_ char** output);

ORT_API_STATUS_IMPL(AddProperty, _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const char* property_name, _In_ enum OrtPropertyType property_type,
                    _In_ void* property_value);

ORT_API_STATUS_IMPL(GetProperty, _In_ const OrtCheckpointState* checkpoint_state,
                    _In_ const char* property_name, _Inout_ OrtAllocator* allocator,
                    _Out_ enum OrtPropertyType* property_type, _Outptr_ void** property_value);

ORT_API_STATUS_IMPL(LoadCheckpointFromBuffer, _In_ const void* checkpoint_buffer,
                    _In_ const size_t num_bytes, _Outptr_ OrtCheckpointState** checkpoint_state);

ORT_API_STATUS_IMPL(GetParameterTypeAndShape, _In_ const OrtCheckpointState* checkpoint_state,
                    _In_ const char* parameter_name, _Outptr_ OrtTensorTypeAndShapeInfo** parameter_type_and_shape);

ORT_API_STATUS_IMPL(UpdateParameter, _Inout_ OrtCheckpointState* checkpoint_state,
                    _In_ const char* parameter_name, _In_ OrtValue* parameter);

ORT_API_STATUS_IMPL(GetParameter, _In_ const OrtCheckpointState* checkpoint_state,
                    _In_ const char* parameter_name, _Inout_ OrtAllocator* allocator,
                    _Outptr_ OrtValue** parameter);

}  // namespace OrtTrainingApis
