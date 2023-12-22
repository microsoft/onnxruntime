// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace OrtApis {

ORT_API(const OrtApi*, GetApi, uint32_t version);

ORT_API(const char*, GetVersionString);

ORT_API(void, ReleaseEnv, OrtEnv*);
ORT_API(void, ReleaseStatus, _Frees_ptr_opt_ OrtStatus*);
ORT_API(void, ReleaseMemoryInfo, _Frees_ptr_opt_ OrtMemoryInfo*);
ORT_API(void, ReleaseSession, _Frees_ptr_opt_ OrtSession*);
ORT_API(void, ReleaseValue, _Frees_ptr_opt_ OrtValue*);
ORT_API(void, ReleaseRunOptions, _Frees_ptr_opt_ OrtRunOptions*);
ORT_API(void, ReleaseTypeInfo, _Frees_ptr_opt_ OrtTypeInfo*);
ORT_API(void, ReleaseTensorTypeAndShapeInfo, _Frees_ptr_opt_ OrtTensorTypeAndShapeInfo*);
ORT_API(void, ReleaseSessionOptions, _Frees_ptr_opt_ OrtSessionOptions*);
ORT_API(void, ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain*);
ORT_API(void, ReleaseMapTypeInfo, _Frees_ptr_opt_ OrtMapTypeInfo*);
ORT_API(void, ReleaseSequenceTypeInfo, _Frees_ptr_opt_ OrtSequenceTypeInfo*);
ORT_API(void, ReleaseModelMetadata, _Frees_ptr_opt_ OrtModelMetadata*);

_Check_return_ _Ret_notnull_ [[nodiscard]] OrtStatus* ORT_API_CALL CreateStatus(OrtErrorCode code, _In_z_ const char* msg)
    NO_EXCEPTION;

OrtErrorCode ORT_API_CALL GetErrorCode(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;
const char* ORT_API_CALL GetErrorMessage(_In_ const OrtStatus* status) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS_IMPL(CreateEnv, OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateEnvWithCustomLogger, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel logging_level, _In_ const char* logid, _Outptr_ OrtEnv** out);
ORT_API_STATUS_IMPL(CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level, _In_ const char* logid,
                    _In_ const struct OrtThreadingOptions* t_options, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out)
ORT_ALL_ARGS_NONNULL;

ORT_API_STATUS_IMPL(EnableTelemetryEvents, _In_ const OrtEnv* env);
ORT_API_STATUS_IMPL(DisableTelemetryEvents, _In_ const OrtEnv* env);

ORT_API_STATUS_IMPL(CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names1, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output);

ORT_API_STATUS_IMPL(CreateSessionOptions, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(CloneSessionOptions, const OrtSessionOptions* input, OrtSessionOptions** out);
ORT_API_STATUS_IMPL(SetSessionExecutionMode, _In_ OrtSessionOptions* options, ExecutionMode execution_mode);
ORT_API_STATUS_IMPL(SetOptimizedModelFilePath, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* optimized_model_filepath);
ORT_API_STATUS_IMPL(EnableProfiling, _In_ OrtSessionOptions* options, _In_ const ORTCHAR_T* profile_file_prefix);
ORT_API_STATUS_IMPL(DisableProfiling, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableMemPattern, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(EnableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(DisableCpuMemArena, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(SetSessionLogId, _In_ OrtSessionOptions* options, const char* logid);
ORT_API_STATUS_IMPL(SetSessionLogVerbosityLevel, _In_ OrtSessionOptions* options, int session_log_verbosity_level);
ORT_API_STATUS_IMPL(SetSessionLogSeverityLevel, _In_ OrtSessionOptions* options, int session_log_severity_level);
ORT_API_STATUS_IMPL(SetSessionGraphOptimizationLevel, _In_ OrtSessionOptions* options,
                    GraphOptimizationLevel graph_optimization_level);
ORT_API_STATUS_IMPL(SetIntraOpNumThreads, _Inout_ OrtSessionOptions* options, int intra_op_num_threads);
ORT_API_STATUS_IMPL(SetInterOpNumThreads, _Inout_ OrtSessionOptions* options, int inter_op_num_threads);

ORT_API_STATUS_IMPL(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);
ORT_API_STATUS_IMPL(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);
ORT_API_STATUS_IMPL(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);
ORT_API_STATUS_IMPL(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, _Outptr_ void** library_handle);

ORT_API_STATUS_IMPL(SessionGetInputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOutputCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerCount, _In_ const OrtSession* sess, _Out_ size_t* out);
ORT_API_STATUS_IMPL(SessionGetInputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOutputTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* sess, size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(SessionGetInputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOutputName, _In_ const OrtSession* sess, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionGetOverridableInitializerName, _In_ const OrtSession* sess, size_t index,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(SessionEndProfiling, _In_ OrtSession* sess, _Inout_ OrtAllocator* allocator,
                    _Outptr_ char** out);
ORT_API_STATUS_IMPL(SessionGetModelMetadata, _In_ const OrtSession* sess,
                    _Outptr_ OrtModelMetadata** out);

ORT_API_STATUS_IMPL(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataGetGraphDescription, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** value);
ORT_API_STATUS_IMPL(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

ORT_API_STATUS_IMPL(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata,
                    _Out_ int64_t* value);

ORT_API_STATUS_IMPL(CreateRunOptions, _Outptr_ OrtRunOptions** out);

ORT_API_STATUS_IMPL(RunOptionsSetRunLogVerbosityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunLogSeverityLevel, _Inout_ OrtRunOptions* options, int value);
ORT_API_STATUS_IMPL(RunOptionsSetRunTag, _Inout_ OrtRunOptions*, _In_ const char* run_tag);

ORT_API_STATUS_IMPL(RunOptionsGetRunLogVerbosityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunLogSeverityLevel, _In_ const OrtRunOptions* options, _Out_ int* out);
ORT_API_STATUS_IMPL(RunOptionsGetRunTag, _In_ const OrtRunOptions*, _Out_ const char** out);

ORT_API_STATUS_IMPL(RunOptionsSetTerminate, _Inout_ OrtRunOptions* options);
ORT_API_STATUS_IMPL(RunOptionsUnsetTerminate, _Inout_ OrtRunOptions* options);

ORT_API_STATUS_IMPL(CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(IsTensor, _In_ const OrtValue* value, _Out_ int* out);
ORT_API_STATUS_IMPL(HasValue, _In_ const OrtValue* value, _Out_ int* out);
ORT_API_STATUS_IMPL(GetTensorMutableData, _Inout_ OrtValue* value, _Outptr_ void** out);
ORT_API_STATUS_IMPL(FillStringTensor, _Inout_ OrtValue* value, _In_ const char* const* s, size_t s_len);
ORT_API_STATUS_IMPL(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);
ORT_API_STATUS_IMPL(GetStringTensorDataLength, _In_ const OrtValue* value, _Out_ size_t* len);
ORT_API_STATUS_IMPL(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetStringTensorContent, _In_ const OrtValue* value, _Out_writes_bytes_all_(s_len) void* s,
                    size_t s_len, _Out_writes_all_(offsets_len) size_t* offsets, size_t offsets_len);
ORT_API_STATUS_IMPL(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);
ORT_API_STATUS_IMPL(CastTypeInfoToTensorInfo, _In_ const OrtTypeInfo*,
                    _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetOnnxTypeFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(CreateTensorTypeAndShapeInfo, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(SetTensorElementType, _Inout_ OrtTensorTypeAndShapeInfo*, enum ONNXTensorElementDataType type);
ORT_API_STATUS_IMPL(SetDimensions, OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);
ORT_API_STATUS_IMPL(GetTensorElementType, _In_ const OrtTensorTypeAndShapeInfo*, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetDimensionsCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetDimensions, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
ORT_API_STATUS_IMPL(GetSymbolicDimensions, _In_ const OrtTensorTypeAndShapeInfo* info,
                    _Out_writes_all_(dim_params_length) const char* dim_params[], size_t dim_params_length);
ORT_API_STATUS_IMPL(GetTensorShapeElementCount, _In_ const OrtTensorTypeAndShapeInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(GetTensorTypeAndShape, _In_ const OrtValue* value, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetTypeInfo, _In_ const OrtValue* value, _Outptr_result_maybenull_ OrtTypeInfo** out);
ORT_API_STATUS_IMPL(GetValueType, _In_ const OrtValue* value, _Out_ enum ONNXType* out);
ORT_API_STATUS_IMPL(AddFreeDimensionOverride, _Inout_ OrtSessionOptions* options, _In_ const char* dim_denotation, _In_ int64_t dim_value);

ORT_API_STATUS_IMPL(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CreateCpuMemoryInfo, enum OrtAllocatorType type, enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(CompareMemoryInfo, _In_ const OrtMemoryInfo* info1, _In_ const OrtMemoryInfo* info2, _Out_ int* out)
ORT_ALL_ARGS_NONNULL;
ORT_API_STATUS_IMPL(MemoryInfoGetName, _In_ const OrtMemoryInfo* ptr, _Out_ const char** out);
ORT_API_STATUS_IMPL(MemoryInfoGetId, _In_ const OrtMemoryInfo* ptr, _Out_ int* out);
ORT_API_STATUS_IMPL(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
ORT_API_STATUS_IMPL(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

ORT_API_STATUS_IMPL(AllocatorAlloc, _Inout_ OrtAllocator* ptr, size_t size, _Outptr_ void** out);
ORT_API_STATUS_IMPL(AllocatorFree, _Inout_ OrtAllocator* ptr, void* p);
ORT_API_STATUS_IMPL(AllocatorGetInfo, _In_ const OrtAllocator* ptr, _Outptr_ const struct OrtMemoryInfo** out);
ORT_API_STATUS_IMPL(GetAllocatorWithDefaultOptions, _Outptr_ OrtAllocator** out);
ORT_API_STATUS_IMPL(GetValue, _In_ const OrtValue* value, int index, _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetValueCount, _In_ const OrtValue* value, _Out_ size_t* out);
ORT_API_STATUS_IMPL(CreateValue, _In_reads_(num_values) const OrtValue* const* in, size_t num_values,
                    enum ONNXType value_type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(CreateOpaqueValue, _In_z_ const char* domain_name, _In_z_ const char* type_name,
                    _In_ const void* data_container, size_t data_container_size, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(GetOpaqueValue, _In_ const char* domain_name, _In_ const char* type_name,
                    _In_ const OrtValue* in, _Out_ void* data_container, size_t data_container_size);

ORT_API_STATUS_IMPL(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);

ORT_API_STATUS_IMPL(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);
ORT_API_STATUS_IMPL(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Out_ OrtValue** out);

// OrtTypeInfo methods
ORT_API_STATUS_IMPL(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation, _Out_ size_t* len);
ORT_API_STATUS_IMPL(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtMapTypeInfo** out);
ORT_API_STATUS_IMPL(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out);

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

// OrtSequenceTypeInfo Accessors
ORT_API_STATUS_IMPL(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);

ORT_API_STATUS_IMPL(DisablePerSessionThreads, _In_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);
ORT_API(void, ReleaseThreadingOptions, _Frees_ptr_opt_ OrtThreadingOptions*);

ORT_API_STATUS_IMPL(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata,
                    _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);

ORT_API_STATUS_IMPL(AddFreeDimensionOverrideByName, _Inout_ OrtSessionOptions* options, _In_ const char* dim_name, _In_ int64_t dim_value);

ORT_API_STATUS_IMPL(CreateAllocator, const OrtSession* sess, const OrtMemoryInfo* mem_info,
                    _Outptr_ OrtAllocator** out);
ORT_API(void, ReleaseAllocator, _Frees_ptr_opt_ OrtAllocator* allocator);

ORT_API_STATUS_IMPL(RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options, _In_ const OrtIoBinding* binding_ptr);

ORT_API_STATUS_IMPL(CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out);
ORT_API(void, ReleaseIoBinding, _Frees_ptr_opt_ OrtIoBinding* allocator);

ORT_API_STATUS_IMPL(BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);
ORT_API_STATUS_IMPL(BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);
ORT_API_STATUS_IMPL(BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* val_ptr);
ORT_API_STATUS_IMPL(GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Out_ char** buffer, _Outptr_result_maybenull_ size_t** lengths, _Out_ size_t* count);
ORT_API_STATUS_IMPL(GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Outptr_result_maybenull_ OrtValue*** output, _Out_ size_t* output_count);

ORT_API(void, ClearBoundInputs, _Inout_ OrtIoBinding* binding_ptr);
ORT_API(void, ClearBoundOutputs, _Inout_ OrtIoBinding* binding_ptr);

ORT_API_STATUS_IMPL(GetAvailableProviders, _Outptr_ char*** out_ptr,
                    _In_ int* providers_length);
ORT_API_STATUS_IMPL(ReleaseAvailableProviders, _In_ char** ptr,
                    _In_ int providers_length);

ORT_API_STATUS_IMPL(AddSessionConfigEntry, _Inout_ OrtSessionOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value);

ORT_API_STATUS_IMPL(TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count, _Outptr_ void** out);

ORT_API_STATUS_IMPL(CreateAndRegisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg);

ORT_API_STATUS_IMPL(SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection);
ORT_API_STATUS_IMPL(SessionGetProfilingStartTimeNs, _In_ const OrtSession* sess, _Out_ uint64_t* out);

ORT_API_STATUS_IMPL(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads);
ORT_API_STATUS_IMPL(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads);
ORT_API_STATUS_IMPL(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning);
ORT_API_STATUS_IMPL(AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name,
                    _In_ const OrtValue* val);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_CUDA,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_ROCM,
                    _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options);
ORT_API_STATUS_IMPL(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* options);

ORT_API_STATUS_IMPL(CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes,
                    int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out);
ORT_API(void, ReleaseArenaCfg, _Frees_ptr_opt_ OrtArenaCfg*);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_TensorRT,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_MIGraphX,
                    _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options);
ORT_API_STATUS_IMPL(SetCurrentGpuDeviceId, _In_ int device_id);
ORT_API_STATUS_IMPL(GetCurrentGpuDeviceId, _In_ int* device_id);
ORT_API_STATUS_IMPL(KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out, _Inout_ size_t* size);
ORT_API_STATUS_IMPL(KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out, _Inout_ size_t* size);
ORT_API_STATUS_IMPL(CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys, _In_reads_(num_keys) const size_t* arena_config_values,
                    _In_ size_t num_keys, _Outptr_ OrtArenaCfg** out);
ORT_API_STATUS_IMPL(AddRunConfigEntry, _Inout_ OrtRunOptions* options,
                    _In_z_ const char* config_key, _In_z_ const char* config_value);
ORT_API_STATUS_IMPL(CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out);
ORT_API(void, ReleasePrepackedWeightsContainer, _Frees_ptr_opt_ OrtPrepackedWeightsContainer*);
ORT_API_STATUS_IMPL(CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out);
ORT_API_STATUS_IMPL(CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container,
                    _Outptr_ OrtSession** out);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_TensorRT_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options);
ORT_API_STATUS_IMPL(CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out);
ORT_API_STATUS_IMPL(UpdateTensorRTProviderOptions, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys);
ORT_API_STATUS_IMPL(GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
ORT_API(void, ReleaseTensorRTProviderOptions, _Frees_ptr_opt_ OrtTensorRTProviderOptionsV2*);
ORT_API_STATUS_IMPL(EnableOrtCustomOps, _Inout_ OrtSessionOptions* options);
ORT_API_STATUS_IMPL(RegisterAllocator, _Inout_ OrtEnv* env, _In_ OrtAllocator* allocator);
ORT_API_STATUS_IMPL(UnregisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info);
// SparseTensor related API
ORT_API_STATUS_IMPL(IsSparseTensor, _In_ const OrtValue* value, _Out_ int* out);
ORT_API_STATUS_IMPL(CreateSparseTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* dense_shape,
                    size_t dense_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(FillSparseTensorCoo, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_data, size_t indices_num);
ORT_API_STATUS_IMPL(FillSparseTensorCsr, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* inner_indices_data, size_t inner_indices_num,
                    _In_ const int64_t* outer_indices_data, size_t outer_indices_num);
ORT_API_STATUS_IMPL(FillSparseTensorBlockSparse, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_shape_data, size_t indices_shape_len,
                    _In_ const int32_t* indices_data);
ORT_API_STATUS_IMPL(CreateSparseTensorWithValuesAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                    _In_ const int64_t* dense_shape, size_t dense_shape_len,
                    _In_ const int64_t* values_shape, size_t values_shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out);
ORT_API_STATUS_IMPL(UseCooIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* indices_data, size_t indices_num);
ORT_API_STATUS_IMPL(UseCsrIndices, _Inout_ OrtValue*, _Inout_ int64_t* inner_data, size_t inner_num, _Inout_ int64_t* outer_data, size_t outer_num);
ORT_API_STATUS_IMPL(UseBlockSparseIndices, _Inout_ OrtValue* ort_value, const int64_t* indices_shape, size_t indices_shape_len, _Inout_ int32_t* indices_data);
ORT_API_STATUS_IMPL(GetSparseTensorFormat, _In_ const OrtValue* ort_value, _Out_ enum OrtSparseFormat* out);
ORT_API_STATUS_IMPL(GetSparseTensorValuesTypeAndShape, _In_ const OrtValue* ort_value, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetSparseTensorValues, _In_ const OrtValue* ort_value, _Outptr_ const void** out);
ORT_API_STATUS_IMPL(GetSparseTensorIndicesTypeShape, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Outptr_ OrtTensorTypeAndShapeInfo** out);
ORT_API_STATUS_IMPL(GetSparseTensorIndices, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Out_ size_t* num_indices, _Outptr_ const void** indices);
ORT_API_STATUS_IMPL(KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context, _Outptr_ void** out);
ORT_API_STATUS_IMPL(GetTensorMemoryInfo, _In_ const OrtValue* value, _Outptr_ const OrtMemoryInfo** memory_info);
ORT_API_STATUS_IMPL(GetExecutionProviderApi, _In_ const char* provider_name, _In_ uint32_t version, _Outptr_ const void** provider_api);
ORT_API_STATUS_IMPL(SessionOptionsSetCustomCreateThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn);
ORT_API_STATUS_IMPL(SessionOptionsSetCustomThreadCreationOptions, _Inout_ OrtSessionOptions* options, _In_ void* ort_custom_thread_creation_options);
ORT_API_STATUS_IMPL(SessionOptionsSetCustomJoinThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn);
ORT_API_STATUS_IMPL(SetGlobalCustomCreateThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn);
ORT_API_STATUS_IMPL(SetGlobalCustomThreadCreationOptions, _Inout_ OrtThreadingOptions* tp_options, _In_ void* ort_custom_thread_creation_options);
ORT_API_STATUS_IMPL(SetGlobalCustomJoinThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn);
ORT_API_STATUS_IMPL(SynchronizeBoundInputs, _Inout_ OrtIoBinding* binding_ptr);
ORT_API_STATUS_IMPL(SynchronizeBoundOutputs, _Inout_ OrtIoBinding* binding_ptr);
ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_CUDA_V2,
                    _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptionsV2* cuda_options);
ORT_API_STATUS_IMPL(CreateCUDAProviderOptions, _Outptr_ OrtCUDAProviderOptionsV2** out);
ORT_API_STATUS_IMPL(UpdateCUDAProviderOptions, _Inout_ OrtCUDAProviderOptionsV2* cuda_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys);
ORT_API_STATUS_IMPL(GetCUDAProviderOptionsAsString, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
ORT_API(void, ReleaseCUDAProviderOptions, _Frees_ptr_opt_ OrtCUDAProviderOptionsV2*);

ORT_API_STATUS_IMPL(AddExternalInitializers, _In_ OrtSessionOptions* options,
                    _In_reads_(initializers_num) const char* const* initializer_names,
                    _In_reads_(initializers_num) const OrtValue* const* initializers, size_t initializers_num);

ORT_API_STATUS_IMPL(CreateOpAttr,
                    _In_ const char* name,
                    _In_ const void* data,
                    _In_ int len,
                    _In_ OrtOpAttrType type,
                    _Outptr_ OrtOpAttr** op_attr);

ORT_API(void, ReleaseOpAttr, _Frees_ptr_opt_ OrtOpAttr* op_attr);

ORT_API_STATUS_IMPL(CreateOp,
                    _In_ const OrtKernelInfo* info,
                    _In_z_ const char* op_name,
                    _In_z_ const char* domain,
                    int version,
                    _In_reads_(type_constraint_count) const char** type_constraint_names,
                    _In_reads_(type_constraint_count) const ONNXTensorElementDataType* type_constraint_values,
                    int type_constraint_count,
                    _In_reads_(attr_count) const OrtOpAttr* const* attr_values,
                    int attr_count,
                    int input_count,
                    int output_count,
                    _Outptr_ OrtOp** ort_op);

ORT_API_STATUS_IMPL(InvokeOp,
                    _In_ const OrtKernelContext* context,
                    _In_ const OrtOp* ort_op,
                    _In_ const OrtValue* const* input_values,
                    _In_ int input_count,
                    _Inout_ OrtValue* const* output_values,
                    _In_ int output_count);

ORT_API(void, ReleaseOp, _Frees_ptr_opt_ OrtOp* op);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider,
                    _In_ OrtSessionOptions* options,
                    _In_ const char* provider_name,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    _In_ size_t num_keys);

ORT_API_STATUS_IMPL(CopyKernelInfo, _In_ const OrtKernelInfo* info, _Outptr_ OrtKernelInfo** info_copy);

ORT_API(void, ReleaseKernelInfo, _Frees_ptr_opt_ OrtKernelInfo* info_copy);

ORT_API(const OrtTrainingApi*, GetTrainingApi, uint32_t version);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_CANN,
                    _In_ OrtSessionOptions* options, _In_ const OrtCANNProviderOptions* cann_options);
ORT_API_STATUS_IMPL(CreateCANNProviderOptions, _Outptr_ OrtCANNProviderOptions** out);
ORT_API_STATUS_IMPL(UpdateCANNProviderOptions, _Inout_ OrtCANNProviderOptions* cann_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys);
ORT_API_STATUS_IMPL(GetCANNProviderOptionsAsString, _In_ const OrtCANNProviderOptions* cann_options,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
ORT_API(void, ReleaseCANNProviderOptions, _Frees_ptr_opt_ OrtCANNProviderOptions*);

ORT_API(void, MemoryInfoGetDeviceType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemoryInfoDeviceType* out);

ORT_API_STATUS_IMPL(UpdateEnvWithCustomLogLevel, _In_ OrtEnv* ort_env, OrtLoggingLevel log_severity_level);

ORT_API_STATUS_IMPL(SetGlobalIntraOpThreadAffinity, _Inout_ OrtThreadingOptions* tp_options,
                    const char* affinity_string);

ORT_API_STATUS_IMPL(RegisterCustomOpsLibrary_V2, _Inout_ OrtSessionOptions* options,
                    _In_ const ORTCHAR_T* library_name);
ORT_API_STATUS_IMPL(RegisterCustomOpsUsingFunction, _Inout_ OrtSessionOptions* options,
                    _In_ const char* registration_func_name);

ORT_API_STATUS_IMPL(KernelInfo_GetInputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelInfo_GetOutputCount, _In_ const OrtKernelInfo* info, _Out_ size_t* out);
ORT_API_STATUS_IMPL(KernelInfo_GetInputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out,
                    _Inout_ size_t* size);
ORT_API_STATUS_IMPL(KernelInfo_GetOutputName, _In_ const OrtKernelInfo* info, size_t index, _Out_ char* out,
                    _Inout_ size_t* size);
ORT_API_STATUS_IMPL(KernelInfo_GetInputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(KernelInfo_GetOutputTypeInfo, _In_ const OrtKernelInfo* info, size_t index,
                    _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(KernelInfoGetAttribute_tensor, _In_ const OrtKernelInfo* info, _In_z_ const char* name,
                    _Inout_ OrtAllocator* allocator, _Outptr_ OrtValue** out);

ORT_API_STATUS_IMPL(HasSessionConfigEntry, _In_ const OrtSessionOptions* options,
                    _In_z_ const char* config_key, _Out_ int* out);
ORT_API_STATUS_IMPL(GetSessionConfigEntry, _In_ const OrtSessionOptions* options,
                    _In_z_ const char* config_key, _Out_ char* config_value, _Inout_ size_t* size);

ORT_API_STATUS_IMPL(SessionOptionsAppendExecutionProvider_Dnnl,
                    _In_ OrtSessionOptions* options, _In_ const OrtDnnlProviderOptions* dnnl_options);
ORT_API_STATUS_IMPL(CreateDnnlProviderOptions, _Outptr_ OrtDnnlProviderOptions** out);
ORT_API_STATUS_IMPL(UpdateDnnlProviderOptions, _Inout_ OrtDnnlProviderOptions* dnnl_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys);
ORT_API_STATUS_IMPL(GetDnnlProviderOptionsAsString, _In_ const OrtDnnlProviderOptions* dnnl_options,
                    _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
ORT_API(void, ReleaseDnnlProviderOptions, _Frees_ptr_opt_ OrtDnnlProviderOptions*);

ORT_API_STATUS_IMPL(KernelInfo_GetNodeName, _In_ const OrtKernelInfo* info, _Out_ char* out, _Inout_ size_t* size);
ORT_API_STATUS_IMPL(KernelInfo_GetLogger, _In_ const OrtKernelInfo* info, _Outptr_ const OrtLogger** logger);
ORT_API_STATUS_IMPL(KernelContext_GetLogger, _In_ const OrtKernelContext* context, _Outptr_ const OrtLogger** logger);

ORT_API_STATUS_IMPL(Logger_LogMessage, _In_ const OrtLogger* logger, OrtLoggingLevel log_severity_level,
                    _In_z_ const char* message, _In_z_ const ORTCHAR_T* file_path, int line_number,
                    _In_z_ const char* func_name);
ORT_API_STATUS_IMPL(Logger_GetLoggingSeverityLevel, _In_ const OrtLogger* logger, _Out_ OrtLoggingLevel* out);

ORT_API_STATUS_IMPL(KernelInfoGetConstantInput_tensor, _In_ const OrtKernelInfo* info, _In_ size_t index,
                    _Out_ int* is_constant, _Outptr_ const OrtValue** out);

ORT_API_STATUS_IMPL(CastTypeInfoToOptionalTypeInfo, _In_ const OrtTypeInfo* type_info,
                    _Outptr_result_maybenull_ const OrtOptionalTypeInfo** out);

ORT_API_STATUS_IMPL(GetOptionalContainedTypeInfo, _In_ const OrtOptionalTypeInfo* optional_type_info,
                    _Outptr_ OrtTypeInfo** out);

ORT_API_STATUS_IMPL(GetResizedStringTensorElementBuffer, _Inout_ OrtValue* value,
                    _In_ size_t index, _In_ size_t length_in_bytes, _Inout_ char**);

ORT_API_STATUS_IMPL(KernelContext_GetAllocator, _In_ const OrtKernelContext* context, _In_ const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out);

ORT_API(const char*, GetBuildInfoString);

ORT_API_STATUS_IMPL(CreateROCMProviderOptions, _Outptr_ OrtROCMProviderOptions** out);
ORT_API_STATUS_IMPL(UpdateROCMProviderOptions, _Inout_ OrtROCMProviderOptions* rocm_options,
                    _In_reads_(num_keys) const char* const* provider_options_keys,
                    _In_reads_(num_keys) const char* const* provider_options_values,
                    size_t num_keys);
ORT_API_STATUS_IMPL(GetROCMProviderOptionsAsString, _In_ const OrtROCMProviderOptions* rocm_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);
ORT_API(void, ReleaseROCMProviderOptions, _Frees_ptr_opt_ OrtROCMProviderOptions*);

ORT_API_STATUS_IMPL(CreateAndRegisterAllocatorV2, _Inout_ OrtEnv* env, _In_ const char* provider_type, _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg,
                    _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);

ORT_API_STATUS_IMPL(RunAsync, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** outputs,
                    _In_ RunAsyncCallbackFn run_async_callback, _In_opt_ void* user_data);

ORT_API_STATUS_IMPL(UpdateTensorRTProviderOptionsWithValue, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options, _In_ const char* key, _In_ void* value);
ORT_API_STATUS_IMPL(GetTensorRTProviderOptionsByName, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _In_ const char* key, _Outptr_ void** ptr);
ORT_API_STATUS_IMPL(UpdateCUDAProviderOptionsWithValue, _Inout_ OrtCUDAProviderOptionsV2* cuda_options, _In_ const char* key, _In_ void* value);
ORT_API_STATUS_IMPL(GetCUDAProviderOptionsByName, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _In_ const char* key, _Outptr_ void** ptr);
ORT_API_STATUS_IMPL(KernelContext_GetResource, _In_ const OrtKernelContext* context, _In_ int resource_version, _In_ int resource_id, _Outptr_ void** stream);

ORT_API_STATUS_IMPL(SetUserLoggingFunction, _Inout_ OrtSessionOptions* options,
                    _In_ OrtLoggingFunction user_logging_function, _In_opt_ void* user_logging_param);
ORT_API_STATUS_IMPL(ShapeInferContext_GetInputCount, _In_ const OrtShapeInferContext* context, _Out_ size_t* out);
ORT_API_STATUS_IMPL(ShapeInferContext_GetInputTypeShape, _In_ const OrtShapeInferContext* context, _In_ size_t index, _Outptr_ OrtTensorTypeAndShapeInfo** info);
ORT_API_STATUS_IMPL(ShapeInferContext_GetAttribute, _In_ const OrtShapeInferContext* context, _In_ const char* attr_name, _Outptr_ const OrtOpAttr** attr);
ORT_API_STATUS_IMPL(ShapeInferContext_SetOutputTypeShape, _In_ const OrtShapeInferContext* context, _In_ size_t index, _In_ const OrtTensorTypeAndShapeInfo* info);
ORT_API_STATUS_IMPL(SetSymbolicDimensions, _In_ OrtTensorTypeAndShapeInfo* info, _In_ const char* dim_params[], _In_ size_t dim_params_length);
ORT_API_STATUS_IMPL(ReadOpAttr, _In_ const OrtOpAttr* op_attr, _In_ OrtOpAttrType type, _Inout_ void* data, _In_ size_t len, _Out_ size_t* out);

ORT_API_STATUS_IMPL(KernelContext_ParallelFor, _In_ const OrtKernelContext* context, _In_ void (*fn)(void*, size_t), _In_ size_t total, _In_ size_t num_batch, _In_ void* user_data);

}  // namespace OrtApis
