// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"

namespace Windows {
namespace AI {
namespace MachineLearning {
namespace Adapter {

ORT_API(void, ReleaseModel, OrtModel*);
ORT_API(void, ReleaseExecutionProvider, OrtExecutionProvider*);

ORT_API_STATUS(OverrideSchema);

// OrtEnv methods
ORT_API_STATUS(EnvConfigureCustomLoggerAndProfiler, _In_ OrtEnv* env, OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

// OrtModel methods
ORT_API_STATUS(CreateModelFromPath, _In_ const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out);
ORT_API_STATUS(CreateModelFromData, _In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out);
ORT_API_STATUS(CloneModel, _In_ const OrtModel* in, _Outptr_ OrtModel** out);
ORT_API_STATUS(ModelGetAuthor, _In_ const OrtModel* model, _Out_ const char** const author, _Out_ size_t* len);
ORT_API_STATUS(ModelGetName, _In_ const OrtModel* model, _Out_ const char** const name, _Out_ size_t* len);
ORT_API_STATUS(ModelGetDomain, _In_ const OrtModel* model, _Out_ const char** const domain, _Out_ size_t* len);
ORT_API_STATUS(ModelGetDescription, _In_ const OrtModel* model, _Out_ const char** const description, _Out_ size_t* len);
ORT_API_STATUS(ModelGetVersion, _In_ const OrtModel* model, _Out_ int64_t* version);
ORT_API_STATUS(ModelGetInputCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS(ModelGetOutputCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS(ModelGetInputName, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_name, _Out_ size_t* count);
ORT_API_STATUS(ModelGetOutputName, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_name, _Out_ size_t* count);
ORT_API_STATUS(ModelGetInputDescription, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_description, _Out_ size_t* count);
ORT_API_STATUS(ModelGetOutputDescription, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_description, _Out_ size_t* count);
ORT_API_STATUS(ModelGetInputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS(ModelGetOutputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS(ModelGetMetadataCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS(ModelGetMetadata, _In_ const OrtModel* model, _Out_ size_t count, _Out_ const char** const key, _Out_ size_t* key_len, _Out_ const char** const value, _Out_ size_t* value_len);
ORT_API_STATUS(ModelEnsureNoFloat16, _In_ const OrtModel* model);

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options, _In_ ID3D12Device* d3d_device, _In_ ID3D12CommandQueue* cmd_queue, bool metacommands_enabled);

// OrtSession methods
ORT_API_STATUS(CreateSessionWithoutModel, _In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session);

//Do not release provider... as there is no release method available
ORT_API_STATUS(SessionGetExecutionProvider, _In_ OrtSession* session, size_t index, _Out_ OrtExecutionProvider** provider);
ORT_API_STATUS(SessionInitialize, _In_ OrtSession* session);
ORT_API_STATUS(SessionLoadAndPurloinModel, _In_ OrtSession* session, _In_ OrtModel* model);

ORT_API_STATUS(SessionStartProfiling, _In_ OrtEnv* env, _In_ OrtSession* session);
ORT_API_STATUS(SessionEndProfiling, _In_ OrtSession* session);
ORT_API_STATUS(SessionRegisterGraphTransformers, _In_ OrtSession* session);
ORT_API_STATUS(SessionRegisterCustomRegistry, _In_ OrtSession* session, _In_ IMLOperatorRegistry* registry);
ORT_API_STATUS(SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const char* const input_name, _In_ OrtValue* orig_value, _Outptr_ OrtValue** new_value);
ORT_API_STATUS(SessionGetNumberOfIntraOpThreads, _In_ OrtSession* session, _Out_ uint32_t* num_threads);
ORT_API_STATUS(SessionGetNamedDimensionsOverrides, _In_ OrtSession* session, _Out_ winrt::Windows::Foundation::Collections::IMapView<winrt::hstring, uint32_t>& overrides);

// Dml methods (TODO need to figure out how these need to move to session somehow...)
ORT_API_STATUS(DmlExecutionProviderSetDefaultRoundingMode, _In_ OrtExecutionProvider* dml_provider, _In_ bool is_enabled);
ORT_API_STATUS(DmlExecutionProviderFlushContext, _In_ OrtExecutionProvider* dml_provider);
ORT_API_STATUS(DmlExecutionProviderReleaseCompletedReferences, _In_ OrtExecutionProvider* dml_provider);
ORT_API_STATUS(DmlCreateGPUAllocationFromD3DResource, _In_ ID3D12Resource* pResource, _Out_ void** dml_resource);
ORT_API_STATUS(DmlGetD3D12ResourceFromAllocation, _In_ OrtExecutionProvider* provider, _In_ void* allocation, _Out_ ID3D12Resource** resource);
ORT_API_STATUS(DmlFreeGPUAllocation, _In_ void* ptr);

// note: this returns a weak ref

ORT_API_STATUS(GetProviderMemoryInfo, _In_ OrtExecutionProvider* provider, OrtMemoryInfo** memory_info);
ORT_API_STATUS(GetProviderAllocator, _In_ OrtExecutionProvider* provider, OrtAllocator** allocator);
ORT_API_STATUS(FreeProviderAllocator, _In_ OrtAllocator* allocator);
ORT_API_STATUS(GetValueMemoryInfo, const OrtValue* value, OrtMemoryInfo** memory_info);

// ExecutionProvider Methods
ORT_API_STATUS(ExecutionProviderSync, _In_ OrtExecutionProvider* provider);
ORT_API_STATUS(DmlCopyTensor, _In_ OrtExecutionProvider* provider, _In_ OrtValue* src, _In_ OrtValue* dst);
ORT_API_STATUS(CreateCustomRegistry, _Out_ IMLOperatorRegistry** registry);

ORT_API_STATUS(ValueGetDeviceId, _In_ OrtValue* ort_value, _Out_ int16_t* device_id);
ORT_API_STATUS(SessionGetInputRequiredDeviceId, _In_ OrtSession* session, _In_ const char* const input_name, _Out_ int16_t* device_id);

}  // namespace Adapter
}  // namespace MachineLearning
}  // namespace AI
}  // namespace Windows