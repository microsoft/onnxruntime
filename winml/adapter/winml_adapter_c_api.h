// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

ORT_RUNTIME_CLASS(Model);
ORT_RUNTIME_CLASS(MapTypeInfo);
ORT_RUNTIME_CLASS(SequenceTypeInfo);
ORT_RUNTIME_CLASS(ExecutionProvider);

struct WinmlAdapterApi;
typedef struct WinmlAdapterApi WinmlAdapterApi;

struct ID3D12Resource;
struct ID3D12Device;
struct ID3D12CommandQueue;
struct IMLOperatorRegistry;

ORT_EXPORT const WinmlAdapterApi* ORT_API_CALL GetWinmlAdapterApi(_In_ const OrtApi* ort_api) NO_EXCEPTION;

// TODO: Must match onnxruntime::profiling::EventRecord
enum OrtProfilerEventCategory {
  SESSION_EVENT = 0,
  NODE_EVENT,
  EVENT_CATEGORY_MAX
};

struct OrtProfilerEventRecord {
  OrtProfilerEventCategory category_;
  const char* category_name_;
  int64_t duration_;
  int64_t time_span_;
  const char* event_name_;
  int32_t process_id_;
  int32_t thread_id_;
  const char* op_name_;
  const char* execution_provider_;
};

typedef void(ORT_API_CALL* OrtProfilingFunction)(const OrtProfilerEventRecord* event_record);

struct WinmlAdapterApi {
  OrtStatus*(ORT_API_CALL* OverrideSchema)() NO_EXCEPTION;

  OrtStatus*(ORT_API_CALL* EnvConfigureCustomLoggerAndProfiler)(_In_ OrtEnv* env, OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out)NO_EXCEPTION;

  OrtStatus*(ORT_API_CALL* GetDenotationFromTypeInfo)(_In_ const OrtTypeInfo*, _Out_ const char** const denotation, _Out_ size_t* len)NO_EXCEPTION;

  // OrtTypeInfo Casting methods
  OrtStatus*(ORT_API_CALL* CastTypeInfoToMapTypeInfo)(_In_ const OrtTypeInfo* type_info, _Out_ const OrtMapTypeInfo** out)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* CastTypeInfoToSequenceTypeInfo)(_In_ const OrtTypeInfo* type_info, _Out_ const OrtSequenceTypeInfo** out)NO_EXCEPTION;

  // OrtMapTypeInfo Accessors
  OrtStatus*(ORT_API_CALL* GetMapKeyType)(_In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* GetMapValueType)(_In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;

  // OrtSequenceTypeInfo Accessors
  OrtStatus*(ORT_API_CALL* GetSequenceElementType)(_In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;

  // OrtModel methods
  OrtStatus*(ORT_API_CALL* CreateModelFromPath)(_In_ const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* CreateModelFromData)(_In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* CloneModel)(_In_ const OrtModel* in, _Outptr_ OrtModel** out)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetAuthor)(_In_ const OrtModel* model, _Out_ const char** const author, _Out_ size_t* len)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetName)(_In_ const OrtModel* model, _Out_ const char** const name, _Out_ size_t* len)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetDomain)(_In_ const OrtModel* model, _Out_ const char** const domain, _Out_ size_t* len)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetDescription)(_In_ const OrtModel* model, _Out_ const char** const description, _Out_ size_t* len)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetVersion)(_In_ const OrtModel* model, _Out_ int64_t* version)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetInputCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetOutputCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetInputName)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_name, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetOutputName)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_name, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetInputDescription)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_description, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetOutputDescription)(_In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_description, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetInputTypeInfo)(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetOutputTypeInfo)(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetMetadataCount)(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelGetMetadata)(_In_ const OrtModel* model, _Out_ size_t count, _Out_ const char** const key, _Out_ size_t* key_len, _Out_ const char** const value, _Out_ size_t* value_len)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ModelEnsureNoFloat16)(_In_ const OrtModel* model)NO_EXCEPTION;

  // OrtSessionOptions methods
  OrtStatus*(ORT_API_CALL* OrtSessionOptionsAppendExecutionProvider_CPU)(_In_ OrtSessionOptions* options, int use_arena)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* OrtSessionOptionsAppendExecutionProvider_DML)(_In_ OrtSessionOptions* options, ID3D12Device* device, ID3D12CommandQueue* queue)NO_EXCEPTION;
  
  // OrtSession methods
  OrtStatus*(ORT_API_CALL* CreateSessionWithoutModel)(_In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionGetExecutionProvidersCount)(_In_ OrtSession* session, _Out_ size_t* count)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionGetExecutionProvider)(_In_ OrtSession* session, _In_ size_t index, _Out_ const OrtExecutionProvider** provider)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionInitialize)(_In_ OrtSession* session)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionRegisterGraphTransformers)(_In_ OrtSession* session)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionRegisterCustomRegistry)(_In_ OrtSession* session, _In_ IMLOperatorRegistry* registry)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionLoadAndPurloinModel)(_In_ OrtSession* session, _In_ OrtModel* model)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionStartProfiling)(_In_ OrtEnv* env, _In_ OrtSession* session)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionEndProfiling)(_In_ OrtSession* session)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* SessionCopyOneInputAcrossDevices)(_In_ OrtSession* session, _In_ const char** const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;

  // Dml methods (TODO need to figure out how these need to move to session somehow...)

  OrtStatus*(ORT_API_CALL* DmlExecutionProviderSetDefaultRoundingMode)(_In_ bool is_enabled)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext)(_In_ const OrtExecutionProvider* dml_provider)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderTrimUploadHeap)(_In_ const OrtExecutionProvider* dml_provider)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences)(_In_ const OrtExecutionProvider* dml_provider)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlCreateGPUAllocationFromD3DResource)(_In_ ID3D12Resource* pResource, _Out_ void* dml_resource)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlFreeGPUAllocation)(_In_ void* ptr)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* DmlGetD3D12ResourceFromAllocation)(_In_ const OrtExecutionProvider* provider, _In_ void* allocation, _Out_ ID3D12Resource** resource)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* GetProviderMemoryInfo)(_In_ const OrtExecutionProvider* provider, OrtMemoryInfo** memory_info)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* GetProviderAllocator)(_In_ const OrtExecutionProvider* provider, OrtAllocator** allocator)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* FreeProviderAllocator)(_In_ OrtAllocator* allocator)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* GetValueMemoryInfo)(const OrtValue* value, OrtMemoryInfo** memory_info)NO_EXCEPTION;

  OrtStatus*(ORT_API_CALL* ExecutionProviderSync)(_In_ const OrtExecutionProvider* provider)NO_EXCEPTION;
  OrtStatus*(ORT_API_CALL* ExecutionProviderCopyTensor)(_In_ const OrtExecutionProvider* provider, _In_ OrtValue* src, _In_ OrtValue* dst)NO_EXCEPTION;
  ORT_CLASS_RELEASE(Model);
  ORT_CLASS_RELEASE(MapTypeInfo);
  ORT_CLASS_RELEASE(SequenceTypeInfo);
};
