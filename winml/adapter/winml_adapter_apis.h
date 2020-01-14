// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"

namespace Windows {
namespace AI {
namespace MachineLearning {
namespace Adapter {

ORT_API(void, ReleaseModel, OrtModel*);
ORT_API(void, ReleaseMapTypeInfo, OrtMapTypeInfo*);
ORT_API(void, ReleaseSequenceTypeInfo, OrtSequenceTypeInfo*);
ORT_API(void, ReleaseExecutionProvider, OrtExecutionProvider*);
ORT_API(void, ReleaseOperatorRegistry, OrtOperatorRegistry*);

// OrtEnv methods
ORT_API_STATUS(EnvConfigureCustomLoggerAndProfiler, _In_ OrtEnv* env, OrtLoggingFunction logging_function, OrtProfilingFunction profiling_function, _In_opt_ void* logger_param, OrtLoggingLevel default_warning_level, _In_ const char* logid, _Outptr_ OrtEnv** out);

// OrtTypeInfo methods
ORT_API_STATUS(GetDenotationFromTypeInfo, _In_ const OrtTypeInfo*, _Out_ const char** const denotation, _Out_ size_t* len);
ORT_API_STATUS(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const OrtMapTypeInfo** out);
ORT_API_STATUS(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const OrtSequenceTypeInfo** out);

// OrtMapTypeInfo Accessors
ORT_API_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

// OrtSequenceTypeInfo Accessors
ORT_API_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);

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

// OrtSession methods
ORT_API_STATUS(CreateSessionWithoutModel, _In_ OrtEnv* env, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session);

ORT_API_STATUS(SessionGetExecutionProvidersCount, _In_ OrtSession* session, _Out_ size_t* count);
ORT_API_STATUS(SessionGetExecutionProvider, _In_ OrtSession* session, size_t index, _Out_ OrtExecutionProvider** provider);
ORT_API_STATUS(SessionInitialize, _In_ OrtSession* session);
ORT_API_STATUS(SessionLoadAndPurloinModel, _In_ OrtSession* session, _In_ OrtModel * model);

/*
ORT_API_STATUS(SessionRegisterGraphTransformers, _In_ OrtSession* session);
ORT_API_STATUS(SessionRegisterCustomRegistry, _In_ OrtSession* session, _In_ OrtOperatorRegistry* registry);
ORT_API_STATUS(SessionStartProfiling, _In_ OrtSession* session);
ORT_API_STATUS(SessionEndProfiling, _In_ OrtSession* session);
ORT_API_STATUS(SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const char* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value);
    
// Dml methods (TODO need to figure out how these need to move to session somehow...)
ORT_API_STATUS(DmlExecutionProviderFlushContext, _In_ OrtExecutionProvider * dml_provider);
ORT_API_STATUS(DmlExecutionProviderTrimUploadHeap, _In_ OrtExecutionProvider* dml_provider);
ORT_API_STATUS(DmlExecutionProviderReleaseCompletedReferences, _In_ OrtExecutionProvider* dml_provider);
*/

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
               _In_ ID3D12Device* d3d_device, _In_ ID3D12CommandQueue* cmd_queue);

}  // namespace Adapter
}  // namespace MachineLearning
}  // namespace AI
}  // namespace Windows