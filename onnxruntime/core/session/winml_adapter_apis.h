// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"

namespace Windows { namespace AI { namespace MachineLearning { namespace Adapter {

ORT_API(void, ReleaseModel, OrtModel*);
ORT_API(void, ReleaseMapTypeInfo, OrtMapTypeInfo*);
ORT_API(void, ReleaseSequenceTypeInfo, OrtSequenceTypeInfo*);
ORT_API(void, ReleaseExecutionProvider, OrtExecutionProvider*);
ORT_API(void, ReleaseOperatorRegistry, OrtOperatorRegistry*);

// OrtTypeInfo Casting methods
ORT_API_STATUS_IMPL(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const OrtMapTypeInfo** out);
ORT_API_STATUS_IMPL(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo* type_info, _Out_ const OrtSequenceTypeInfo** out);

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(GetMapKeyType, _In_ OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);
ORT_API_STATUS_IMPL(GetMapValueType, _In_ OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

// OrtSequenceTypeInfo Accessors
ORT_API_STATUS_IMPL(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);

// OrtModel methods
ORT_API_STATUS_IMPL(CreateModelFromPath, _In_ const char* model_path, _In_ size_t size, _Outptr_ OrtModel** out);
ORT_API_STATUS_IMPL(CreateModelFromData, _In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out);


//ORT_API_STATUS_IMPL(CloneModel, _In_ const OrtModel* in, _Outptr_ OrtModel** out);
ORT_API_STATUS_IMPL(ModelGetAuthor, _In_ const OrtModel* model, _Out_ const char** const author, _Out_ size_t* len);
ORT_API_STATUS_IMPL(ModelGetName, _In_ const OrtModel* model, _Out_ const char** const name, _Out_ size_t* len);
ORT_API_STATUS_IMPL(ModelGetDomain, _In_ const OrtModel* model, _Out_ const char** const domain, _Out_ size_t* len);
ORT_API_STATUS_IMPL(ModelGetDescription, _In_ const OrtModel* model, _Out_ const char** const description, _Out_ size_t* len);
ORT_API_STATUS_IMPL(ModelGetVersion, _In_ const OrtModel* model, _Out_ int64_t* version);
ORT_API_STATUS_IMPL(ModelGetInputCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetOutputCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetInputName, _In_ const OrtModel* model, _In_ size_t index,_Out_ const char** input_name, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetOutputName, _In_ const OrtModel* model, _In_ size_t index,_Out_ const char** output_name, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetInputDescription, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** input_description, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetOutputDescription, _In_ const OrtModel* model, _In_ size_t index, _Out_ const char** output_description, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetInputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(ModelGetOutputTypeInfo, _In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(ModelGetMetadataCount, _In_ const OrtModel* model, _Out_ size_t* count);
ORT_API_STATUS_IMPL(ModelGetMetadata, _In_ const OrtModel* model, _Out_ size_t count, _Out_ const char** const key, _Out_ size_t* key_len, _Out_ const char** const value, _Out_ size_t* value_len);
/*
ORT_API_STATUS_IMPL(ModelCheckIfValid, _In_ OrtModel* model, _Out_ bool* valid);

// OrtSession methods
ORT_API_STATUS_IMPL(CreateSessionWihtoutModel, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session);
ORT_API_STATUS_IMPL(SessionRegisterExecutionProvider, _In_ OrtSession* session, _In_ OrtExecutionProvider* provider);
ORT_API_STATUS_IMPL(SessionInitialize, _In_ OrtSession* session, _In_ OrtExecutionProvider* provider);
ORT_API_STATUS_IMPL(SessionRegisterGraphTransformers, _In_ OrtSession* session);
ORT_API_STATUS_IMPL(SessionRegisterCustomRegistry, _In_ OrtSession* session, _In_ OrtOperatorRegistry* registry);
ORT_API_STATUS_IMPL(SessionLoadModel, _In_ OrtSession* session, _In_ OrtModel * model);
ORT_API_STATUS_IMPL(SessionStartProfiling, _In_ OrtSession* session);
ORT_API_STATUS_IMPL(SessionEndProfiling, _In_ OrtSession* session);
ORT_API_STATUS_IMPL(SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const char* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value);
    
// Dml methods (TODO need to figure out how these need to move to session somehow...)
ORT_API_STATUS_IMPL(DmlExecutionProviderFlushContext, _In_ OrtExecutionProvider * dml_provider);
ORT_API_STATUS_IMPL(DmlExecutionProviderTrimUploadHeap, _In_ OrtExecutionProvider* dml_provider);
ORT_API_STATUS_IMPL(DmlExecutionProviderReleaseCompletedReferences, _In_ OrtExecutionProvider* dml_provider);
*/
}}}} // namespace Windows::AI::MachineLearning::Adapter