// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "winml_adapter_c_api.h"

namespace Windows::AI::MachineLearning::Adapter {

ORT_API(void, ReleaseModel, OrtModel*);
ORT_API(void, ReleaseMapTypeInfo, OrtMapTypeInfo*);
ORT_API(void, ReleaseSequenceTypeInfo, OrtSequenceTypeInfo*);

// OrtTypeInfo Casting methods
ORT_API_STATUS_IMPL(CastTypeInfoToMapTypeInfo, _In_ const OrtTypeInfo*, _Out_ const OrtMapTypeInfo** out);
ORT_API_STATUS_IMPL(CastTypeInfoToSequenceTypeInfo, _In_ const OrtTypeInfo*, _Out_ const OrtSequenceTypeInfo** out)NO_EXCEPTION;

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(GetMapKeyType, _In_ OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out)NO_EXCEPTION;
ORT_API_STATUS_IMPL(GetMapValueType, _In_ OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;

// OrtSequenceTypeInfo Accessors
ORT_API_STATUS_IMPL(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
/*
// OrtModel methods
ORT_API_STATUS_IMPL(CreateModel, _In_ const ORTCHAR_T* model_path, _Outptr_ OrtModel** out)NO_EXCEPTION;
ORT_API_STATUS_IMPL(CreateModel, _In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;
ORT_API_STATUS_IMPL(CreateModel, _In_ const OrtModel* in, _Outptr_ OrtModel** out)NO_EXCEPTION;
ORT_API_STATUS_IMPL(CreateModel, ORTCHAR_T* model_path, _Outptr_ OrtModel** out)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetAuthor, In_ const OrtModel* model, _Out_ const ORTCHAR_T* const author, _Out_ size_t* len)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetName, In_ const OrtModel* model, _Out_ const ORTCHAR_T* const name, _Out_ size_t* len)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetDomain, In_ const OrtModel* model, _Out_ const ORTCHAR_T* const domain, _Out_ size_t* len)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetDescription, In_ const OrtModel* model, _Out_ const ORTCHAR_T* const description, _Out_ size_t* len)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetVersion, In_ const OrtModel* model, _Out_ int64_t* version)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetInputCount, In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetOutputCount, In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetInputName, In_ const OrtModel* model, _Out_ const char** input_name, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetOutputName, In_ const OrtModel* model, _Out_ const char** output_name, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetInputDescription, In_ const OrtModel* model, _Out_ const char** input_description, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetOutputDescription, In_ const OrtModel* model, _Out_ const char** output_description, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetInputTypeInfo, In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetOutputTypeInfo, In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetMetadataCount, In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelGetMetadata, In_ const OrtModel* model, _Out_ size_t* count, _Out_ const ORTCHAR_T* const key, _Out_ size_t* key_len, _Out_ const ORTCHAR_T* const value, _Out_ size_t* value_len)NO_EXCEPTION;
ORT_API_STATUS_IMPL(ModelCheckIfValid, _In_ OrtModel* model, _Out_ bool* valid)NO_EXCEPTION;

// OrtSession methods
ORT_API_STATUS_IMPL(CreateSessionWihtoutModel, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionRegisterExecutionProvider, _In_ OrtSession* session, onnxruntime::IExecutionProvider* provider)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionInitialize, _In_ OrtSession* session, onnxruntime::IExecutionProvider* provider)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionRegisterGraphTransformers, _In_ OrtSession* session)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionRegisterCustomRegistry, _In_ OrtSession* session, _In_ IMLOperatorRegistry * registry)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionLoadModel, _In_ OrtSession* session, _In_ OrtModel * model)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionStartProfiling, _In_ OrtSession* session)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionEndProfiling, _In_ OrtSession* session)NO_EXCEPTION;
ORT_API_STATUS_IMPL(SessionCopyOneInputAcrossDevices, _In_ OrtSession* session, _In_ const ORTCHAR_T* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;
    
// Dml methods (TODO need to figure out how these need to move to session somehow...)
ORT_API_STATUS_IMPL(DmlExecutionProviderFlushContext, onnxruntime::IExecutionProvider * dml_provider)NO_EXCEPTION;
ORT_API_STATUS_IMPL(DmlExecutionProviderTrimUploadHeap, onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
ORT_API_STATUS_IMPL(DmlExecutionProviderReleaseCompletedReferences, onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
*/
} // namespace Windows::AI::MachineLearning::Adapter