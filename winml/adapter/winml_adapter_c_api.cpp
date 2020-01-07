// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"
#include "winml_adapter_c_api.h"
#include "winml_adapter_apis.h"
#include "WinMLAdapterErrors.h"
#include "CustomRegistryHelper.h"
#include "PheonixSingleton.h"
#include "LotusEnvironment.h"
#include "AbiCustomRegistryImpl.h"

#ifdef USE_DML
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"
#include "DmlOrtSessionBuilder.h"
#endif USE_DML

uint32_t GetVersion1Api();

namespace winmla = Windows::AI::MachineLearning::Adapter;

static constexpr OrtApi winml_adapter_api_1 = {
  // OrtTypeInfo Casting methods
  &winmla::CastTypeInfoToMapTypeInfo,
  &winmla::CastTypeInfoToSequenceTypeInfo,

  // OrtMapTypeInfo Accessors
  &winmla::GetMapKeyType, 
  &winmla::GetMapValueType,

  // OrtSequenceTypeInfo Accessors
  &winmla::GetSequenceElementType, 

  // OrtModel methods
  nullptr, // OrtStatus*(ORT_API_CALL* CreateModel)(_In_ const ORTCHAR_T* model_path, _Outptr_ OrtModel** out)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* CreateModel)(_In_ void* data, _In_ size_t size, _Outptr_ OrtModel** out)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* CreateModel)(_In_ const OrtModel* in, _Outptr_ OrtModel** out)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* CreateModel)(ORTCHAR_T* model_path, _Outptr_ OrtModel** out)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetAuthor(_In_ const OrtModel* model, _Out_ const ORTCHAR_T* const author, _Out_ size_t* len)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetName(_In_ const OrtModel* model, _Out_ const ORTCHAR_T* const name, _Out_ size_t* len)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetDomain(_In_ const OrtModel* model, _Out_ const ORTCHAR_T* const domain, _Out_ size_t* len)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetDescription(_In_ const OrtModel* model, _Out_ const ORTCHAR_T* const description, _Out_ size_t* len)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetVersion(_In_ const OrtModel* model, _Out_ int64_t* version)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetInputCount(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetOutputCount(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetInputName(_In_ const OrtModel* model, _Out_ const char** input_name, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetOutputName(_In_ const OrtModel* model, _Out_ const char** output_name, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetInputDescription(_In_ const OrtModel* model, _Out_ const char** input_description, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetOutputDescription(_In_ const OrtModel* model, _Out_ const char** output_description, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetInputTypeInfo(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetOutputTypeInfo(_In_ const OrtModel* model, _In_ size_t index, _Outptr_ OrtTypeInfo** type_info)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetMetadataCount(_In_ const OrtModel* model, _Out_ size_t* count)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelGetMetadata(_In_ const OrtModel* model, _Out_ size_t* count, _Out_ const ORTCHAR_T* const key, _Out_ size_t* key_len, _Out_ const ORTCHAR_T* const value, _Out_ size_t* value_len)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* ModelCheckIfValid)(_In_ OrtModel* model, _Out_ bool* valid)NO_EXCEPTION;

  // OrtSession methods
  nullptr, // OrtStatus*(ORT_API_CALL* CreateSessionWihtoutModel)(_In_ const OrtSessionOptions* options, _Outptr_ OrtSession** session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionRegisterExecutionProvider)(_In_ OrtSession* session, onnxruntime::IExecutionProvider* provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionInitialize)(_In_ OrtSession* session, onnxruntime::IExecutionProvider* provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionRegisterGraphTransformers)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionRegisterCustomRegistry)(_In_ OrtSession* session, _In_ IMLOperatorRegistry * registry)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionLoadAndPurloinModel)(_In_ OrtSession* session, _In_ OrtModel * model)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionStartProfiling)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionEndProfiling)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionCopyOneInputAcrossDevices)(_In_ OrtSession* session, _In_ const ORTCHAR_T* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;
      
  // Dml methods (TODO need to figure out how these need to move to session somehow...)
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext(onnxruntime::IExecutionProvider * dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderTrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  &winmla::ReleaseModel,
  &winmla::ReleaseMapTypeInfo,
  &winmla::ReleaseSequenceTypeInfo,
};

const WinmlAdapterApi* ORT_API_CALL GetWinmlAdapterApi(OrtApi* ort_api) NO_EXCEPTION {
  if (GetVersion1Api() == ort_api)
  {
    return &winml_adapter_api_1;
  }

  return nullptr;
}