// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include "pch.h"

#include "winml_adapter_c_api.h"
#include "winml_adapter_apis.h"

#include <core/providers/cpu/cpu_provider_factory.h>
#include <core/providers/dml/dml_provider_factory.h>

#ifdef USE_DML
//#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
//#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
//#include "core/providers/dml/OperatorAuthorHelper/SchemaInferenceOverrider.h"
#endif USE_DML

const OrtApi* GetVersion1Api();

namespace winmla = Windows::AI::MachineLearning::Adapter;


static constexpr WinmlAdapterApi winml_adapter_api_1 = {
  &winmla::EnvConfigureCustomLoggerAndProfiler,
  &winmla::GetDenotationFromTypeInfo,

  // OrtTypeInfo Casting methods
  &winmla::CastTypeInfoToMapTypeInfo,
  &winmla::CastTypeInfoToSequenceTypeInfo,

  // OrtMapTypeInfo Accessors
  &winmla::GetMapKeyType, 
  &winmla::GetMapValueType,

  // OrtSequenceTypeInfo Accessors
  &winmla::GetSequenceElementType, 

  // OrtModel methods
  &winmla::CreateModelFromPath,
  &winmla::CreateModelFromData,  
  &winmla::CloneModel,
  &winmla::ModelGetAuthor,
  &winmla::ModelGetName, // OrtStatus*(ORT_API_CALL* ModelGetName(_In_ const OrtModel* model, _Out_ const char* const name, _Out_ size_t* len)NO_EXCEPTION;
  &winmla::ModelGetDomain, // OrtStatus*(ORT_API_CALL* ModelGetDomain(_In_ const OrtModel* model, _Out_ const char* const domain, _Out_ size_t* len)NO_EXCEPTION;
  &winmla::ModelGetDescription, // OrtStatus*(ORT_API_CALL* ModelGetDescription(_In_ const OrtModel* model, _Out_ const char* const description, _Out_ size_t* len)NO_EXCEPTION;
  &winmla::ModelGetVersion, // OrtStatus*(ORT_API_CALL* ModelGetVersion(_In_ const OrtModel* model, _Out_ int64_t* version)NO_EXCEPTION;
  &winmla::ModelGetInputCount,
  &winmla::ModelGetOutputCount,
  &winmla::ModelGetInputName,
  &winmla::ModelGetOutputName,
  &winmla::ModelGetInputDescription,
  &winmla::ModelGetOutputDescription,
  &winmla::ModelGetInputTypeInfo,
  &winmla::ModelGetOutputTypeInfo,
  &winmla::ModelGetMetadataCount,
  &winmla::ModelGetMetadata,
  &winmla::ModelEnsureNoFloat16,

  // OrtSession methods
  &winmla::CreateSessionWithoutModel,
  &winmla::SessionGetExecutionProvidersCount,
  &winmla::SessionGetExecutionProvider,
  &winmla::SessionInitialize,
  nullptr, // OrtStatus*(ORT_API_CALL* SessionRegisterGraphTransformers)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionRegisterCustomRegistry)(_In_ OrtSession* session, _In_ IMLOperatorRegistry * registry)NO_EXCEPTION;
  &winmla::SessionLoadAndPurloinModel,
  nullptr, // OrtStatus*(ORT_API_CALL* SessionStartProfiling)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionEndProfiling)(_In_ OrtSession* session)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* SessionCopyOneInputAcrossDevices)(_In_ OrtSession* session, _In_ const char* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;
      
  // Dml methods (TODO need to figure out how these need to move to session somehow...)
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext(onnxruntime::IExecutionProvider * dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderTrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  &OrtSessionOptionsAppendExecutionProvider_CPU,
  &winmla::OrtSessionOptionsAppendExecutionProviderEx_DML, 
  &winmla::ReleaseModel,
  &winmla::ReleaseMapTypeInfo,
  &winmla::ReleaseSequenceTypeInfo,
  &winmla::ReleaseExecutionProvider,
  &winmla::ReleaseOperatorRegistry
};

const WinmlAdapterApi* ORT_API_CALL GetWinmlAdapterApi(const OrtApi* ort_api) NO_EXCEPTION {
  if (GetVersion1Api() == ort_api)
  {
    return &winml_adapter_api_1;
  }

  return nullptr;
}