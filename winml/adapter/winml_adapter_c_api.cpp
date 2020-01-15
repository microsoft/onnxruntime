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
  // Schema override
  &winmla::OverrideSchema,

  // OrtEnv methods
  &winmla::EnvConfigureCustomLoggerAndProfiler,

  // OrtTypeInfo Casting methods
  &winmla::GetDenotationFromTypeInfo,
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
  &winmla::ModelGetName,
  &winmla::ModelGetDomain,
  &winmla::ModelGetDescription,
  &winmla::ModelGetVersion,
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

  // OrtSessionOptions methods
  &OrtSessionOptionsAppendExecutionProvider_CPU,
  &winmla::OrtSessionOptionsAppendExecutionProviderEx_DML, 

  // OrtSession methods
  &winmla::CreateSessionWithoutModel,
  &winmla::SessionGetExecutionProvidersCount,
  &winmla::SessionGetExecutionProvider,
  &winmla::SessionInitialize,
  &winmla::SessionRegisterGraphTransformers,
  &winmla::SessionRegisterCustomRegistry,
  &winmla::SessionLoadAndPurloinModel,
  &winmla::SessionStartProfiling,
  &winmla::SessionEndProfiling,
  nullptr, // OrtStatus*(ORT_API_CALL* SessionCopyOneInputAcrossDevices)(_In_ OrtSession* session, _In_ const char* const input_name, _In_ const OrtValue* orig_value, _Outptr_ OrtValue** new_value)NO_EXCEPTION;
      
  // Dml methods (TODO need to figure out how these need to move to session somehow...)
  nullptr, //OrtStatus*(ORT_API_CALL* DmlExecutionProviderSetDefaultRoundingMode)(_In_ bool is_enabled)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderFlushContext(onnxruntime::IExecutionProvider * dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderTrimUploadHeap(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlExecutionProviderReleaseCompletedReferences(onnxruntime::IExecutionProvider* dml_provider)NO_EXCEPTION;
  
  nullptr, // OrtStatus*(ORT_API_CALL* DmlCreateGPUAllocationFromD3DResource)(_In_ ID3D12Resource* pResource, _Out_ void* dml_resource)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlFreeGPUAllocation)(_In_ void* ptr)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* DmlGetD3D12ResourceFromAllocation)(_In_ OrtExecutionProvider* provider, _In_ void* allocation, _Out_ ID3D12Resource** resource)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* GetProviderMemoryInfo)(_In_ OrtExecutionProvider* provider, OrtMemoryInfo** memory_info)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* GetProviderAllocator)(_In_ OrtExecutionProvider* provider, OrtAllocator** allocator)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* FreeProviderAllocator)(_In_ OrtAllocator* allocator)NO_EXCEPTION;
  nullptr, // OrtStatus*(ORT_API_CALL* GetValueMemoryInfo)(const OrtValue * value, OrtMemoryInfo** memory_info)NO_EXCEPTION;

  &winmla::ExecutionProviderSync,
  
  nullptr, // OrtStatus*(ORT_API_CALL* ExecutionProviderCopyTensor)(_In_ OrtExecutionProvider* provider, _In_ OrtValue* src, _In_ OrtValue* dst)NO_EXCEPTION;
  // Release
  &winmla::ReleaseModel,
  &winmla::ReleaseMapTypeInfo,
  &winmla::ReleaseSequenceTypeInfo
};

const WinmlAdapterApi* ORT_API_CALL GetWinmlAdapterApi(const OrtApi* ort_api) NO_EXCEPTION {
  if (GetVersion1Api() == ort_api)
  {
    return &winml_adapter_api_1;
  }

  return nullptr;
}