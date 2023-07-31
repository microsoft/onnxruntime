// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "adapter/pch.h"

#include "winml_adapter_c_api.h"
#include "winml_adapter_apis.h"
#include "core/session/ort_apis.h"

#include <core/providers/winml/winml_provider_factory.h>
#include <core/providers/cpu/cpu_provider_factory.h>

const OrtApi* GetVersion1Api();

namespace winmla = Windows::AI::MachineLearning::Adapter;

static constexpr WinmlAdapterApi winml_adapter_api_1 = {
    // Schema override
  &winmla::OverrideSchema,

    // OrtEnv methods
  &winmla::EnvConfigureCustomLoggerAndProfiler,

    // OrtModel methods
  &winmla::CreateModelFromPath,
  &winmla::CreateModelFromData,
  &winmla::CloneModel,
  &winmla::ModelGetAuthor,
  &winmla::ModelGetName,
  &winmla::ModelSetName,
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
  &winmla::SaveModel,

    // OrtSessionOptions methods
  &OrtSessionOptionsAppendExecutionProvider_CPU,
  &winmla::OrtSessionOptionsAppendExecutionProviderEx_DML,

    // OrtSession methods
  &winmla::CreateSessionWithoutModel,
  &winmla::SessionGetExecutionProvider,
  &winmla::SessionInitialize,
  &winmla::SessionRegisterGraphTransformers,
  &winmla::SessionRegisterCustomRegistry,
  &winmla::SessionLoadAndPurloinModel,
  &winmla::SessionStartProfiling,
  &winmla::SessionEndProfiling,
  &winmla::SessionCopyOneInputAcrossDevices,
  &winmla::SessionGetNumberOfIntraOpThreads,
  &winmla::SessionGetIntraOpThreadSpinning,
  &winmla::SessionGetNamedDimensionsOverrides,

    // Dml methods (TODO need to figure out how these need to move to session somehow...)
  &winmla::DmlExecutionProviderSetDefaultRoundingMode,
  &winmla::DmlExecutionProviderFlushContext,
  &winmla::DmlExecutionProviderReleaseCompletedReferences,
  &winmla::DmlCopyTensor,

  &winmla::GetProviderMemoryInfo,
  &winmla::GetProviderAllocator,
  &winmla::FreeProviderAllocator,

  &winmla::ExecutionProviderSync,

  &winmla::CreateCustomRegistry,

  &winmla::ValueGetDeviceId,
  &winmla::SessionGetInputRequiredDeviceId,

  &winmla::CreateTensorTypeInfo,
  &winmla::CreateSequenceTypeInfo,
  &winmla::CreateMapTypeInfo,
  &winmla::CreateModel,
  &winmla::ModelAddInput,
  &winmla::ModelAddConstantInput,
  &winmla::ModelAddOutput,
  &winmla::ModelAddOperator,
  &winmla::ModelGetOpsetVersion,
  &winmla::OperatorGetNumInputs,
  &winmla::OperatorGetInputName,
  &winmla::OperatorGetNumOutputs,
  &winmla::OperatorGetOutputName,
  &winmla::JoinModels,
  &winmla::CreateThreadPool,

    // Release
  &winmla::ReleaseModel,
  &winmla::ReleaseThreadPool,
};

const WinmlAdapterApi* ORT_API_CALL OrtGetWinMLAdapter(_In_ uint32_t ort_version) NO_EXCEPTION {
  if (ort_version >= 2) {
    return &winml_adapter_api_1;
  }

  return nullptr;
}
