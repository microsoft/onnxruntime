// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "adapter/pch.h"

#include "onnxruntime_win_c_api.h"
#include "winml_adapter_apis.h"
#include "core/session/ort_apis.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

static constexpr OrtWinApi onnxruntime_win_api_1 = {
    &winmla::SessionGetExecutionProvider,
    &winmla::GetProviderMemoryInfo,
    &winmla::GetProviderAllocator,
    &winmla::FreeProviderAllocator,

    // DML methods
    &winmla::DmlExecutionProviderSetDefaultRoundingMode,
    &winmla::DmlExecutionProviderFlushContext,
    &winmla::DmlExecutionProviderReleaseCompletedReferences,
    &winmla::DmlCreateGPUAllocationFromD3DResource,
    &winmla::DmlFreeGPUAllocation,
    &winmla::DmlGetD3D12ResourceFromAllocation,
    &winmla::DmlCopyTensor
};

const OrtWinApi* ORT_API_CALL OrtGetWindowsApi(_In_ const OrtApi* ort_api) NO_EXCEPTION {
  if (OrtApis::GetApi(2) == ort_api) {
    return &onnxruntime_win_api_1;
  }

  return nullptr;
}
