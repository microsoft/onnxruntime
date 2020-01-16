// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"

#include "core/providers/dml/dml_provider_factory.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

Microsoft::WRL::ComPtr<IDMLDevice> CreateDmlDevice(ID3D12Device* d3d12Device) {
  // Dynamically load DML to avoid WinML taking a static dependency on DirectML.dll
  wil::unique_hmodule dmlDll(LoadLibraryW(L"DirectML.dll"));
  THROW_LAST_ERROR_IF(!dmlDll);

  auto dmlCreateDevice1Fn = reinterpret_cast<decltype(&DMLCreateDevice1)>(
      GetProcAddress(dmlDll.get(), "DMLCreateDevice1"));
  THROW_LAST_ERROR_IF(!dmlCreateDevice1Fn);

  DML_CREATE_DEVICE_FLAGS dmlFlags = DML_CREATE_DEVICE_FLAG_NONE;

  // Enable the DML debug layer in DEBUG builds, if the D3D12 debug layer is also enabled
#if _DEBUG
  Microsoft::WRL::ComPtr<ID3D12DebugDevice> d3d12DebugDevice;
  if (SUCCEEDED(d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12DebugDevice)))) {
    d3d12DebugDevice = nullptr;
    dmlFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
  }
#endif

  Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice;
  THROW_IF_FAILED(dmlCreateDevice1Fn(d3d12Device, dmlFlags, DML_FEATURE_LEVEL_2_0, IID_PPV_ARGS(&dmlDevice)));

  // Keep DirectML.dll loaded by leaking the handle. This is equivalent behavior to if we delay-loaded the DLL.
  dmlDll.release();

  return dmlDevice;
}

ORT_API_STATUS_IMPL(winmla::OrtSessionOptionsAppendExecutionProviderEx_DML, _In_ OrtSessionOptions* options,
                    ID3D12Device* d3d_device, ID3D12CommandQueue* queue) {
  auto dml_device = CreateDmlDevice(d3d_device);
  return OrtSessionOptionsAppendExecutionProviderEx_DML(options, dml_device.Get(), queue);
}

ORT_API_STATUS_IMPL(winmla::DmlExecutionProviderFlushContext, _In_ OrtExecutionProvider* dml_provider) {
  API_IMPL_BEGIN
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  Dml::FlushContext(dml_provider_internal);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::DmlExecutionProviderTrimUploadHeap, _In_ OrtExecutionProvider* dml_provider) {
  API_IMPL_BEGIN
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  Dml::TrimUploadHeap(dml_provider_internal);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::DmlExecutionProviderReleaseCompletedReferences, _In_ OrtExecutionProvider* dml_provider) {
  API_IMPL_BEGIN
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  Dml::ReleaseCompletedReferences(dml_provider_internal);
  return nullptr;
  API_IMPL_END
}