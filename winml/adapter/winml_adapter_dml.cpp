// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "adapter/pch.h"

#include "winml_adapter_c_api.h"
#include "core/session/ort_apis.h"
#include "winml_adapter_apis.h"
#include "core/framework/error_code_helper.h"

#ifdef USE_DML
#include "core/session/abi_session_options_impl.h"
#include "core/providers/dml/dml_provider_factory.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include <windows.h>
#endif  // USE_DML

namespace winmla = Windows::AI::MachineLearning::Adapter;

#ifdef USE_DML

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

static std::wstring CurrentModulePath() {
  WCHAR path[MAX_PATH];
  FAIL_FAST_IF(0 == GetModuleFileNameW((HINSTANCE)&__ImageBase, path, _countof(path)));

  WCHAR absolute_path[MAX_PATH];
  WCHAR* name;
  FAIL_FAST_IF(0 == GetFullPathNameW(path, _countof(path), absolute_path, &name));

  auto idx = std::distance(absolute_path, name);
  auto out_path = std::wstring(absolute_path);
  out_path.resize(idx);

  return out_path;
}

Microsoft::WRL::ComPtr<IDMLDevice> CreateDmlDevice(ID3D12Device* d3d12Device) {
  // Dynamically load DML to avoid WinML taking a static dependency on DirectML.dll
  auto directml_dll = CurrentModulePath() + L"DirectML.dll";
  wil::unique_hmodule dmlDll(LoadLibraryExW(directml_dll.c_str(), nullptr, 0));
  THROW_LAST_ERROR_IF(!dmlDll);

  auto dmlCreateDevice1Fn =
    reinterpret_cast<decltype(&DMLCreateDevice1)>(GetProcAddress(dmlDll.get(), "DMLCreateDevice1"));
  THROW_LAST_ERROR_IF(!dmlCreateDevice1Fn);

  DML_CREATE_DEVICE_FLAGS dmlFlags = DML_CREATE_DEVICE_FLAG_NONE;

  // Enable the DML debug layer in DEBUG builds, if the D3D12 debug layer is also enabled
#if _DEBUG
  Microsoft::WRL::ComPtr<ID3D12DebugDevice> d3d12DebugDevice;
  if (SUCCEEDED(d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12DebugDevice)))) {
    d3d12DebugDevice = nullptr;
    dmlFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
  }
#endif  // USE_DML

  Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice;
  THROW_IF_FAILED(dmlCreateDevice1Fn(d3d12Device, dmlFlags, DML_FEATURE_LEVEL_2_0, IID_PPV_ARGS(&dmlDevice)));

  // Keep DirectML.dll loaded by leaking the handle. This is equivalent behavior to if we delay-loaded the DLL.
  dmlDll.release();

  return dmlDevice;
}

namespace onnxruntime {
void DmlConfigureProviderFactoryMetacommandsEnabled(IExecutionProviderFactory* factory, bool metacommandsEnabled);
}// namespace onnxruntime

#endif  // USE_DML

ORT_API_STATUS_IMPL(
  winmla::OrtSessionOptionsAppendExecutionProviderEx_DML,
  _In_ OrtSessionOptions* options,
  _In_ ID3D12Device* d3d_device,
  _In_ ID3D12CommandQueue* queue,
  bool metacommands_enabled
) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto dml_device = CreateDmlDevice(d3d_device);
  if (auto status = OrtSessionOptionsAppendExecutionProviderEx_DML(options, dml_device.Get(), queue)) {
    return status;
  }
  auto factory = options->provider_factories.back().get();
  
  onnxruntime::DmlConfigureProviderFactoryMetacommandsEnabled(factory, metacommands_enabled);
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::DmlExecutionProviderFlushContext, _In_ OrtExecutionProvider* dml_provider) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  Dml::FlushContext(dml_provider_internal);
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(winmla::DmlExecutionProviderReleaseCompletedReferences, _In_ OrtExecutionProvider* dml_provider) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  Dml::ReleaseCompletedReferences(dml_provider_internal);
#endif  // USE_DML
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(
  winmla::DmlCopyTensor, _In_ OrtExecutionProvider* dml_provider, _In_ OrtValue* src, _In_ OrtValue* dst
) {
  API_IMPL_BEGIN
#ifdef USE_DML
  auto dml_provider_internal = reinterpret_cast<::onnxruntime::IExecutionProvider*>(dml_provider);
  auto status = Dml::CopyTensor(
    dml_provider_internal, *(src->GetMutable<onnxruntime::Tensor>()), *(dst->GetMutable<onnxruntime::Tensor>())
  );
  if (!status.IsOK()) {
    return onnxruntime::ToOrtStatus(status);
  }
  return nullptr;
#else
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Out of memory");
#endif  // USE_DML USE_DML
  API_IMPL_END
}
