// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#ifdef USE_DML

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#include "DmlOrtSessionBuilder.h"
#include "WinMLAdapterErrors.h"

// winml includes
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "CustomRegistryHelper.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "LearningModelDevice.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"

// ort includes
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/customRegistry.h"
#include "core/framework/data_transfer.h"
#include "core/session/abi_session_options_impl.h"

using namespace Windows::AI::MachineLearning;

namespace Windows::AI::MachineLearning::Adapter {

DmlOrtSessionBuilder::DmlOrtSessionBuilder(
    ID3D12Device* device,
    ID3D12CommandQueue* queue) {
  device_.copy_from(device);
  queue_.copy_from(queue);
}

HRESULT
DmlOrtSessionBuilder::CreateSessionOptions(
    OrtSessionOptions** options) try {
  RETURN_HR_IF_NULL(E_POINTER, options);

  Ort::ThrowOnError(Ort::GetApi().CreateSessionOptions(options));
  Ort::SessionOptions session_options(*options);

  // set the graph optimization level to all (used to be called level 3)
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Disable the mem pattern session option for DML. It will cause problems with how memory is allocated.
  session_options.DisableMemPattern();

  // call release() so the underlying OrtSessionOptions object isn't freed
  session_options.release();
  return S_OK;
}
WINMLA_CATCH_ALL_COM

static HRESULT
RegisterCustomRegistry(
    onnxruntime::InferenceSession* p_session,
    IMLOperatorRegistry* registry) {
  if (registry != nullptr) {
    RETURN_HR_IF_NULL(E_POINTER, p_session);

    auto custom_registries = GetLotusCustomRegistries(registry);

    // Register
    for (auto& custom_registry : custom_registries) {
      ORT_THROW_IF_ERROR(p_session->RegisterCustomRegistry(custom_registry));
    }
  }

  return S_OK;
}

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

HRESULT DmlOrtSessionBuilder::CreateSession(
    OrtSessionOptions* options,
    winmla::IInferenceSession** p_session,
    onnxruntime::IExecutionProvider** pp_provider) try {
  RETURN_HR_IF_NULL(E_POINTER, p_session);
  RETURN_HR_IF_NULL(E_POINTER, pp_provider);
  RETURN_HR_IF(E_POINTER, *pp_provider != nullptr);

  auto p_d3d_device = device_.get();
  auto p_queue = queue_.get();

  Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice = CreateDmlDevice(p_d3d_device);

  std::unique_ptr<onnxruntime::IExecutionProvider> gpu_provider = Dml::CreateExecutionProvider(dmlDevice.Get(), p_queue);
  auto session = std::make_unique<onnxruntime::InferenceSession>(options->value);

  const onnxruntime::Env& env = onnxruntime::Env::Default();
  LUID temp_LUID = p_d3d_device->GetAdapterLuid();
  env.GetTelemetryProvider().LogExecutionProviderEvent(&temp_LUID);
  // Cache the provider's raw pointer
  *pp_provider = gpu_provider.get();

  ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(gpu_provider)));

  // assign the session to the out parameter
  auto sessionptr = wil::MakeOrThrow<winmla::InferenceSession>(session.release());
  RETURN_IF_FAILED(sessionptr.CopyTo(_uuidof(winmla::IInferenceSession), (void**)p_session));

  return S_OK;
}
WINMLA_CATCH_ALL_COM

HRESULT DmlOrtSessionBuilder::Initialize(
    winmla::IInferenceSession* p_session,
    onnxruntime::IExecutionProvider* p_provider) try {
  RETURN_HR_IF_NULL(E_INVALIDARG, p_session);
  RETURN_HR_IF_NULL(E_INVALIDARG, p_provider);

  // OnnxRuntime uses the default rounding mode when calling the session's allocator.
  // During initialization, OnnxRuntime allocates weights, which are permanent across session
  // lifetime and can be large, so shouldn't be rounded.
  Dml::SetDefaultRoundingMode(p_provider, AllocatorRoundingMode::Disabled);

  ORT_THROW_IF_ERROR(p_session->get()->Initialize());

  Dml::SetDefaultRoundingMode(p_provider, AllocatorRoundingMode::Enabled);

  // Flush the D3D12 work from the DML execution provider
  Dml::FlushContext(p_provider);

  return S_OK;
}
WINMLA_CATCH_ALL_COM

}  // namespace Windows::AI::MachineLearning::Adapter

#endif USE_DML