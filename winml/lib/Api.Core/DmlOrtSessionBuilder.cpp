// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

// Needed to work around the fact that OnnxRuntime defines ERROR
#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
// Restore ERROR define
#define ERROR 0

#include "DmlOrtSessionBuilder.h"

// winml includes
#include "core/providers/dml/GraphTransformers/GraphTransformerHelpers.h"
#include "inc/CustomRegistryHelper.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "LearningModelDevice.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"

// ort includes
#include "core/framework/op_kernel.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/customRegistry.h"
#include "core/framework/data_transfer.h"

using namespace Windows::AI::MachineLearning;

DmlOrtSessionBuilder::DmlOrtSessionBuilder(
    ID3D12Device* device, ID3D12CommandQueue* queue){
  device_.copy_from(device);
  queue_.copy_from(queue);
}

HRESULT
DmlOrtSessionBuilder::CreateSessionOptions(
    onnxruntime::SessionOptions* p_options) {
  RETURN_HR_IF_NULL(E_POINTER, p_options);

  *p_options = onnxruntime::SessionOptions();
  
  p_options->graph_optimization_level = onnxruntime::TransformerLevel::Level3;

  // Disable the mem pattern session option for DML. It will cause problems with how memory is allocated.
  p_options->enable_mem_pattern = false;

  return S_OK;
}

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
    const onnxruntime::SessionOptions& options,
    std::unique_ptr<onnxruntime::InferenceSession>* p_session,
    onnxruntime::IExecutionProvider** pp_provider) {
  RETURN_HR_IF_NULL(E_POINTER, p_session);
  RETURN_HR_IF_NULL(E_POINTER, pp_provider);
  RETURN_HR_IF(E_POINTER, *pp_provider != nullptr);

  auto p_d3d_device = device_.get();
  auto p_queue = queue_.get();

  Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice = CreateDmlDevice(p_d3d_device);

  std::unique_ptr<onnxruntime::IExecutionProvider> gpu_provider = Dml::CreateExecutionProvider(dmlDevice.Get(), p_queue);
  auto session = std::make_unique<onnxruntime::InferenceSession>(options);

  // Cache the provider's raw pointer
  *pp_provider = gpu_provider.get();

  ORT_THROW_IF_ERROR(session->RegisterExecutionProvider(std::move(gpu_provider)));

  // return the session
  *p_session = std::move(session);

  return S_OK;
}

HRESULT DmlOrtSessionBuilder::Initialize(
    onnxruntime::InferenceSession* p_session,
    onnxruntime::IExecutionProvider* p_provider) {
  RETURN_HR_IF_NULL(E_INVALIDARG, p_session);
  RETURN_HR_IF_NULL(E_INVALIDARG, p_provider);

  // OnnxRuntime uses the default rounding mode when calling the session's allocator.
  // During initialization, OnnxRuntime allocates weights, which are permanent across session
  // lifetime and can be large, so shouldn't be rounded.
  Dml::SetDefaultRoundingMode(p_provider, AllocatorRoundingMode::Disabled);

  ORT_THROW_IF_ERROR(p_session->Initialize());

  Dml::SetDefaultRoundingMode(p_provider, AllocatorRoundingMode::Enabled);

  // Flush the D3D12 work from the DML execution provider
  Dml::FlushContext(p_provider);

  return S_OK;
}