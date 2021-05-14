// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "inc/D3DDeviceCache.h"
#include <directxmath.h>
#include <d3d11on12.h>
#include "inc/DeviceHelpers.h"
#include "CommonDeviceHelpers.h"

namespace float32 {
#include "shaders\SurfaceToTensor-SurfaceToTensorBGR8.h"
#include "shaders\SurfaceToTensor-SurfaceToTensorRGB8.h"
#include "shaders\SurfaceToTensor-SurfaceToTensorGRAY8.h"
#include "shaders\SurfaceToTensor-SurfaceGRAY8ToTensorBGR8.h"
#include "shaders\SurfaceToTensor-SurfaceGRAY8ToTensorGRAY8.h"
#include "shaders\TensorToSurface-TensorBGR8ToSurface.h"
#include "shaders\TensorToSurface-TensorRGB8ToSurface.h"
#include "shaders\TensorToSurface-TensorGRAY8ToSurface.h"
#include "shaders\TensorToSurface-TensorBGR8ToSurfaceGRAY8.h"
#include "shaders\TensorToSurface-TensorRGB8ToSurfaceGRAY8.h"
#include "shaders\TensorToSurface-TensorGRAY8ToSurfaceGRAY8.h"
}  // namespace float32

namespace float16 {
#include "shaders\SurfaceToTensor16-SurfaceToTensorBGR8.h"
#include "shaders\SurfaceToTensor16-SurfaceToTensorRGB8.h"
#include "shaders\SurfaceToTensor16-SurfaceToTensorGRAY8.h"
#include "shaders\SurfaceToTensor16-SurfaceGRAY8ToTensorBGR8.h"
#include "shaders\SurfaceToTensor16-SurfaceGRAY8ToTensorGRAY8.h"
#include "shaders\TensorToSurface16-TensorBGR8ToSurface.h"
#include "shaders\TensorToSurface16-TensorRGB8ToSurface.h"
#include "shaders\TensorToSurface16-TensorGRAY8ToSurface.h"
#include "shaders\TensorToSurface16-TensorBGR8ToSurfaceGRAY8.h"
#include "shaders\TensorToSurface16-TensorRGB8ToSurfaceGRAY8.h"
#include "shaders\TensorToSurface16-TensorGRAY8ToSurfaceGRAY8.h"
}  // namespace float16

using namespace Microsoft::WRL;

using namespace _winml;

D3DDeviceCache::D3DDeviceCache(winml::LearningModelDeviceKind const& deviceKind) {
  WINML_THROW_IF_FAILED(CoCreateGuid(&fence_guid_));

  if (deviceKind == winml::LearningModelDeviceKind::Cpu || deviceKind == winml::LearningModelDeviceKind::Default) {
    // CPU device don't make any GPU devices
    device_luid_.HighPart = device_luid_.LowPart = 0;
    return;
  }

  DXGI_GPU_PREFERENCE preference;
  WINML_THROW_IF_FAILED(GetGPUPreference(deviceKind, &preference));

  CommonDeviceHelpers::AdapterEnumerationSupport support;
  WINML_THROW_IF_FAILED(CommonDeviceHelpers::GetAdapterEnumerationSupport(&support));

  const char noHardwareAdaptersAvailableErrStr[] = "No hardware adapters available";
  const char failedToObtainHardwareAdaptersErrStr[] = "Failed to obtain hardware adapters.";
  HRESULT hardwareAdapterSuccessfullyObtained = S_OK;
  if (support.has_dxgi) {
    winrt::com_ptr<IDXGIAdapter1> spAdapter;
    hardwareAdapterSuccessfullyObtained = GetDXGIHardwareAdapterWithPreference(preference, spAdapter.put());
    if (hardwareAdapterSuccessfullyObtained == HRESULT_FROM_WIN32(ERROR_NOT_FOUND)) {
      WINML_THROW_HR_MSG_NO_TELEMETRY_SENT(hardwareAdapterSuccessfullyObtained, noHardwareAdaptersAvailableErrStr);
    } else {
      WINML_THROW_IF_FAILED_MSG(hardwareAdapterSuccessfullyObtained, failedToObtainHardwareAdaptersErrStr);
    }
    WINML_THROW_IF_FAILED(D3D12CreateDevice(spAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device_.put())));
  }
#ifdef ENABLE_DXCORE
  if (support.has_dxgi == false) {
    winrt::com_ptr<IDXCoreAdapter> spAdapter;
    hardwareAdapterSuccessfullyObtained = GetDXCoreHardwareAdapterWithPreference(preference, spAdapter.put());
    if (hardwareAdapterSuccessfullyObtained == HRESULT_FROM_WIN32(ERROR_NOT_FOUND)) {
      WINML_THROW_HR_MSG_NO_TELEMETRY_SENT(hardwareAdapterSuccessfullyObtained, noHardwareAdaptersAvailableErrStr);
    } else {
      WINML_THROW_IF_FAILED_MSG(hardwareAdapterSuccessfullyObtained, failedToObtainHardwareAdaptersErrStr);
    }
    WINML_THROW_IF_FAILED(D3D12CreateDevice(spAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device_.put())));
  }
#endif
  InitializeCommandQueue(device_.get());

  device_luid_ = device_->GetAdapterLuid();
}

D3DDeviceCache::D3DDeviceCache(wgdx::Direct3D11::IDirect3DDevice const& device) {
  WINML_THROW_IF_FAILED(CoCreateGuid(&fence_guid_));

  // Use the 11 device to initialize 12
  winrt_device_ = device;

  // they told us which device to run on, crack the interop wrapper to get the dxgi device
  winrt::com_ptr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> dxgi;
  dxgi = device.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();

  winrt::com_ptr<IDXGIDevice> dxgiDevice;
  WINML_THROW_IF_FAILED(dxgi->GetInterface(IID_PPV_ARGS(dxgiDevice.put())));

  device_11_ = dxgiDevice.as<ID3D11Device>();

  winrt::com_ptr<ID3D11DeviceContext> spContext;
  device_11_->GetImmediateContext(spContext.put());
  spContext.as(device_context11_);

  winrt::com_ptr<IDXGIDevice> pDXGIDevice;
  WINML_THROW_IF_FAILED(dxgi->GetInterface(IID_PPV_ARGS(pDXGIDevice.put())));

  winrt::com_ptr<IDXGIAdapter> adapter;
  WINML_THROW_IF_FAILED(pDXGIDevice->GetAdapter(adapter.put()));

  WINML_THROW_IF_FAILED(D3D12CreateDevice(adapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device_.put())));

  InitializeCommandQueue(device_.get());

  device_luid_ = device_->GetAdapterLuid();
}

D3DDeviceCache::D3DDeviceCache(ID3D12CommandQueue* queue) {
  WINML_THROW_IF_FAILED(CoCreateGuid(&fence_guid_));

  // Use the command queue to initialize all of the needed D3D11 interop
  command_queue_.copy_from(queue);
  command_queue_->QueryInterface(IID_PPV_ARGS(sharing_contract_.put()));

  WINML_THROW_IF_FAILED(queue->GetDevice(IID_PPV_ARGS(device_.put())));

  device_luid_ = device_->GetAdapterLuid();
}

D3DDeviceCache::~D3DDeviceCache() {
  // If this is a CPU instance device_ will not have been created.
  // Ensure the device is still valid before doing work.
  if (device_ != nullptr && (device_->GetDeviceRemovedReason() == S_OK)) {
    // dx11 stack is optional, and we lazy load it when available
    if (device_context11_ != nullptr) {
      // Sync 11 to 12 then Sync 12 to the CPU. This ensures that all inflight work is done before we delete the d3d objects.
      GPUSyncD3D11ToD3D12();
    }
    SyncD3D12ToCPU();
  }
}

bool D3DDeviceCache::IsFloat16Supported() {
  if (device_ != nullptr) {
    return CommonDeviceHelpers::IsFloat16Supported(device_.get());
  }

  return true;
}

ID3D11Device* D3DDeviceCache::GetD3D11Device() {
  EnsureD3D11FromD3D12();
  return device_11_.get();
}

const GUID& D3DDeviceCache::GetFenceGuid() const {
  return fence_guid_;
}

ID3D11DeviceContext4* D3DDeviceCache::GetD3D11DeviceContext() {
  EnsureD3D11FromD3D12();
  return device_context11_.get();
}

wgdx::Direct3D11::IDirect3DDevice D3DDeviceCache::GetWinrtDevice() {
  EnsureD3D11FromD3D12();
  return winrt_device_;
}

void D3DDeviceCache::InitializeCommandQueue(ID3D12Device1* device) {
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
  WINML_THROW_IF_FAILED(device->CreateCommandQueue(&queueDesc, winrt::guid_of<ID3D12CommandQueue>(), command_queue_.put_void()));

  // If possible get the sharing context. If not leave nullptr;
  command_queue_->QueryInterface(IID_PPV_ARGS(sharing_contract_.put()));
}

// this initializes the following variables, making them from the dx12 device
//      device_11_
//      device_context11_
//      winrt_device_
void D3DDeviceCache::EnsureD3D11FromD3D12() {
  // do we even have a device?  (CPU will use the cache but not have a device) .
  if (device_ == nullptr)
    return;

  // are we already initialized
  if (winrt_device_ != nullptr)
    return;

  CWinMLAutoLock lock(&lock_);

  // check with the lock held, are we already initialized
  if (winrt_device_ != nullptr)
    return;

  winrt::com_ptr<::IInspectable> spInspectable;
  winrt::com_ptr<IDXGIDevice> spDXGIDevice;

  // call our SEH version (for delay loading)
  WINML_THROW_IF_FAILED(CreateD3D11On12Device(device_.get(), device_11_.put()));
  winrt::com_ptr<ID3D11DeviceContext> spContext;
  device_11_->GetImmediateContext(spContext.put());
  spContext.as(device_context11_);

  WINML_THROW_IF_FAILED(device_11_->QueryInterface(IID_PPV_ARGS(spDXGIDevice.put())));
  // Convert to Winrt wrapper. This doesn't actually make a new device.
  WINML_THROW_IF_FAILED(CreateDirect3D11DeviceFromDXGIDevice(spDXGIDevice.get(), spInspectable.put()));
  WINML_THROW_IF_FAILED(spInspectable->QueryInterface(winrt::guid_of<wgdx::Direct3D11::IDirect3DDevice>(), reinterpret_cast<void**>(winrt::put_abi(winrt_device_))));
}

void D3DDeviceCache::EnsureD3D12Fence() {
  // are we already initialized?
  if (d3d12_fence_ != nullptr)
    return;

  CWinMLAutoLock lock(&lock_);

  // with the lock held, are we already initialized?
  if (d3d12_fence_ != nullptr)
    return;

  WINML_THROW_IF_FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(d3d12_fence_.put())));
}

// this initializes the following variables, so that we can share dx12 with dx11
//      d3d11_fence_
//      d3d12_fence_
void D3DDeviceCache::EnsureSharedFences() {
  // are we already initialized?
  if (d3d11_fence_ != nullptr)
    return;

  CWinMLAutoLock lock(&lock_);

  // with the lock held, are we already initialized?
  if (d3d11_fence_ != nullptr)
    return;

  EnsureD3D12Fence();

  // ensure the d11 stack is alive, the 11 stack doesn't exist on WCOSHeadless yet, so be resilient
  EnsureD3D11FromD3D12();

  winrt::com_ptr<ID3D12DeviceChild> spD3D12DeviceChild;
  d3d12_fence_.as(spD3D12DeviceChild);
  HANDLE hSharedFence;
  WINML_THROW_IF_FAILED(device_->CreateSharedHandle(spD3D12DeviceChild.get(), NULL, GENERIC_ALL, nullptr, &hSharedFence));

  winrt::com_ptr<ID3D11Device5> spD3D11Device5;
  device_11_.as(spD3D11Device5);
  wil::unique_handle safe(hSharedFence);
  WINML_THROW_IF_FAILED(spD3D11Device5->OpenSharedFence(safe.get(), IID_PPV_ARGS(d3d11_fence_.put())));
}

void D3DDeviceCache::GPUSyncD3D11ToD3D12() {
  EnsureSharedFences();

  UINT64 currentFence = fence_value_++;
  WINML_THROW_IF_FAILED(device_context11_->Signal(d3d11_fence_.get(), currentFence));

  WINML_THROW_IF_FAILED(command_queue_->Wait(d3d12_fence_.get(), currentFence));

  if (sharing_contract_ != nullptr) {
    sharing_contract_->SharedFenceSignal(d3d12_fence_.get(), currentFence);
  }
}

void D3DDeviceCache::GPUSyncD3D12ToD3D11() {
  EnsureSharedFences();

  UINT64 currentFence = fence_value_++;
  WINML_THROW_IF_FAILED(command_queue_->Signal(d3d12_fence_.get(), currentFence));

  WINML_THROW_IF_FAILED(device_context11_->Wait(d3d11_fence_.get(), currentFence));
}

void D3DDeviceCache::SyncD3D12ToCPU() {
  UINT64 currentFence = QueueFenceToD3D12();
  WaitForFenceValue(currentFence);
}

UINT64 D3DDeviceCache::QueueFenceToD3D12() {
  EnsureD3D12Fence();

  UINT64 currentFence = fence_value_++;
  WINML_THROW_IF_FAILED(command_queue_->Signal(d3d12_fence_.get(), currentFence));

  return currentFence;
}

void D3DDeviceCache::WaitForFenceValue(UINT64 fenceValue) {
  EnsureD3D12Fence();

  wil::unique_handle event(CreateEvent(nullptr, FALSE, FALSE, nullptr));
  THROW_LAST_ERROR_IF(!event);

  WINML_THROW_IF_FAILED(d3d12_fence_->SetEventOnCompletion(fenceValue, event.get()));

  DWORD retVal = WaitForSingleObject(event.get(), INFINITE);
  if (retVal != WAIT_OBJECT_0) {
    WINML_THROW_IF_FAILED(E_UNEXPECTED);
  }
}

ID3D12RootSignature* D3DDeviceCache::GetTensorizeRootSignature() {
  if (tensorize_root_signature_ == nullptr) {
    winrt::com_ptr<ID3D12RootSignature> newRootSignature;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

    // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
      featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // Compute root signature.
    {
      CD3DX12_DESCRIPTOR_RANGE1 ranges[2] = {};
      ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
      ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);

      CD3DX12_ROOT_PARAMETER1 rootParameters[3] = {};
      rootParameters[0].InitAsConstants(4, 0);
      rootParameters[1].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
      rootParameters[2].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_ALL);

      CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC computeRootSignatureDesc;
      computeRootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr);

      winrt::com_ptr<ID3DBlob> signature;
      winrt::com_ptr<ID3DBlob> error;
      WINML_THROW_IF_FAILED(D3DX12SerializeVersionedRootSignature(&computeRootSignatureDesc, featureData.HighestVersion, signature.put(), error.put()));
      WINML_THROW_IF_FAILED(device_->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(newRootSignature.put())));
      newRootSignature->SetName(L"Tensorize Rootsignature");
    }

    if (InterlockedCompareExchangePointer(
            tensorize_root_signature_.put_void(),
            newRootSignature.get(),
            nullptr) == nullptr) {
      // This thread won the race and just cached the PSO
      newRootSignature.detach();
    }
  }

  return tensorize_root_signature_.get();
}

ID3D12RootSignature* D3DDeviceCache::GetDetensorizeRootSignature() {
  if (detensorize_root_signature_ == nullptr) {
    winrt::com_ptr<ID3D12RootSignature> newRootSignature;
    D3D12_FEATURE_DATA_ROOT_SIGNATURE featureData = {};

    // This is the highest version the sample supports. If CheckFeatureSupport succeeds, the HighestVersion returned will not be greater than this.
    featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_1;

    if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &featureData, sizeof(featureData)))) {
      featureData.HighestVersion = D3D_ROOT_SIGNATURE_VERSION_1_0;
    }

    // Compute root signature.
    {
      CD3DX12_DESCRIPTOR_RANGE1 ranges[2] = {};
      ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
      ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE);

      CD3DX12_ROOT_PARAMETER1 rootParameters[3] = {};
      rootParameters[0].InitAsConstants(4, 0);
      rootParameters[1].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);
      rootParameters[2].InitAsDescriptorTable(1, &ranges[1], D3D12_SHADER_VISIBILITY_ALL);

      CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
      rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

      winrt::com_ptr<ID3DBlob> signature;
      winrt::com_ptr<ID3DBlob> error;
      WINML_THROW_IF_FAILED(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, featureData.HighestVersion, signature.put(), error.put()));
      WINML_THROW_IF_FAILED(device_->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(newRootSignature.put())));
      newRootSignature->SetName(L"Detensorize Rootsignature");
    }

    if (InterlockedCompareExchangePointer(
            detensorize_root_signature_.put_void(),
            newRootSignature.get(),
            nullptr) == nullptr) {
      // This thread won the race and just cached the PSO
      newRootSignature.detach();
    }
  }

  return detensorize_root_signature_.get();
}

ID3D12PipelineState* D3DDeviceCache::GetCachedPipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo, PipelineStateCacheOperation operation) {
  if (cached_pipeline_state[static_cast<int>(type)][static_cast<int>(formatFrom)][static_cast<int>(formatTo)][static_cast<int>(operation)] == nullptr) {
    winrt::com_ptr<ID3D12PipelineState> newPSO;
    if (operation == PipelineStateCacheOperation::kTensorize) {
      newPSO.attach(CreateTensorizePipelineState(type, formatFrom, formatTo));
    } else {
      newPSO.attach(CreateDetensorizePipelineState(type, formatFrom, formatTo));
    }

    if (InterlockedCompareExchangePointer(
            cached_pipeline_state[static_cast<int>(type)][static_cast<int>(formatFrom)][static_cast<int>(formatTo)][static_cast<int>(operation)].put_void(),
            newPSO.get(),
            nullptr) == nullptr) {
      // This thread won the race and just cached the PSO
      newPSO.detach();
    }
  }

  return cached_pipeline_state[static_cast<int>(type)][static_cast<int>(formatFrom)][static_cast<int>(formatTo)][static_cast<int>(operation)].get();
}

ID3D12PipelineState* D3DDeviceCache::CreateTensorizePipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo) {
  static_assert(static_cast<unsigned int>(PipelineStateCacheFormat::kCount) == 3, "PipelineStateCacheFormat changed, update D3DDeviceCache::CreateTensorizePipelineState()");

  const BYTE* shaderBytecode = nullptr;
  uint64_t shaderBytecodeSize = 0;

  switch (formatFrom) {
    case PipelineStateCacheFormat::kBGR8:
    case PipelineStateCacheFormat::kRGB8:
      if (type == PipelineStateCacheType::kFloat32) {
        if (formatTo == PipelineStateCacheFormat::kBGR8) {
          shaderBytecode = float32::g_csSurfaceToTensorBGR8;
          shaderBytecodeSize = sizeof(float32::g_csSurfaceToTensorBGR8);
        } else if (formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float32::g_csSurfaceToTensorRGB8;
          shaderBytecodeSize = sizeof(float32::g_csSurfaceToTensorRGB8);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float32::g_csSurfaceToTensorGRAY8;
          shaderBytecodeSize = sizeof(float32::g_csSurfaceToTensorGRAY8);
        } else {
          assert(false);
        }
      } else if (type == PipelineStateCacheType::kFloat16) {
        if (formatTo == PipelineStateCacheFormat::kBGR8) {
          shaderBytecode = float16::g_csSurfaceToTensorBGR8;
          shaderBytecodeSize = sizeof(float16::g_csSurfaceToTensorBGR8);
        } else if (formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float16::g_csSurfaceToTensorRGB8;
          shaderBytecodeSize = sizeof(float16::g_csSurfaceToTensorRGB8);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float16::g_csSurfaceToTensorGRAY8;
          shaderBytecodeSize = sizeof(float16::g_csSurfaceToTensorGRAY8);
        } else {
          assert(false);
        }
      }
      break;
    case PipelineStateCacheFormat::kGRAY8:
      if (type == PipelineStateCacheType::kFloat32) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          // GRAY -> RGB is the same shader as GRAY -> BGR
          shaderBytecode = float32::g_csSurfaceGRAY8ToTensorBGR8;
          shaderBytecodeSize = sizeof(float32::g_csSurfaceGRAY8ToTensorBGR8);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float32::g_csSurfaceGRAY8ToTensorGRAY8;
          shaderBytecodeSize = sizeof(float32::g_csSurfaceGRAY8ToTensorGRAY8);
        } else {
          assert(false);
        }
      } else if (type == PipelineStateCacheType::kFloat16) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          // GRAY -> RGB is the same shader as GRAY -> BGR
          shaderBytecode = float16::g_csSurfaceGRAY8ToTensorBGR8;
          shaderBytecodeSize = sizeof(float16::g_csSurfaceGRAY8ToTensorBGR8);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float16::g_csSurfaceGRAY8ToTensorGRAY8;
          shaderBytecodeSize = sizeof(float16::g_csSurfaceGRAY8ToTensorGRAY8);
        } else {
          assert(false);
        }
      }
      break;
    default:
      assert(false);
      break;
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
  computePsoDesc.pRootSignature = GetTensorizeRootSignature();
  computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(shaderBytecode, static_cast<size_t>(shaderBytecodeSize));

  winrt::com_ptr<ID3D12PipelineState> pipelineState;
  WINML_THROW_IF_FAILED(device_->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(pipelineState.put())));

  return pipelineState.detach();
}

ID3D12PipelineState* D3DDeviceCache::CreateDetensorizePipelineState(PipelineStateCacheType type, PipelineStateCacheFormat formatFrom, PipelineStateCacheFormat formatTo) {
  static_assert(static_cast<unsigned int>(PipelineStateCacheFormat::kCount) == 3, "PipelineStateCacheFormat changed, update D3DDeviceCache::CreateDetensorizePipelineState()");

  const BYTE* shaderBytecode = nullptr;
  uint64_t shaderBytecodeSize = 0;

  switch (formatFrom) {
    case PipelineStateCacheFormat::kBGR8:
      if (type == PipelineStateCacheType::kFloat32) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float32::g_csTensorBGR8ToSurface;
          shaderBytecodeSize = sizeof(float32::g_csTensorBGR8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float32::g_csTensorBGR8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float32::g_csTensorBGR8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      } else if (type == PipelineStateCacheType::kFloat16) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float16::g_csTensorBGR8ToSurface;
          shaderBytecodeSize = sizeof(float16::g_csTensorBGR8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float16::g_csTensorBGR8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float16::g_csTensorBGR8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      }
      break;
    case PipelineStateCacheFormat::kRGB8:
      if (type == PipelineStateCacheType::kFloat32) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float32::g_csTensorRGB8ToSurface;
          shaderBytecodeSize = sizeof(float32::g_csTensorRGB8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float32::g_csTensorRGB8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float32::g_csTensorRGB8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      } else if (type == PipelineStateCacheType::kFloat16) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          shaderBytecode = float16::g_csTensorRGB8ToSurface;
          shaderBytecodeSize = sizeof(float16::g_csTensorRGB8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float16::g_csTensorRGB8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float16::g_csTensorRGB8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      }
      break;
    case PipelineStateCacheFormat::kGRAY8:
      if (type == PipelineStateCacheType::kFloat32) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          // GRAY -> RGB is the same shader as GRAY -> BGR
          shaderBytecode = float32::g_csTensorGRAY8ToSurface;
          shaderBytecodeSize = sizeof(float32::g_csTensorGRAY8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float32::g_csTensorGRAY8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float32::g_csTensorGRAY8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      } else if (type == PipelineStateCacheType::kFloat16) {
        if (formatTo == PipelineStateCacheFormat::kBGR8 || formatTo == PipelineStateCacheFormat::kRGB8) {
          // GRAY -> RGB is the same shader as GRAY -> BGR
          shaderBytecode = float16::g_csTensorGRAY8ToSurface;
          shaderBytecodeSize = sizeof(float16::g_csTensorGRAY8ToSurface);
        } else if (formatTo == PipelineStateCacheFormat::kGRAY8) {
          shaderBytecode = float16::g_csTensorGRAY8ToSurfaceGRAY8;
          shaderBytecodeSize = sizeof(float16::g_csTensorGRAY8ToSurfaceGRAY8);
        } else {
          assert(false);
        }
      }
      break;
    default:
      assert(false);
      break;
  }

  D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
  computePsoDesc.pRootSignature = GetDetensorizeRootSignature();
  computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(shaderBytecode, static_cast<size_t>(shaderBytecodeSize));

  winrt::com_ptr<ID3D12PipelineState> pipelineState;
  WINML_THROW_IF_FAILED(device_->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(pipelineState.put())));

  return pipelineState.detach();
}

ID3D12Resource* D3DDeviceCache::GetDetensorizeVertexBuffer(_Out_ UINT* vertexBufferSize) {
  if (detensorize_vertex_buffer_ == nullptr) {
    winrt::com_ptr<ID3D12Resource> newResource;
    // Create the vertex buffer.
    // 2 triangles for full screen
    DirectX::XMFLOAT3 triangleVertices[] =
        {
            {-1.0f, 1.0f, 0.0f},
            {1.0f, 1.0f, 0.0f},
            {-1.0f, -1.0f, 0.0f},
            {1.0f, -1.0f, 0.0f},
        };

    assert(sc_vertexBufferSize == sizeof(triangleVertices));

    CD3DX12_HEAP_PROPERTIES heapProp(D3D12_HEAP_TYPE_UPLOAD);
    D3D12_RESOURCE_DESC resourceDiscription = CD3DX12_RESOURCE_DESC::Buffer(sc_vertexBufferSize);
    WINML_THROW_IF_FAILED(device_->CreateCommittedResource(
        &heapProp,
        D3D12_HEAP_FLAG_NONE,
        &resourceDiscription,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(newResource.put())));

    // Copy the triangle data to the vertex buffer.
    UINT8* pVertexDataBegin;
    CD3DX12_RANGE readRange(0, 0);  // We do not intend to read from this resource on the CPU.
    WINML_THROW_IF_FAILED(newResource->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
    memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
    newResource->Unmap(0, nullptr);

    if (InterlockedCompareExchangePointer(
            detensorize_vertex_buffer_.put_void(),
            newResource.get(),
            nullptr) == nullptr) {
      // This thread won the race and just cached the PSO
      newResource.detach();
    }
  }

  *vertexBufferSize = sc_vertexBufferSize;
  return detensorize_vertex_buffer_.get();
}

HANDLE D3DDeviceCache::GetConverterFenceHandle() {
  // Lazily create the fence since we may never need to use it
  if (!converter_fence_) {
    WINML_THROW_IF_FAILED(device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER, IID_PPV_ARGS(converter_fence_.put())));

    HANDLE hSharedFence;
    WINML_THROW_IF_FAILED(device_->CreateSharedHandle(converter_fence_.get(), nullptr, GENERIC_ALL, nullptr, &hSharedFence));

    converter_fence_handle_ = wil::unique_handle(hSharedFence);
  }

  return converter_fence_handle_.get();
}

void D3DDeviceCache::SyncConverterToD3D11Device(_In_ ID3D11Fence* pD3D11Fence) {
  assert(command_queue_ != nullptr);
  assert(pD3D11Fence != nullptr);

  ComPtr<ID3D11Device> spD3D11Device;
  pD3D11Fence->GetDevice(&spD3D11Device);

  ComPtr<ID3D11DeviceContext> spD3D11DeviceContext;
  spD3D11Device->GetImmediateContext(&spD3D11DeviceContext);

  ComPtr<ID3D11DeviceContext4> spD3D11DeviceContext4;
  WINML_THROW_IF_FAILED(spD3D11DeviceContext->QueryInterface(IID_PPV_ARGS(&spD3D11DeviceContext4)));

  UINT64 newfenceValue = converter_fence_value_++;
  WINML_THROW_IF_FAILED(command_queue_->Signal(converter_fence_.get(), newfenceValue));
  WINML_THROW_IF_FAILED(spD3D11DeviceContext4->Wait(pD3D11Fence, newfenceValue));
}

void D3DDeviceCache::SyncD3D11DeviceToConverter(_In_ ID3D11Fence* pD3D11Fence) {
  assert(command_queue_ != nullptr);
  assert(pD3D11Fence != nullptr);

  ComPtr<ID3D11Device> spD3D11Device;
  pD3D11Fence->GetDevice(&spD3D11Device);

  ComPtr<ID3D11DeviceContext> spD3D11DeviceContext;
  spD3D11Device->GetImmediateContext(&spD3D11DeviceContext);

  ComPtr<ID3D11DeviceContext4> spD3D11DeviceContext4;
  WINML_THROW_IF_FAILED(spD3D11DeviceContext->QueryInterface(IID_PPV_ARGS(&spD3D11DeviceContext4)));

  UINT64 newfenceValue = converter_fence_value_++;
  WINML_THROW_IF_FAILED(spD3D11DeviceContext4->Signal(pD3D11Fence, newfenceValue));
  WINML_THROW_IF_FAILED(command_queue_->Wait(converter_fence_.get(), newfenceValue));
}

bool D3DDeviceCache::SharedHandleInitialized() {
  return d3d11_fence_ != nullptr;
}
