// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api/pch/pch.h"
#include "LearningModelDevice.h"

#include <D3d11_4.h>
#include <d3d11on12.h>
#include "D3DDeviceCache.h"

#include "ConverterResourceStore.h"

namespace WINMLP {

/*static*/ void LearningModelDevice::DllUnload() {
}

wg::DisplayAdapterId LearningModelDevice::AdapterId() try {
  wg::DisplayAdapterId id;
  id.LowPart = m_deviceCache->GetDeviceLuid().LowPart;
  id.HighPart = m_deviceCache->GetDeviceLuid().HighPart;
  return id;
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(winml::LearningModelDeviceKind const& deviceKind) try
  : m_deviceCache(std::make_unique<_winml::D3DDeviceCache>(deviceKind)) {
  m_deviceKind = deviceKind;
  telemetry_helper.SetLearningModelDeviceKind(static_cast<int>(deviceKind));
  m_isCpuDevice = m_deviceKind == LearningModelDeviceKind::Cpu || m_deviceKind == LearningModelDeviceKind::Default;
  if (m_isCpuDevice) {
    assert(m_deviceCache->GetD3D12Device() == nullptr);
  }
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(wgdx::Direct3D11::IDirect3DDevice const& device) try
  : m_deviceCache(std::make_unique<_winml::D3DDeviceCache>(device)) {
  m_deviceKind = LearningModelDeviceKind::DirectX;
  m_isCpuDevice = false;
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(ID3D12CommandQueue* queue) try
  : m_deviceKind(LearningModelDeviceKind::DirectX),
    m_deviceCache(std::make_unique<_winml::D3DDeviceCache>(queue)) {
  m_isCpuDevice = false;
}
WINML_CATCH_ALL

LearningModelDevice::~LearningModelDevice() {
  // needed for shared ptr destruction
}

winml::LearningModelDevice LearningModelDevice::CreateFromDirect3D11Device(
  wgdx::Direct3D11::IDirect3DDevice const& device
) try {
  return make<LearningModelDevice>(device);
}
WINML_CATCH_ALL

std::shared_ptr<_winml::ConverterResourceStore> LearningModelDevice::TensorizerStore() {
  std::call_once(m_tensorizerStoreInitialized, [this]() {
    m_tensorizerStore = _winml::ConverterResourceStore::Create(5);
  });
  return m_tensorizerStore;
}

std::shared_ptr<_winml::ConverterResourceStore> LearningModelDevice::DetensorizerStore() {
  std::call_once(m_detensorizerStoreInitialized, [this]() {
    m_detensorizerStore = _winml::ConverterResourceStore::Create(5);
  });
  return m_detensorizerStore;
}

winml::LearningModelDeviceKind LearningModelDevice::GetDeviceKind() {
  return m_deviceKind;
}

bool LearningModelDevice::IsCpuDevice() {
  return m_isCpuDevice;
}

const LUID& LearningModelDevice::GetDeviceLuid() {
  return m_deviceCache->GetDeviceLuid();
}

_winml::D3DDeviceCache* LearningModelDevice::GetD3DDeviceCache() {
  return m_deviceCache.get();
}

wgdx::Direct3D11::IDirect3DDevice LearningModelDevice::Direct3D11Device() try {
  return m_deviceCache->GetWinrtDevice();
}
WINML_CATCH_ALL

ID3D12Device* LearningModelDevice::GetD3DDevice() {
  return m_deviceCache->GetD3D12Device();
}

ID3D12CommandQueue* LearningModelDevice::GetDeviceQueue() {
  return m_deviceCache->GetCommandQueue();
}

STDMETHODIMP
LearningModelDevice::SetMetacommandsEnabled(boolean enabled) {
  m_areMetacommandsEnabled = (enabled != 0);
  return S_OK;
}

bool LearningModelDevice::MetacommandsEnabled() {
  return m_areMetacommandsEnabled;
}

STDMETHODIMP_(boolean)
LearningModelDevice::SharedHandleInitialized() {
  return m_deviceCache->SharedHandleInitialized();
}

STDMETHODIMP
LearningModelDevice::GetThreadPool(_winml::IThreading** thread_pool) {
  m_threadPool.copy_to(thread_pool);
  return S_OK;
}

STDMETHODIMP
LearningModelDevice::CacheThreadPool(_winml::IThreading* thread_pool) {
  m_threadPool.copy_from(thread_pool);
  return S_OK;
}

uint32_t LearningModelDevice::NumberOfIntraOpThreads() {
  if (IsCpuDevice()) {
    return std::thread::hardware_concurrency();
  } else {
    // GPU sessions should not rely on intra op threads.
    // Creating a large thread pool is unnecessary and wasteful, and can cause
    // thread competition in the process.
    return 1;
  }
}

bool LearningModelDevice::AllowSpinning() {
  if (IsCpuDevice()) {
    return true;
  } else {
    // GPU sessions should not run operators on cpu threads.
    // CPU threads created should not spin, as it will drain cpu resources unnecessarily.
    return false;
  }
}

}  // namespace WINMLP

namespace WINML::factory_implementation {

// copied from cppwinrt magic to create abi wrappers.   Need to do it this way
// since peeps underneath (like the constructor) will throw
HRESULT __stdcall LearningModelDevice::CreateFromD3D12CommandQueue(
  ID3D12CommandQueue* queue, IUnknown** device
) noexcept {
  try {
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, queue, "Failed to create LearningModelDevice. Invalid argument queue.");
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, device, "Failed to create LearningModelDevice. Invalid argument device.");

    auto machineLearningDevice = make<implementation::LearningModelDevice>(queue);
    *device = machineLearningDevice.as<IUnknown>().detach();
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}

}  // namespace WINML::factory_implementation
