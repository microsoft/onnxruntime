// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "LearningModelDevice.h"

#include <D3d11_4.h>
#include <d3d11on12.h>
#include "D3DDeviceCache.h"

#include "ConverterResourceStore.h"

namespace winrt::Windows::AI::MachineLearning::implementation {
/*static*/ void LearningModelDevice::DllUnload() {
}

Windows::Graphics::DisplayAdapterId LearningModelDevice::AdapterId() try {
  Windows::Graphics::DisplayAdapterId id;
  id.LowPart = m_deviceCache->GetDeviceLuid().LowPart;
  id.HighPart = m_deviceCache->GetDeviceLuid().HighPart;
  return id;
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(Windows::AI::MachineLearning::LearningModelDeviceKind const& deviceKind) try : m_deviceCache(std::make_unique<D3DDeviceCache>(deviceKind)) {
  m_deviceKind = deviceKind;
  m_isCpuDevice = m_deviceKind == LearningModelDeviceKind::Cpu || m_deviceKind == LearningModelDeviceKind::Default;
  if (m_isCpuDevice) {
    assert(m_deviceCache->GetD3D12Device() == nullptr);
  }
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device) try : m_deviceCache(std::make_unique<D3DDeviceCache>(device)) {
  m_deviceKind = LearningModelDeviceKind::DirectX;
  m_isCpuDevice = false;
}
WINML_CATCH_ALL

LearningModelDevice::LearningModelDevice(ID3D12CommandQueue* queue) try : m_deviceKind(LearningModelDeviceKind::DirectX),
                                                                          m_deviceCache(std::make_unique<D3DDeviceCache>(queue)) {
  m_isCpuDevice = false;
}
WINML_CATCH_ALL

LearningModelDevice::~LearningModelDevice() {
  // needed for shared ptr destruction
}

Windows::AI::MachineLearning::LearningModelDevice LearningModelDevice::CreateFromDirect3D11Device(Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device) try {
  return make<LearningModelDevice>(device);
}
WINML_CATCH_ALL

std::shared_ptr<::Windows::AI::MachineLearning::ConverterResourceStore> LearningModelDevice::TensorizerStore() {
  if (m_tensorizerStore == nullptr) {
    m_tensorizerStore = ::Windows::AI::MachineLearning::ConverterResourceStore::Create(5);
  }
  return m_tensorizerStore;
}

std::shared_ptr<::Windows::AI::MachineLearning::ConverterResourceStore> LearningModelDevice::DetensorizerStore() {
  if (m_detensorizerStore == nullptr) {
    m_detensorizerStore = ::Windows::AI::MachineLearning::ConverterResourceStore::Create(5);
  }
  return m_detensorizerStore;
}

winml::LearningModelDeviceKind
LearningModelDevice::GetDeviceKind() {
  return m_deviceKind;
}

bool LearningModelDevice::IsCpuDevice() {
  return m_isCpuDevice;
}

const LUID&
LearningModelDevice::GetDeviceLuid() {
  return m_deviceCache->GetDeviceLuid();
}

D3DDeviceCache*
LearningModelDevice::GetD3DDeviceCache() {
  return m_deviceCache.get();
}

wgdx::Direct3D11::IDirect3DDevice
LearningModelDevice::Direct3D11Device() try {
  return m_deviceCache->GetWinrtDevice();
}
WINML_CATCH_ALL

ID3D12Device*
LearningModelDevice::GetD3DDevice() {
  return m_deviceCache->GetD3D12Device();
}

ID3D12CommandQueue*
LearningModelDevice::GetDeviceQueue() {
  return m_deviceCache->GetCommandQueue();
}

STDMETHODIMP
LearningModelDevice::SetMetacommandsEnabled(boolean enabled) {
  m_areMetacommandsEnabled = enabled;
  return S_OK;
}

bool LearningModelDevice::MetacommandsEnabled() {
  return m_areMetacommandsEnabled;
}

STDMETHODIMP_(boolean)
LearningModelDevice::SharedHandleInitialized() {
  return m_deviceCache->SharedHandleInitialized();
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation

namespace winrt::Windows::AI::MachineLearning::factory_implementation {
// copied from cppwinrt magic to create abi wrappers.   Need to do it this way
// since peeps underneath (like the constructor) will throw
HRESULT __stdcall LearningModelDevice::CreateFromD3D12CommandQueue(
    ID3D12CommandQueue* queue,
    IUnknown** device) noexcept {
  try {
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, queue, "Failed to create LearningModelDevice. Ivalid argument queue.");
    WINML_THROW_HR_IF_NULL_MSG(E_INVALIDARG, device, "Failed to create LearningModelDevice. Ivalid argument device.");

    auto machineLearningDevice = make<implementation::LearningModelDevice>(queue);
    *device = machineLearningDevice.as<IUnknown>().detach();
    return S_OK;
  }
  WINML_CATCH_ALL_COM
}
}  // namespace winrt::Windows::AI::MachineLearning::factory_implementation
