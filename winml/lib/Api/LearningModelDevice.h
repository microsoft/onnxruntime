// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "LearningModelDevice.g.h"

#include "iengine.h"

namespace _winml {
class ConverterResourceStore;
class D3DDeviceCache;
}  // namespace _winml

namespace WINMLP {

struct LearningModelDevice : LearningModelDeviceT<LearningModelDevice, IMetacommandsController, IDeviceFenceValidator> {
 public:
  LearningModelDevice() = delete;

  LearningModelDevice(winml::LearningModelDeviceKind const& deviceKind);

  LearningModelDevice(wgdx::Direct3D11::IDirect3DDevice const& device);

  LearningModelDevice(ID3D12CommandQueue* queue);

  ~LearningModelDevice();

  wg::DisplayAdapterId AdapterId();

  static winml::LearningModelDevice CreateFromDirect3D11Device(wgdx::Direct3D11::IDirect3DDevice const& device);

  // internal:
  STDMETHOD(SetMetacommandsEnabled)
  (boolean enabled) final;

  // internal:
  STDMETHOD_(boolean, SharedHandleInitialized)
  ();

  STDMETHOD(GetThreadPool)(_winml::IThreading** thread_pool);

  STDMETHOD(CacheThreadPool)(_winml::IThreading* thread_pool);

  // internal:

  winml::LearningModelDeviceKind GetDeviceKind();

  bool MetacommandsEnabled();

  bool IsCpuDevice();

  const LUID& GetDeviceLuid();

  _winml::D3DDeviceCache* GetD3DDeviceCache();

  wgdx::Direct3D11::IDirect3DDevice Direct3D11Device();

  ID3D12Device* GetD3DDevice();

  ID3D12CommandQueue* GetDeviceQueue();

  static void DllUnload();

  std::shared_ptr<_winml::ConverterResourceStore> TensorizerStore();

  std::shared_ptr<_winml::ConverterResourceStore> DetensorizerStore();

  uint32_t NumberOfIntraOpThreads();

  bool AllowSpinning();

 private:
  // stores the device kind that was originally chosen in the constructor
  winml::LearningModelDeviceKind m_deviceKind;
  // if the user asked us to run on the cpu, or asked us to choose and we chose cpu
  bool m_isCpuDevice;
  bool m_areMetacommandsEnabled = true;
  std::shared_ptr<_winml::ConverterResourceStore> m_detensorizerStore;
  std::once_flag m_detensorizerStoreInitialized;
  std::shared_ptr<_winml::ConverterResourceStore> m_tensorizerStore;
  std::once_flag m_tensorizerStoreInitialized;

  std::unique_ptr<_winml::D3DDeviceCache> m_deviceCache;

  com_ptr<_winml::IThreading> m_threadPool;
};
}  // namespace WINMLP

namespace WINML::factory_implementation {
struct LearningModelDevice
  : LearningModelDeviceT<LearningModelDevice, implementation::LearningModelDevice, ILearningModelDeviceFactoryNative> {
  HRESULT __stdcall CreateFromD3D12CommandQueue(ID3D12CommandQueue* queue, IUnknown** device) noexcept final;
};
}  // namespace WINML::factory_implementation
