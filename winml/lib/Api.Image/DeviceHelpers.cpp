// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Api.Image/pch.h"

#if USE_DML
#include <DirectML.h>
#endif USE_DML
#include <d3d11on12.h>
#include <wil/winrt.h>
#include "inc/DeviceHelpers.h"
#include "CommonDeviceHelpers.h"
#include "LearningModelDevice.h"

HRESULT IsWarpAdapter(IDXGIAdapter1* pAdapter, bool* isWarpAdapter) {
  DXGI_ADAPTER_DESC1 pDesc;
  RETURN_IF_FAILED(pAdapter->GetDesc1(&pDesc));

  // see here for documentation on filtering WARP adapter:
  // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
  auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
  auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
  auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
  *isWarpAdapter = isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);
  return S_OK;
}

HRESULT
_winml::GetDXGIHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXGIAdapter1** ppAdapter) {
  winrt::com_ptr<IDXGIAdapter1> spAdapter;
  UINT i = 0;
  // Avoids using EnumAdapterByGpuPreference for standard GPU path to enable downlevel to RS3
  if (preference == DXGI_GPU_PREFERENCE::DXGI_GPU_PREFERENCE_UNSPECIFIED) {
    winrt::com_ptr<IDXGIFactory1> spFactory;
    RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));

    while (spFactory->EnumAdapters1(i, spAdapter.put()) != DXGI_ERROR_NOT_FOUND) {
      bool isWarpAdapter = false;
      RETURN_IF_FAILED(IsWarpAdapter(spAdapter.get(), &isWarpAdapter));
      if (!isWarpAdapter) {
        spAdapter.copy_to(ppAdapter);
        return S_OK;
      }
      spAdapter = nullptr;
      ++i;
    }
  } else {
    winrt::com_ptr<IDXGIFactory6> spFactory;
    RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));

    while (spFactory->EnumAdapterByGpuPreference(i, preference, IID_PPV_ARGS(spAdapter.put())) != DXGI_ERROR_NOT_FOUND
    ) {
      bool isWarpAdapter = false;
      RETURN_IF_FAILED(IsWarpAdapter(spAdapter.get(), &isWarpAdapter));
      if (!isWarpAdapter) {
        spAdapter.copy_to(ppAdapter);
        return S_OK;
      }
      spAdapter = nullptr;
      ++i;
    }
  }
  return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

#ifdef ENABLE_DXCORE
// Return the first adapter that matches the preference:
// DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE => DXCoreAdapterProperty::IsDetachable
// DXGI_GPU_PREFERENCE_MINIMUM_POWER => DXCoreAdapterProperty::IsIntegrated
HRESULT _winml::GetDXCoreHardwareAdapterWithPreference(
  DXGI_GPU_PREFERENCE preference, _COM_Outptr_ IDXCoreAdapter** ppAdapter
) {
  winrt::com_ptr<IDXCoreAdapterFactory> spFactory;
  RETURN_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(spFactory.put())));

  winrt::com_ptr<IDXCoreAdapterList> spAdapterList;
  const GUID gpuFilter[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS};
  RETURN_IF_FAILED(spFactory->CreateAdapterList(1, gpuFilter, IID_PPV_ARGS(spAdapterList.put())));

  winrt::com_ptr<IDXCoreAdapter> firstHardwareAdapter;
  bool firstHardwareAdapterFound = false;
  // select first hardware adapter with given preference
  for (uint32_t i = 0; i < spAdapterList->GetAdapterCount(); i++) {
    winrt::com_ptr<IDXCoreAdapter> spCurrAdapter;
    RETURN_IF_FAILED(spAdapterList->GetAdapter(i, IID_PPV_ARGS(spCurrAdapter.put())));

    bool isHardware;
    RETURN_IF_FAILED(spCurrAdapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware));

    if (isHardware) {
      if (preference == DXGI_GPU_PREFERENCE_UNSPECIFIED) {
        spCurrAdapter.copy_to(ppAdapter);
        return S_OK;
      }

      if (!firstHardwareAdapterFound) {
        spCurrAdapter.copy_to(firstHardwareAdapter.put());
        firstHardwareAdapterFound = true;
      }

      if (preference == DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) {
        bool isDetached;
        RETURN_IF_FAILED(spCurrAdapter->GetProperty(DXCoreAdapterProperty::IsDetachable, &isDetached));

        if (isDetached) {
          spCurrAdapter.copy_to(ppAdapter);
          return S_OK;
        }
      } else if (preference == DXGI_GPU_PREFERENCE_MINIMUM_POWER) {
        bool isIntegrated;
        RETURN_IF_FAILED(spCurrAdapter->GetProperty(DXCoreAdapterProperty::IsIntegrated, &isIntegrated));

        if (isIntegrated) {
          spCurrAdapter.copy_to(ppAdapter);
          return S_OK;
        }
      }
    }
  }
  // If a preference match wasn't found, return the first hardware adapter in the list
  RETURN_HR_IF(HRESULT_FROM_WIN32(ERROR_NOT_FOUND), !firstHardwareAdapterFound);
  firstHardwareAdapter.copy_to(ppAdapter);
  return S_OK;
}
#endif

HRESULT _winml::CreateD3D11On12Device(ID3D12Device* device12, ID3D11Device** device11) {
  return CommonDeviceHelpers::RunDelayLoadedApi(
    D3D11On12CreateDevice,
    device12,  // pointer to d3d12 device
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,  // required in order to interop with Direct2D
    nullptr,  // feature level (defaults to d3d12)
    0,  // size of feature levels in bytes
    nullptr,  // an array of unique command queues for D3D11On12 to use
    0,  // size of the command queue array
    0,  // D3D12 device node to use
    device11,  // d3d11 device out param
    nullptr,  // pointer to d3d11 device context (unused)
    nullptr
  );  // pointer to the returned feature level (unused)
}

HRESULT _winml::GetGPUPreference(winml::LearningModelDeviceKind deviceKind, DXGI_GPU_PREFERENCE* preference) noexcept {
  switch (deviceKind) {
    case winml::LearningModelDeviceKind::DirectX: {
      *preference = DXGI_GPU_PREFERENCE_UNSPECIFIED;
      return S_OK;
    }
    case winml::LearningModelDeviceKind::DirectXHighPerformance: {
      *preference = DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
      return S_OK;
    }
    case winml::LearningModelDeviceKind::DirectXMinPower: {
      *preference = DXGI_GPU_PREFERENCE_MINIMUM_POWER;
      return S_OK;
    }
    default:
      // this should never be reached
      return E_INVALIDARG;
  }
}
