// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Common/inc/pch.h"
#if USE_DML
// TODO (pavignol): Remove
#include "core/providers/dml/DirectML2.h"
// #include <DirectML.h>
#endif USE_DML
#include "inc/CommonDeviceHelpers.h"
#include <d3d11on12.h>
#include <wil/winrt.h>
#include "LearningModelDevice.h"

namespace {
constexpr uint32_t c_intelVendorId = 0x8086;
constexpr uint32_t c_nvidiaVendorId = 0x10DE;
constexpr uint32_t c_amdVendorId = 0x1002;

bool CheckAdapterFP16Blocked(bool isMcdmAdapter, uint32_t vendorId, uint32_t majorVersion, uint32_t minorVersion) {
  switch (vendorId) {
    case c_intelVendorId: {
      if (isMcdmAdapter) {
        return false;
      }

      // Check Intel GPU driver version
      return (majorVersion < 25) || (majorVersion == 25 && minorVersion < 6574) ||
        (majorVersion == 26 && minorVersion < 6572);
    }
  }
  return false;
}

void ParseDriverVersion(LARGE_INTEGER& version, uint32_t& majorVersion, uint32_t& minorVersion) {
  majorVersion = HIWORD(version.HighPart);
  minorVersion = LOWORD(version.LowPart);
}

HRESULT
GetDXGIAdapterMetadata(ID3D12Device& device, uint32_t& vendorId, uint32_t& majorVersion, uint32_t& minorVersion) {
  winrt::com_ptr<IDXGIFactory4> spFactory;
  RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));

  winrt::com_ptr<IDXGIAdapter> spAdapter;
  RETURN_IF_FAILED(spFactory->EnumAdapterByLuid(device.GetAdapterLuid(), IID_PPV_ARGS(spAdapter.put())));

  DXGI_ADAPTER_DESC adapterDesc = {};
  RETURN_IF_FAILED(spAdapter->GetDesc(&adapterDesc));

  LARGE_INTEGER driverVersion;
  RETURN_IF_FAILED(spAdapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driverVersion));

  vendorId = adapterDesc.VendorId;
  ParseDriverVersion(driverVersion, majorVersion, minorVersion);
  return S_OK;
}

#ifdef ENABLE_DXCORE
HRESULT GetDXCoreAdapterMetadata(
  ID3D12Device& device, bool& isMcdmAdapter, uint32_t& vendorId, uint32_t& majorVersion, uint32_t& minorVersion
) {
  winrt::com_ptr<IDXCoreAdapterFactory> spFactory;
  RETURN_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(spFactory.put())));

  winrt::com_ptr<IDXCoreAdapter> spAdapter;
  RETURN_IF_FAILED(spFactory->GetAdapterByLuid(device.GetAdapterLuid(), IID_PPV_ARGS(spAdapter.put())));

  if (spAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE) &&
        (!(spAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS) ||
           spAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D11_GRAPHICS)))) {
    isMcdmAdapter = true;
  } else {
    isMcdmAdapter = false;
  }

  DXCoreHardwareID hardwareId;
  RETURN_IF_FAILED(spAdapter->GetProperty(DXCoreAdapterProperty::HardwareID, &hardwareId));
  vendorId = hardwareId.vendorID;

  uint64_t rawDriverVersion;
  RETURN_IF_FAILED(spAdapter->GetProperty(DXCoreAdapterProperty::DriverVersion, &rawDriverVersion));

  LARGE_INTEGER driverVersion;
  driverVersion.QuadPart = static_cast<LONGLONG>(rawDriverVersion);
  ParseDriverVersion(driverVersion, majorVersion, minorVersion);
  return S_OK;
}
#endif

HRESULT GetD3D12Device(const winml::LearningModelDevice& device, ID3D12Device** outDevice) {
  _LUID id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  CommonDeviceHelpers::AdapterEnumerationSupport support;
  RETURN_IF_FAILED(GetAdapterEnumerationSupport(&support));

  if (support.has_dxgi) {
    winrt::com_ptr<IDXGIFactory4> spFactory;
    RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));

    winrt::com_ptr<IDXGIAdapter1> spAdapter;
    RETURN_IF_FAILED(spFactory->EnumAdapterByLuid(id, IID_PPV_ARGS(spAdapter.put())));
    RETURN_IF_FAILED(D3D12CreateDevice(spAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(outDevice)));
  }
#ifdef ENABLE_DXCORE
  if (support.has_dxgi == false) {
    winrt::com_ptr<IDXCoreAdapterFactory> spFactory;
    RETURN_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(spFactory.put())));

    winrt::com_ptr<IDXCoreAdapter> spAdapter;
    RETURN_IF_FAILED(spFactory->GetAdapterByLuid(id, IID_PPV_ARGS(spAdapter.put())));
    RETURN_IF_FAILED(D3D12CreateDevice(spAdapter.get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(outDevice)));
  }
#endif
  return S_OK;
}

HRESULT IsFloat16Blocked(ID3D12Device& device, bool* isBlocked) {
  uint32_t vendorId;
  uint32_t majorVersion;
  uint32_t minorVersion;
  bool isMcdmAdapter;
  *isBlocked = true;
  CommonDeviceHelpers::AdapterEnumerationSupport support;
  RETURN_IF_FAILED(CommonDeviceHelpers::GetAdapterEnumerationSupport(&support));
#ifdef ENABLE_DXCORE
  if (support.has_dxcore) {
    RETURN_IF_FAILED(GetDXCoreAdapterMetadata(device, isMcdmAdapter, vendorId, majorVersion, minorVersion));
    *isBlocked = CheckAdapterFP16Blocked(isMcdmAdapter, vendorId, majorVersion, minorVersion);
    return S_OK;
  }
#endif
  RETURN_IF_FAILED(GetDXGIAdapterMetadata(device, vendorId, majorVersion, minorVersion));
  isMcdmAdapter = false;
  *isBlocked = CheckAdapterFP16Blocked(isMcdmAdapter, vendorId, majorVersion, minorVersion);
  return S_OK;
}
}  // namespace

namespace CommonDeviceHelpers {
constexpr uint32_t c_intelVendorId = 0x8086;
constexpr uint32_t c_nvidiaVendorId = 0x10DE;
constexpr uint32_t c_amdVendorId = 0x1002;

bool IsFloat16Supported(const winml::LearningModelDevice& device) {
  auto adapterId = device.AdapterId();
  if (!adapterId.HighPart && !adapterId.LowPart) {
    // CPU device
    return true;
  }
  winrt::com_ptr<ID3D12Device> d3d12Device;
  if (FAILED(GetD3D12Device(device, d3d12Device.put()))) {
    return false;
  }
  return IsFloat16Supported(d3d12Device.get());
}

bool IsFloat16Supported(ID3D12Device* device) {
#ifndef USE_DML
  throw winrt::hresult_error(E_NOTIMPL, L"IsFloat16Supported is not implemented for WinML only build.");
#else
  bool isBlocked;
  if (FAILED(IsFloat16Blocked(*device, &isBlocked)) || isBlocked) {
    return false;
  }
  winrt::com_ptr<IDMLDevice> dmlDevice;
  winrt::check_hresult(DMLCreateDevice(device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(dmlDevice.put())));

  DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT float16Query = {DML_TENSOR_DATA_TYPE_FLOAT16};
  DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT float16Data = {};

  winrt::check_hresult(dmlDevice->CheckFeatureSupport(
    DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(float16Query), &float16Query, sizeof(float16Data), &float16Data
  ));
  return float16Data.IsSupported;
#endif
}

HRESULT GetAdapterEnumerationSupport(AdapterEnumerationSupport* support) {
  static std::optional<AdapterEnumerationSupport> s_adapterEnumerationSupport;
  if (!s_adapterEnumerationSupport.has_value()) {
    // check for support, starting with DXGI
    winrt::com_ptr<IDXGIFactory4> dxgiFactory;
#ifdef ENABLE_DXCORE
    winrt::com_ptr<IDXCoreAdapterFactory> dxcoreFactory;
    // necessary because DXCoreCreateAdapterFactory is overloaded
    HRESULT(WINAPI * pDxCoreTestFunc)
    (REFIID, void**) = DXCoreCreateAdapterFactory;
#endif
    AdapterEnumerationSupport adapterEnumerationSupport = {};

    if (SUCCEEDED(RunDelayLoadedApi(CreateDXGIFactory1, IID_PPV_ARGS(dxgiFactory.put())))) {
      adapterEnumerationSupport.has_dxgi = true;
    }
#ifdef ENABLE_DXCORE
    if (SUCCEEDED(RunDelayLoadedApi(pDxCoreTestFunc, IID_PPV_ARGS(dxcoreFactory.put())))) {
      adapterEnumerationSupport.has_dxcore = true;
    }
#endif

    s_adapterEnumerationSupport = adapterEnumerationSupport;

    if (!(adapterEnumerationSupport.has_dxgi || adapterEnumerationSupport.has_dxcore)) {
      return TYPE_E_CANTLOADLIBRARY;
    }
  }
  *support = s_adapterEnumerationSupport.value();
  return S_OK;
}

}  // namespace CommonDeviceHelpers
