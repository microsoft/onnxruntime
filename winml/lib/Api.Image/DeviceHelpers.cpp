// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#if USE_DML
#include <DirectML.h>
#endif USE_DML
#include <d3d11on12.h>
#include <wil/winrt.h>
#include "inc/DeviceHelpers.h"

namespace DeviceHelpers {
constexpr uint32_t c_intelVendorId = 0x8086;
constexpr uint32_t c_nvidiaVendorId = 0x10DE;
constexpr uint32_t c_amdVendorId = 0x1002;
static std::optional<AdapterEnumerationSupport> s_adapterEnumerationSupport;

static bool CheckAdapterFP16Blocked(bool isMcdmAdapter, uint32_t vendorId, uint32_t majorVersion, uint32_t minorVersion) {
  switch (vendorId) {
    case c_intelVendorId: {
      if (isMcdmAdapter) {
        return false;
      }

      // Check Intel GPU driver version
      return (majorVersion < 25) || (majorVersion == 25 && minorVersion < 6574) || (majorVersion == 26 && minorVersion < 6572);
    }
  }
  return false;
}

static void ParseDriverVersion(LARGE_INTEGER& version, uint32_t& majorVersion, uint32_t& minorVersion) {
  majorVersion = HIWORD(version.HighPart);
  minorVersion = LOWORD(version.LowPart);
}

static HRESULT GetDXGIAdapterMetadata(ID3D12Device& device, uint32_t& vendorId, uint32_t& majorVersion, uint32_t& minorVersion) {
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
static HRESULT GetDXCoreAdapterMetadata(ID3D12Device& device, bool& isMcdmAdapter, uint32_t& vendorId, uint32_t& majorVersion, uint32_t& minorVersion) {
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

static HRESULT GetD3D12Device(const winrt::Windows::AI::MachineLearning::LearningModelDevice& device, ID3D12Device** outDevice) {
  _LUID id;
  id.LowPart = device.AdapterId().LowPart;
  id.HighPart = device.AdapterId().HighPart;
  AdapterEnumerationSupport support;
  RETURN_IF_FAILED(GetAdapterEnumerationSupport(&support));

  if (support.has_dxgi) {
    winrt::com_ptr<IDXGIFactory6> spFactory;
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

static HRESULT IsFloat16Blocked(ID3D12Device& device, bool* isBlocked) {
  uint32_t vendorId;
  uint32_t majorVersion;
  uint32_t minorVersion;
  bool isMcdmAdapter;
  *isBlocked = true;
  AdapterEnumerationSupport support;
  RETURN_IF_FAILED(GetAdapterEnumerationSupport(&support));
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

bool IsFloat16Supported(const winrt::Windows::AI::MachineLearning::LearningModelDevice& device) {
  if (device.AdapterId().HighPart == 0 && device.AdapterId().LowPart == 0) {
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
    WINML_THROW_HR_IF_TRUE_MSG(ERROR_NOT_SUPPORTED, true, "IsFloat16Supported is not implemented for WinML only build.");
    return false;
#else
  bool isBlocked;
  if (FAILED(IsFloat16Blocked(*device, &isBlocked)) || isBlocked) {
    return false;
  }
  winrt::com_ptr<IDMLDevice> dmlDevice;
  winrt::check_hresult(DMLCreateDevice(
      device,
      DML_CREATE_DEVICE_FLAG_NONE,
      IID_PPV_ARGS(dmlDevice.put())));

  DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT float16Query = {DML_TENSOR_DATA_TYPE_FLOAT16};
  DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT float16Data = {};

  winrt::check_hresult(dmlDevice->CheckFeatureSupport(
      DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
      sizeof(float16Query),
      &float16Query,
      sizeof(float16Data),
      &float16Data));
  return float16Data.IsSupported;
#endif USE_DML
}

// uses Structured Exception Handling (SEH) to detect for delay load failures of target API.
// You cannot mix and match SEH with C++ exception and object unwinding
// In this case we will catch it, and report up to the caller via HRESULT so our callers can use
// C++ exceptions
template <typename TFunc, typename... TArgs>
static HRESULT RunDelayLoadedApi(TFunc& tfunc, TArgs&&... args) {
  __try {
    return tfunc(std::forward<TArgs>(args)...);
  } __except (GetExceptionCode() == VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
    // this could be ok, just let people know that it failed to load
    return HRESULT_FROM_WIN32(ERROR_MOD_NOT_FOUND);
  }
}

HRESULT GetAdapterEnumerationSupport(AdapterEnumerationSupport* support) {
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

HRESULT GetDXGIHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, IDXGIAdapter1** ppAdapter) {
  winrt::com_ptr<IDXGIFactory6> spFactory;
  RETURN_IF_FAILED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));

  winrt::com_ptr<IDXGIAdapter1> spAdapter;
  UINT i = 0;
  while (spFactory->EnumAdapterByGpuPreference(i, preference, IID_PPV_ARGS(spAdapter.put())) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 pDesc;
    spAdapter->GetDesc1(&pDesc);

    // see here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
    auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
    if (!isSoftwareAdapter && !(isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId)) {
      spAdapter.copy_to(ppAdapter);
      return S_OK;
    }

    spAdapter = nullptr;
    ++i;
  }
  return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
}

#ifdef ENABLE_DXCORE
// Return the first adapter that matches the preference:
// DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE => DXCoreAdapterProperty::IsDetachable
// DXGI_GPU_PREFERENCE_MINIMUM_POWER => DXCoreAdapterProperty::IsIntegrated
HRESULT GetDXCoreHardwareAdapterWithPreference(DXGI_GPU_PREFERENCE preference, IDXCoreAdapter** ppAdapter) {
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

HRESULT CreateD3D11On12Device(ID3D12Device* device12, ID3D11Device** device11) {
  return DeviceHelpers::RunDelayLoadedApi(
      D3D11On12CreateDevice,
      device12,                          // pointer to d3d12 device
      D3D11_CREATE_DEVICE_BGRA_SUPPORT,  // required in order to interop with Direct2D
      nullptr,                           // feature level (defaults to d3d12)
      0,                                 // size of feature levels in bytes
      nullptr,                           // an array of unique command queues for D3D11On12 to use
      0,                                 // size of the command queue array
      0,                                 // D3D12 device node to use
      device11,                          // d3d11 device out param
      nullptr,                           // pointer to d3d11 device context (unused)
      nullptr);                          // pointer to the returned feature level (unused)
}

HRESULT GetGPUPreference(winrt::Windows::AI::MachineLearning::LearningModelDeviceKind deviceKind, DXGI_GPU_PREFERENCE* preference) noexcept {
  switch (deviceKind) {
    case winrt::Windows::AI::MachineLearning::LearningModelDeviceKind::DirectX: {
      *preference = DXGI_GPU_PREFERENCE_UNSPECIFIED;
      return S_OK;
    }
    case winrt::Windows::AI::MachineLearning::LearningModelDeviceKind::DirectXHighPerformance: {
      *preference = DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE;
      return S_OK;
    }
    case winrt::Windows::AI::MachineLearning::LearningModelDeviceKind::DirectXMinPower: {
      *preference = DXGI_GPU_PREFERENCE_MINIMUM_POWER;
      return S_OK;
    }
    default:
      // this should never be reached
      return E_INVALIDARG;
  }
}
}  // namespace DeviceHelpers
