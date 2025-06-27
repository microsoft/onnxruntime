// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <array>
#include <cassert>
#include <codecvt>
#include <locale>
#include <string>
#include <unordered_set>

#include "core/common/cpuid_info.h"
#include "core/common/logging/logging.h"
#include "core/session/abi_devices.h"

//// For SetupApi info
#include <Windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cfgmgr32.h>
#pragma comment(lib, "setupapi.lib")

//// For D3D12 info
// #include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

//// For DXCore info.
#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wil/com.h>

#include "core/common/cpuid_info.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
// unsupported in minimal build. also needs xbox specific handling to be implemented.
#if !defined(ORT_MINIMAL_BUILD) && !defined(_GAMING_XBOX)
namespace {

// device info we accumulate from various sources
struct DeviceInfo {
  OrtHardwareDeviceType type;
  uint32_t vendor_id;
  uint32_t device_id;
  std::wstring vendor;
  std::wstring description;
  std::unordered_map<std::wstring, std::wstring> metadata;
};

uint64_t GetDeviceKey(uint32_t vendor_id, uint32_t device_id) {
  return (uint64_t(vendor_id) << 32) | device_id;
}

uint64_t GetDeviceKey(const DeviceInfo& device_info) {
  return GetDeviceKey(device_info.vendor_id, device_info.device_id);
}

uint64_t GetLuidKey(LUID luid) {
  return (uint64_t(luid.HighPart) << 32) | luid.LowPart;
}

// returns info for display and processor entries. key is (vendor_id << 32 | device_id)
// npus: (vendor_id << 32 | device_id) for devices we think are NPUs from DXCORE
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoSetupApi(const std::unordered_set<uint64_t>& npus) {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  const GUID local_DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML = {0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88};
  const GUID local_DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU = {0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed};

  std::array<GUID, 3> guids = {
      GUID_DEVCLASS_DISPLAY,
      GUID_DEVCLASS_PROCESSOR,
      GUID_DEVCLASS_SYSTEM,
  };

  for (auto guid : guids) {
    HDEVINFO devInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfo == INVALID_HANDLE_VALUE) {
      continue;
    }

    SP_DEVINFO_DATA devData = {};
    devData.cbSize = sizeof(SP_DEVINFO_DATA);

    WCHAR buffer[1024];

    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devData); ++i) {
      DWORD size = 0;
      DWORD regDataType = 0;

      uint64_t key;
      DeviceInfo* entry = nullptr;

      //// Get hardware ID (contains VEN_xxxx&DEV_xxxx)
      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_HARDWAREID, &regDataType,
                                            (PBYTE)buffer, sizeof(buffer), &size)) {
        // PCI\VEN_xxxx&DEV_yyyy&...
        // ACPI\VEN_xxxx&DEV_yyyy&... if we're lucky.
        // ACPI values seem to be very inconsistent, so we check fairly carefully and always require a device id.
        const auto get_id = [](const std::wstring& hardware_id, const std::wstring& prefix) -> uint32_t {
          if (auto idx = hardware_id.find(prefix); idx != std::wstring::npos) {
            auto id = hardware_id.substr(idx + prefix.size(), 4);
            if (std::all_of(id.begin(), id.end(), iswxdigit)) {
              return std::stoul(id, nullptr, 16);
            }
          }

          return 0;
        };

        uint32_t vendor_id = get_id(buffer, L"VEN_");
        uint32_t device_id = get_id(buffer, L"DEV_");

        // Processor ID should come from CPUID mapping.
        if (vendor_id == 0 && guid == GUID_DEVCLASS_PROCESSOR) {
          vendor_id = CPUIDInfo::GetCPUIDInfo().GetCPUVendorId();
        }

        // Won't always have a vendor id from an ACPI entry.  ACPI is not defined for this purpose.
        if (vendor_id == 0 && device_id == 0) {
          continue;
        }

        key = GetDeviceKey(vendor_id, device_id);

        if (device_info.find(key) == device_info.end()) {
          device_info[key] = {};
        } else {
          if (guid == GUID_DEVCLASS_PROCESSOR) {
            // skip duplicate processor entries
            continue;
          }
        }

        entry = &device_info[key];
        entry->vendor_id = vendor_id;
        entry->device_id = device_id;
        // put the first hardware id string in the metadata. ignore the other lines. not sure if this is of value.
        // entry->metadata.emplace(L"SPDRP_HARDWAREID", std::wstring(buffer, wcslen(buffer)));
      } else {
        // need valid ids
        continue;
      }

      // Use the friendly name if available.
      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_FRIENDLYNAME, nullptr,
                                            (PBYTE)buffer, sizeof(buffer), &size)) {
        entry->description = std::wstring{buffer};
      }

      // Set type using the device description to try and infer an NPU.
      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_DEVICEDESC, nullptr,
                                            (PBYTE)buffer, sizeof(buffer), &size)) {
        std::wstring desc{buffer};

        // For now, require dxcore to identify an NPU.
        // If we want to try and infer it from the description this _may_ work but is untested.
        // const auto possible_npu = [](const std::wstring& desc) {
        //  return (desc.find(L"NPU") != std::wstring::npos ||
        //           desc.find(L"Neural") != std::wstring::npos ||
        //           desc.find(L"AI Engine") != std::wstring::npos ||
        //           desc.find(L"VPU") != std::wstring::npos);
        // };

        // use description if no friendly name
        if (entry->description.empty()) {
          entry->description = desc;
        }

        uint64_t npu_key = GetDeviceKey(*entry);
        bool is_npu = npus.count(npu_key) > 0;  // rely on dxcore to determine if something is an NPU

        if (guid == GUID_DEVCLASS_DISPLAY) {
          entry->type = OrtHardwareDeviceType_GPU;
        } else if (guid == GUID_DEVCLASS_PROCESSOR) {
          entry->type = is_npu ? OrtHardwareDeviceType_NPU : OrtHardwareDeviceType_CPU;
        } else if (guid == GUID_DEVCLASS_SYSTEM) {
          if (!is_npu) {
            // we're only iterating system devices to look for NPUs so drop anything else
            device_info.erase(key);
            continue;
          }

          entry->type = OrtHardwareDeviceType_NPU;
        } else {
          // unknown device type
          continue;
        }
      } else {
        // can't set device type
        continue;
      }

      if (entry->type == OrtHardwareDeviceType_CPU) {
        // get 12 byte string from CPUID. easier for a user to match this if they are explicitly picking a device.
        std::string_view cpuid_vendor = CPUIDInfo::GetCPUIDInfo().GetCPUVendor();
        entry->vendor = std::wstring(cpuid_vendor.begin(), cpuid_vendor.end());
      }

      if (entry->vendor.empty()) {
        if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_MFG, nullptr,
                                              (PBYTE)buffer, sizeof(buffer), &size)) {
          entry->vendor = std::wstring(buffer, wcslen(buffer));
        }
      }
    }

    SetupDiDestroyDeviceInfoList(devInfo);
  }

  return device_info;
}

// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoD3D12() {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  ComPtr<IDXGIFactory6> factory;
  if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
    std::cerr << "Failed to create DXGI factory.\n";
    return device_info;
  }

  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc))) {
      continue;
    }

    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
        (desc.Flags & DXGI_ADAPTER_FLAG_REMOTE) != 0) {
      // software or remote. skip
      continue;
    }

    static_assert(sizeof(LUID) == sizeof(uint64_t), "LUID and uint64_t are not the same size");
    uint64_t key = GetLuidKey(desc.AdapterLuid);

    DeviceInfo& info = device_info[key];
    info.type = OrtHardwareDeviceType_GPU;
    info.vendor_id = desc.VendorId;
    info.device_id = desc.DeviceId;
    info.description = std::wstring(desc.Description);

    info.metadata[L"DxgiAdapterNumber"] = std::to_wstring(i);
    info.metadata[L"DxgiVideoMemory"] = std::to_wstring(desc.DedicatedVideoMemory / (1024 * 1024)) + L" MB";
  }

  // iterate by high-performance GPU preference to add that info
  for (UINT i = 0; factory->EnumAdapterByGpuPreference(
                       i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                       IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf())) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc))) {
      continue;
    }

    uint64_t key = GetLuidKey(desc.AdapterLuid);

    auto it = device_info.find(key);
    if (it != device_info.end()) {
      DeviceInfo& info = it->second;
      info.metadata[L"DxgiHighPerformanceIndex"] = std::to_wstring(i);
    }
  }

  return device_info;
}

typedef HRESULT(WINAPI* PFN_DXCoreCreateAdapterFactory)(REFIID riid, void** ppvFactory);

// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoDxcore() {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  // Load dxcore.dll. We do this manually so there's not a hard dependency on dxcore which is newer.
  wil::unique_hmodule dxcore_lib{LoadLibraryExW(L"dxcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)};
  if (!dxcore_lib) {
    LOGS_DEFAULT(INFO) << "Failed to load dxcore.dll. Expected on older Windows version that do not support dxcore.";
    return device_info;
  }

  auto pfnDXCoreCreateAdapterFactory = reinterpret_cast<PFN_DXCoreCreateAdapterFactory>(
      GetProcAddress(dxcore_lib.get(), "DXCoreCreateAdapterFactory"));

  if (!pfnDXCoreCreateAdapterFactory) {
    // this isn't expected to fail so ERROR not WARNING
    LOGS_DEFAULT(ERROR) << "Failed to get DXCoreCreateAdapterFactory function address.";
    return device_info;
  }

  // Get all GPUs and NPUs by querying WDDM/MCDM.
  wil::com_ptr<IDXCoreAdapterFactory> adapterFactory;
  if (FAILED(pfnDXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapterFactory)))) {
    return device_info;
  }

  // NOTE: These GUIDs requires a newer Windows SDK than what we target by default.
  // They were added in 10.0.22621.0 as part of DirectXCore API
  // To workaround this we define a local copy of the values. On an older Windows machine they won't match anything.
  static const GUID local_DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML = {0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88};
  static const GUID local_DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU = {0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed};

  // Look for devices that expose compute engines
  std::vector<const GUID*> allowedAttributes;
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
  allowedAttributes.push_back(&local_DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
  allowedAttributes.push_back(&local_DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);

  // These attributes are not OR'd.  Have to query one at a time to get a full view.
  for (const auto& hwAttribute : allowedAttributes) {
    wil::com_ptr<IDXCoreAdapterList> adapterList;
    if (FAILED(adapterFactory->CreateAdapterList(1, hwAttribute, IID_PPV_ARGS(&adapterList)))) {
      continue;
    }

    const uint32_t adapterCount{adapterList->GetAdapterCount()};
    for (uint32_t adapterIndex = 0; adapterIndex < adapterCount; adapterIndex++) {
      wil::com_ptr<IDXCoreAdapter> adapter;
      if (FAILED(adapterList->GetAdapter(adapterIndex, IID_PPV_ARGS(&adapter)))) {
        continue;
      }

      // Ignore software based devices
      bool isHardware{false};
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware)) || !isHardware) {
        continue;
      }

      static_assert(sizeof(LUID) == sizeof(uint64_t), "LUID and uint64_t are not the same size");
      LUID luid;  // really a LUID but we only need it to skip duplicated devices
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, sizeof(luid), &luid))) {
        continue;  // need this for the key
      }

      uint64_t key = GetLuidKey(luid);
      if (device_info.find(key) != device_info.end()) {
        // already found this device
        continue;
      }

      DeviceInfo& info = device_info[key];

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts))) {
        continue;  // also need valid ids
      }

      info.vendor_id = idParts.vendorID;
      info.device_id = idParts.deviceID;

      // Is this a GPU or NPU
      if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        info.type = OrtHardwareDeviceType::OrtHardwareDeviceType_GPU;
      } else {
        info.type = OrtHardwareDeviceType::OrtHardwareDeviceType_NPU;
      }

      bool is_integrated = false;
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsIntegrated) &&
          SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated, sizeof(is_integrated),
                                         &is_integrated))) {
        info.metadata[L"Discrete"] = is_integrated ? L"0" : L"1";
      }
    }
  }

  return device_info;
}
}  // namespace

// Get devices from various sources and combine them into a single set of devices.
// For CPU we use setupapi data.
// For GPU we augment the d3d12 and dxcore data with the setupapi data.
// For NPU we augment the dxcore data with the setupapi data.
std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  // dxcore info. key is luid
  std::unordered_map<uint64_t, DeviceInfo> luid_to_dxinfo = GetDeviceInfoDxcore();
  std::unordered_set<uint64_t> npus;  // NPU devices found in dxcore info
  for (auto& [luid, device] : luid_to_dxinfo) {
    if (device.type == OrtHardwareDeviceType_NPU) {
      npus.insert(GetDeviceKey(device));
    }
  }

  // d3d12 info. key is luid
  std::unordered_map<uint64_t, DeviceInfo> luid_to_d3d12_info = GetDeviceInfoD3D12();
  // setupapi_info. key is vendor_id+device_id
  std::unordered_map<uint64_t, DeviceInfo> setupapi_info = GetDeviceInfoSetupApi(npus);

  // add dxcore info for any devices that are not in d3d12.
  // d3d12 info is more complete and has a good description and metadata.
  // dxcore has 'Discrete' in metadata so add that if found
  for (auto& [luid, device] : luid_to_dxinfo) {
    if (auto it = luid_to_d3d12_info.find(luid); it != luid_to_d3d12_info.end()) {
      // merge the metadata
      const auto& dxcore_metadata = device.metadata;
      auto& d3d12_metadata = it->second.metadata;

      for (auto& [key, value] : dxcore_metadata) {
        if (d3d12_metadata.find(key) == d3d12_metadata.end()) {
          d3d12_metadata[key] = value;
        }
      }
    } else {
      luid_to_d3d12_info[luid] = device;
    }
  }

  // our log output to std::wclog breaks with UTF8 chars that are not supported by the current code page.
  // e.g. (TM) symbol. that stops ALL logging working on at least arm64.
  // safest way to avoid that is to keep it to single byte chars.
  // process the OrtHardwareDevice values this way so it can be safely logged.
  // only the 'description' metadata is likely to be affected and that is mainly for diagnostic purposes.
  const auto to_safe_string = [](const std::wstring& wstr) -> std::string {
    std::string str(wstr.size(), ' ');
    std::transform(wstr.begin(), wstr.end(), str.begin(), [](wchar_t wchar) {
      if (wchar >= 0 && wchar <= 127) {
        return static_cast<char>(wchar);
      }
      return ' ';
    });
    return str;
  };

  const auto device_to_ortdevice = [&to_safe_string](
                                       DeviceInfo& device,
                                       std::unordered_map<std::wstring, std::wstring>* extra_metadata = nullptr) {
    OrtHardwareDevice ortdevice{device.type, device.vendor_id, device.device_id, to_safe_string(device.vendor)};

    if (!device.description.empty()) {
      ortdevice.metadata.Add("Description", to_safe_string(device.description));
    }

    for (auto& [key, value] : device.metadata) {
      ortdevice.metadata.Add(to_safe_string(key), to_safe_string(value));
    }

    if (extra_metadata) {
      // add any extra metadata from the dxcore info
      for (auto& [key, value] : *extra_metadata) {
        if (device.metadata.find(key) == device.metadata.end()) {
          ortdevice.metadata.Add(to_safe_string(key), to_safe_string(value));
        }
      }
    }

    std::ostringstream oss;
    oss << "Adding OrtHardwareDevice {vendor_id:0x" << std::hex << ortdevice.vendor_id
        << ", device_id:0x" << ortdevice.device_id
        << ", vendor:" << ortdevice.vendor
        << ", type:" << std::dec << static_cast<int>(ortdevice.type)
        << ", metadata: [";
    for (auto& [key, value] : ortdevice.metadata.entries) {
      oss << key << "=" << value << ", ";
    }

    oss << "]}" << std::endl;
    LOGS_DEFAULT(INFO) << oss.str();

    return ortdevice;
  };

  // create final set of devices with info from everything
  std::unordered_set<OrtHardwareDevice> devices;

  // CPU from SetupAPI
  for (auto& [idstr, device] : setupapi_info) {
    OrtHardwareDevice ort_device;
    if (device.type == OrtHardwareDeviceType_CPU) {
      // use the SetupApi info as-is
      devices.emplace(device_to_ortdevice(device));
    }
  }

  // filter GPU/NPU to devices in combined d3d12/dxcore info.
  // TODO: If we found what we think is an NPU from SetupApi should we include it?
  for (auto& [luid, device] : luid_to_d3d12_info) {
    if (auto it = setupapi_info.find(GetDeviceKey(device)); it != setupapi_info.end()) {
      // use SetupApi info. merge metadata.
      devices.emplace(device_to_ortdevice(it->second, &device.metadata));
    } else {
      // no matching entry in SetupApi. use the dxinfo. will be missing vendor name and UI_NUMBER
      devices.emplace(device_to_ortdevice(device));
    }
  }

  return devices;
}
#else  // !defined(ORT_MINIMAL_BUILD) && !defined(_GAMING_XBOX)
std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  return {};
}
#endif
}  // namespace onnxruntime
