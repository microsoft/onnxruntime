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
#include "core/platform/env.h"
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

struct DriverInfo {
  std::wstring driver_versions;
  std::wstring driver_names;

  void AddDevice(const std::wstring& driver_version, const std::wstring& driver_name) {
    if (!driver_version.empty()) {
      if (!driver_versions.empty()) {
        driver_versions += L", ";
      }
      driver_versions += driver_version;
    }
    if (!driver_name.empty()) {
      if (!driver_names.empty()) {
        driver_names += L", ";
      }
      driver_names += driver_name;
    }
  }
};

bool IsHexString(const std::wstring& str) {
  for (const wchar_t& c : str) {
    if (!((c >= L'0' && c <= L'9') || (c >= L'A' && c <= L'F') || (c >= L'a' && c <= L'f'))) {
      return false;
    }
  }
  return true;
}

// Converts a wide string ACPI (up to 4 characters) representing a hardware ID component from into a uint32_t.
// e.g., "QCOM" from "VEN_QCOM". The conversion is done in a little-endian manner, meaning the first character
// of the string becomes the least significant byte of the integer, and the fourth character
// becomes the most significant byte.
uint32_t AcpiWStringToUint32Id(const std::wstring& vendor_name) {
  uint32_t vendor_id = 0;
  for (size_t i = 0; i < 4 && i < vendor_name.size(); ++i) {
    // For little-endian, place each character at the appropriate byte position
    // First character goes into lowest byte, last character into highest byte
    vendor_id |= static_cast<unsigned char>(vendor_name[i] & 0xFF) << (i * 8);
  }
  return vendor_id;
}

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
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoSetupApi(const std::unordered_set<uint64_t>& npus,
                                                               bool& have_remote_display_adapter) {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  const GUID local_DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML = {0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88};
  const GUID local_DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU = {0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed};
  const GUID local_GUID_DEVCLASS_COMPUTEACCELERATOR = {0xf01a9d53, 0x3ff6, 0x48d2, 0x9f, 0x97, 0xc8, 0xa7, 0x00, 0x4b, 0xe1, 0x0c};

  std::unordered_map<OrtHardwareDeviceType, DriverInfo> device_version_info;

  std::array<GUID, 3> guids = {
      GUID_DEVCLASS_DISPLAY,
      GUID_DEVCLASS_PROCESSOR,
      local_GUID_DEVCLASS_COMPUTEACCELERATOR,
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
        uint32_t vendor_id = 0;
        uint32_t device_id = 0;

        // PCI\VEN_xxxx&DEV_yyyy&...
        // ACPI\VEN_xxxx&DEV_yyyy&... if we're lucky.
        // ACPI values seem to be very inconsistent, so we check fairly carefully and always require a device id.
        const auto get_id = [](bool is_pci, const std::wstring& hardware_id, const std::wstring& prefix) -> uint32_t {
          if (auto idx = hardware_id.find(prefix); idx != std::wstring::npos) {
            auto id = hardware_id.substr(idx + prefix.size(), 4);

            if (id.size() == 4) {
              if (is_pci || IsHexString(id)) {
                // PCI entries have hex numbers. ACPI might.
                return static_cast<uint32_t>(std::stoul(id, nullptr, 16));
              } else {
                // ACPI can have things like "VEN_QCOM". Fallback to using this conversion where the characters
                // are converted in little-endian order.
                return AcpiWStringToUint32Id(id);
              }
            }
          }

          return 0;
        };

        const bool is_pci = std::wstring(buffer, 3) == std::wstring(L"PCI");

        if (guid == GUID_DEVCLASS_PROCESSOR) {
          // Processor ID should come from CPUID mapping.
          vendor_id = CPUIDInfo::GetCPUIDInfo().GetCPUVendorId();
        } else {
          vendor_id = get_id(is_pci, buffer, L"VEN_");
        }

        device_id = get_id(is_pci, buffer, L"DEV_");

        // Won't always have a vendor id from an ACPI entry.  ACPI is not defined for this purpose.
        if (vendor_id == 0 && device_id == 0) {
          static const std::wstring remote_display_adapter_id(L"RdpIdd_IndirectDisplay");
          if (guid == GUID_DEVCLASS_DISPLAY && remote_display_adapter_id == buffer) {
            have_remote_display_adapter = true;
          }

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
        } else if (guid == local_GUID_DEVCLASS_COMPUTEACCELERATOR) {
          if (!is_npu) {
            // we're only iterating compute accelerator devices to look for NPUs so drop anything else
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

      // Generate telemetry event to log the GPU and NPU driver name and version.
      if (entry->type == OrtHardwareDeviceType_CPU) {
        // Skip processor entries for telemetry.
        continue;
      }

      // Open the device's driver registry key
      HKEY dev_reg_key = SetupDiOpenDevRegKey(devInfo, &devData,
                                              DICS_FLAG_GLOBAL,
                                              0,
                                              DIREG_DRV,
                                              KEY_READ);

      if (dev_reg_key != INVALID_HANDLE_VALUE) {
        // Query the "DriverVersion" string
        std::wstring driver_version_str;
        wchar_t driver_version[256];
        DWORD str_size = sizeof(driver_version);
        DWORD type = 0;
        if (RegQueryValueExW(dev_reg_key, L"DriverVersion",
                             nullptr, &type,
                             reinterpret_cast<LPBYTE>(driver_version),
                             &str_size) == ERROR_SUCCESS &&
            type == REG_SZ) {
          // Ensure proper null termination of a string retrieved from the Windows Registry API.
          driver_version[(str_size / sizeof(wchar_t)) - 1] = 0;
          driver_version_str = driver_version;
        }
        RegCloseKey(dev_reg_key);
        device_version_info[entry->type].AddDevice(driver_version_str, entry->description);
      }
    }

    SetupDiDestroyDeviceInfoList(devInfo);
  }

  // Log driver information for GPUs and NPUs
  const Env& env = Env::Default();
  for (const auto& [type, info] : device_version_info) {
    if (!info.driver_versions.empty() || !info.driver_names.empty()) {
      const std::string_view driver_class = (type == OrtHardwareDeviceType_GPU) ? "GPU" : "NPU";
      env.GetTelemetryProvider().LogDriverInfoEvent(driver_class, info.driver_names, info.driver_versions);
    }
  }

  return device_info;
}

// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoD3D12(bool have_remote_display_adapter) {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  ComPtr<IDXGIFactory6> factory;
  if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) {
    std::cerr << "Failed to create DXGI factory.\n";
    return device_info;
  }

  UINT num_adapters = 0;

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

    info.metadata[L"LUID"] = std::to_wstring(key);
    info.metadata[L"DxgiAdapterNumber"] = std::to_wstring(i);
    info.metadata[L"DxgiVideoMemory"] = std::to_wstring(desc.DedicatedVideoMemory / (1024 * 1024)) + L" MB";

    ++num_adapters;
  }

  // iterate by high-performance GPU preference to add that info.
  UINT cur_adapter = 0;
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
    if (it == device_info.end()) {
      continue;
    }

    DeviceInfo& info = it->second;

    // try and drop the Microsoft Remote Display Adapter. it does not have the DXGI_ADAPTER_FLAG_SOFTWARE flag set
    // and the vendor id, device id and description are the same as the real device. the LUID is different to the real
    // device.
    // Assumption: it will have the worst performance index of the devices we're considering so we only check the
    //             last adapter
    if (num_adapters > 1 && have_remote_display_adapter && cur_adapter == num_adapters - 1) {
      ComPtr<IDXGIOutput> output;
      if (adapter->EnumOutputs(0, &output) == DXGI_ERROR_NOT_FOUND) {
        // D3D_DRIVER_TYPE_WARP. Software based or disabled adapter.
        // An adapter can be disabled in an RDP session. e.g. integrated GPU is disabled if there's a discrete GPU

        // if we have seen this vendor_id+device_id combination with a different LUID before we drop it.
        if (std::any_of(device_info.begin(), device_info.end(),
                        [key, &info](const auto& entry) {
                          const auto& entry_info = entry.second;
                          return key != entry.first &&
                                 info.vendor_id == entry_info.vendor_id &&
                                 info.device_id == entry_info.device_id;
                        })) {
          device_info.erase(key);
          continue;
        }
      }
    }

    info.metadata[L"DxgiHighPerformanceIndex"] = std::to_wstring(i);

    ++cur_adapter;
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

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts) ||
          FAILED(adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts))) {
        continue;  // also need valid ids
      }

      DeviceInfo& info = device_info[key];
      info.vendor_id = idParts.vendorID;
      info.device_id = idParts.deviceID;
      info.metadata[L"LUID"] = std::to_wstring(key);

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

DeviceInfo GetDeviceInfoCPUID() {
  DeviceInfo cpu_info{};
  cpu_info.type = OrtHardwareDeviceType_CPU;

  auto& cpuinfo = CPUIDInfo::GetCPUIDInfo();
  cpu_info.vendor_id = cpuinfo.GetCPUVendorId();

  std::string_view cpuid_vendor = cpuinfo.GetCPUVendor();
  cpu_info.vendor = std::wstring(cpuid_vendor.begin(), cpuid_vendor.end());
  cpu_info.description = cpu_info.vendor;

  return cpu_info;
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

  // setupapi_info. key is vendor_id+device_id
  bool have_remote_display_adapter = false;  // set if we see the RdpIdd_IndirectDisplay hardware ID.
  std::unordered_map<uint64_t, DeviceInfo> setupapi_info = GetDeviceInfoSetupApi(npus, have_remote_display_adapter);

  // d3d12 info. key is luid
  std::unordered_map<uint64_t, DeviceInfo> luid_to_d3d12_info = GetDeviceInfoD3D12(have_remote_display_adapter);

  // Ensure we have at least one CPU
  bool found_cpu = false;
  for (auto& [key, device] : setupapi_info) {
    if (device.type == OrtHardwareDeviceType_CPU) {
      found_cpu = true;
      break;
    }
  }

  // If no CPU was found via SetupApi, add one from CPUID
  if (!found_cpu) {
    DeviceInfo device = GetDeviceInfoCPUID();
    uint64_t key = GetDeviceKey(device);
    setupapi_info[key] = std::move(device);
  }

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
