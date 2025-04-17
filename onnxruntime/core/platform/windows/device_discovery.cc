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
#include "core/session/abi_devices.h"

//// UsingSetupApi
#include <Windows.h>
#include <SetupAPI.h>
#include <devguid.h>
#include <cfgmgr32.h>
#pragma comment(lib, "setupapi.lib")

//// Using D3D12
// #include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <iostream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

//// Using DXCore. Requires newer Windows SDK than what we target by default.
#define DXCORE_AVAILABLE (NTDDI_VERSION < NTDDI_WIN10_RS5)
#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wil/com.h>

// TODO: Should we define the GUIDs manually as this seems to be the most robust way to find NPUs?
//       What happens if the code runs on a machine that does not have DXCore?
//       If DXCoreCreateAdapterFactory fails gracefully it might be ok to manually define these and always run
// #if !DXCORE_AVAILABLE
// DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);
// DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed);
// #endif

namespace onnxruntime {
namespace {

// device info we accumulate from various sources
struct DeviceInfo {
  OrtHardwareDeviceType type;
  uint32_t vendor_id;
  uint32_t device_id;
  std::wstring vendor;
  std::wstring description;
  std::vector<DWORD> bus_ids;  // assuming could have multiple GPUs that are the same model
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

// key: hardware id with vendor and device id in it
// returns info for display and processor entries. key is (vendor_id << 32 | device_id)
// npus: (vendor_id << 32 | device_id) for devices we think are NPUs from DXCORE
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoSetupApi(const std::unordered_set<uint64_t>& npus) {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  std::array<GUID, 3> guids = {
      GUID_DEVCLASS_DISPLAY,
      GUID_DEVCLASS_PROCESSOR,
      GUID_DEVCLASS_SYSTEM,
  };

  for (auto guid : guids) {
    HDEVINFO devInfo = SetupDiGetClassDevs(&guid, nullptr, nullptr, DIGCF_PRESENT);
    if (devInfo == INVALID_HANDLE_VALUE) {
      return device_info;
    }

    SP_DEVINFO_DATA devData = {};
    devData.cbSize = sizeof(SP_DEVINFO_DATA);

    std::wstring buffer;
    buffer.resize(1024);

    for (DWORD i = 0; SetupDiEnumDeviceInfo(devInfo, i, &devData); ++i) {
      DWORD size = 0;
      DWORD regDataType = 0;

      uint64_t key;
      DeviceInfo* entry = nullptr;

      //// Get hardware ID (contains VEN_xxxx&DEV_xxxx)
      if (SetupDiGetDeviceRegistryPropertyW(devInfo,
                                            &devData,
                                            SPDRP_HARDWAREID,
                                            &regDataType,
                                            (PBYTE)buffer.data(),
                                            (DWORD)buffer.size(),
                                            &size)) {
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
        // won't always have a vendor id from an ACPI entry. need at least a device id to identify the hardware
        if (vendor_id == 0 && device_id == 0) {
          continue;
        }

        key = GetDeviceKey(vendor_id, device_id);

        if (device_info.find(key) == device_info.end()) {
          device_info[key] = {};
        } else {
          if (guid == GUID_DEVCLASS_PROCESSOR) {
            // skip duplicate processor entries as we don't need to accumulate bus numbers for them
            continue;
          }
        }

        entry = &device_info[key];
        entry->vendor_id = vendor_id;
        entry->device_id = device_id;
      } else {
        // need valid ids
        continue;
      }

      // Get device description.
      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_DEVICEDESC, nullptr,
                                            (PBYTE)buffer.data(), (DWORD)buffer.size(), &size)) {
        entry->description = buffer;

        // Should we require the NPU to be found by DXCore or do we want to allow this vague matching?
        // Probably depends on whether we always attempt to run DXCore or not.
        const auto possible_npu = [](const std::wstring& desc) {
          return (desc.find(L"NPU") != std::wstring::npos ||
                  desc.find(L"Neural") != std::wstring::npos ||
                  desc.find(L"AI Engine") != std::wstring::npos ||
                  desc.find(L"VPU") != std::wstring::npos);
        };

        // not 100% accurate. is there a better way?
        uint64_t npu_key = GetDeviceKey(*entry);
        bool is_npu = npus.count(npu_key) > 0 || possible_npu(entry->description);

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

      if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_MFG, nullptr,
                                            (PBYTE)buffer.data(), (DWORD)buffer.size(), &size)) {
        entry->vendor = buffer;
      }

      if (guid != GUID_DEVCLASS_PROCESSOR) {
        DWORD busNumber = 0;
        size = 0;
        if (SetupDiGetDeviceRegistryPropertyW(devInfo, &devData, SPDRP_BUSNUMBER, nullptr,
                                              reinterpret_cast<PBYTE>(&busNumber), sizeof(busNumber), &size)) {
          // push_back in case there are two identical devices. not sure how else to tell them apart
          entry->bus_ids.push_back(busNumber);
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

  IDXGIFactory6* factory = nullptr;
  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
  if (FAILED(hr)) {
    std::cerr << "Failed to create DXGI factory.\n";
    return device_info;
  }

  IDXGIAdapter1* adapter = nullptr;

  // iterate by high-performance GPU preference first
  for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                       IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc))) {
      continue;
    }

    do {
      if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
          (desc.Flags & DXGI_ADAPTER_FLAG_REMOTE) != 0) {
        // software or remote. skip
        break;
      }

      static_assert(sizeof(LUID) == sizeof(uint64_t), "LUID and uint64_t are not the same size");
      uint64_t key = GetLuidKey(desc.AdapterLuid);

      DeviceInfo& info = device_info[key];
      info.type = OrtHardwareDeviceType_GPU;
      info.vendor_id = desc.VendorId;
      info.device_id = desc.DeviceId;
      info.description = std::wstring(desc.Description);

      info.metadata[L"VideoMemory"] = std::to_wstring(desc.DedicatedVideoMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"SystemMemory"] = std::to_wstring(desc.DedicatedSystemMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"SharedSystemMemory"] = std::to_wstring(desc.DedicatedSystemMemory / (1024 * 1024)) + L" MB";
      info.metadata[L"HighPerformanceIndex"] = std::to_wstring(i);
    } while (false);

    adapter->Release();
  }

  factory->Release();

  return device_info;
}

#if DXCORE_AVAILABLE
// returns LUID to DeviceInfo
std::unordered_map<uint64_t, DeviceInfo> GetDeviceInfoDxcore() {
  std::unordered_map<uint64_t, DeviceInfo> device_info;

  // Get all GPUs and NPUs by querying WDDM/MCDM.
  wil::com_ptr<IDXCoreAdapterFactory> adapterFactory;
  if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapterFactory)))) {
    return device_info;
  }

  // Look for devices that expose compute engines
  std::vector<const GUID*> allowedAttributes;
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE);
  allowedAttributes.push_back(&DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML);
  allowedAttributes.push_back(&DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);

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

      // this returns char_t on us-en Windows. assuming it returns wchar_t on other locales but not clear what it
      // does when.
      // The description from SetupApi is wchar_t so assuming we have that and don't need this one.
      //
      // hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      // std::wstring driverDescription;
      // driverDescription.resize(256);
      //// this doesn't seem to return wchar_t
      // if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
      //   hrId = adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(driverDescription),
      //                               &driverDescription);
      //   info.description = driverDescription;
      // }
    }
  }

  return device_info;
}
#endif

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_map<uint64_t, DeviceInfo> luid_to_dxinfo;  // dxcore info. key is luid
  std::unordered_set<uint64_t> npus;                        // NPU devices found in dxcore info
#if DXCORE_AVAILABLE
  luid_to_dxinfo = GetDeviceInfoDxcore();
  for (auto& [luid, device] : luid_to_dxinfo) {
    if (device.type == OrtHardwareDeviceType_NPU) {
      npus.insert(GetDeviceKey(device));
    }
  }
#endif

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

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;  // wstring to string
  const auto device_to_ortdevice = [&converter](
                                       DeviceInfo& device,
                                       std::unordered_map<std::wstring, std::wstring>* extra_metadata = nullptr) {
    OrtHardwareDevice ortdevice{device.type, device.vendor_id, device.device_id, converter.to_bytes(device.vendor)};

    if (device.bus_ids.size() > 0) {
      // use the first bus number. not sure how to handle multiple
      ortdevice.metadata.Add("BusNumber", std::to_string(device.bus_ids.back()).c_str());
      device.bus_ids.pop_back();
    }

    if (!device.description.empty()) {
      ortdevice.metadata.Add("Description", converter.to_bytes(device.description));
    }

    for (auto& [key, value] : device.metadata) {
      ortdevice.metadata.Add(converter.to_bytes(key), converter.to_bytes(value));
    }

    if (extra_metadata) {
      // add any extra metadata from the dxcore info
      for (auto& [key, value] : *extra_metadata) {
        if (device.metadata.find(key) == device.metadata.end()) {
          ortdevice.metadata.Add(converter.to_bytes(key), converter.to_bytes(value));
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
  for (auto& [luid, device] : luid_to_d3d12_info) {
    if (auto it = setupapi_info.find(GetDeviceKey(device)); it != setupapi_info.end()) {
      // use SetupApi info. merge metadata.
      devices.emplace(device_to_ortdevice(it->second, &device.metadata));
    } else {
      // no matching entry in SetupApi. use the dxinfo. no vendor. no BusNumber.
      devices.emplace(device_to_ortdevice(device));
    }
  }

  return devices;
}

}  // namespace onnxruntime
