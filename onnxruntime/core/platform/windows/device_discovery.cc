// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <unordered_set>

#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wil/com.h>

// TEMPORARY: The CI builds target Windows 10 so do not have these GUIDs.
// This is to make the builds pass so any other issues can be resolved, but needs a real solution prior to checkin.
// these values were added in 10.0.22621.0 as part of DirectXCore API
//
// In theory this #if should be fine, but the QNN ARM64 CI fails even with that applied. Not sure what is happening
// with the NTDII_VERSION value there...
//
// Defining a local GUID instead.
// #if NTDDI_VERSION < NTDDI_WIN10_RS5
//  DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);
//  DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed);
// #endif

#include "core/common/cpuid_info.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
#if !defined(ORT_MINIMAL_BUILD)

namespace {
std::unordered_set<OrtHardwareDevice> GetInferencingDevices() {
  std::unordered_set<OrtHardwareDevice> found_devices;

  const GUID local_DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML = {0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88};
  const GUID local_DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU = {0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0x6, 0xfe, 0x8c, 0x3b, 0x58, 0xed};

  // Get information about the CPU device
  auto vendor = CPUIDInfo::GetCPUIDInfo().GetCPUVendor();

  // Create an OrtDevice for the CPU and add it to the list of found devices
  OrtHardwareDevice cpu_device{OrtHardwareDeviceType_CPU, 0, std::string(vendor), 0};

  found_devices.insert(cpu_device);

  // Get all GPUs and NPUs by querying WDDM/MCDM.
  wil::com_ptr<IDXCoreAdapterFactory> adapterFactory;
  THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&adapterFactory)));

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
      THROW_IF_FAILED(adapterList->GetAdapter(adapterIndex, IID_PPV_ARGS(&adapter)));

      // Ignore software based devices
      if (!adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware)) {
        continue;
      }
      bool isHardware{false};
      if (FAILED(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware)) || !isHardware) {
        continue;
      }

      // Get hardware identifying information
      DXCoreHardwareIDParts idParts = {};
      HRESULT hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareIDParts)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareIDParts, sizeof(idParts), &idParts);
      }

      std::string_view mcdm_vendor_managed;
      if (SUCCEEDED(hrId) && adapter->IsPropertySupported(DXCoreAdapterProperty::HardwareID)) {
        DXCoreHardwareID id;
        hrId = adapter->GetProperty(DXCoreAdapterProperty::HardwareID, sizeof(id), &id);
        if (SUCCEEDED(hrId)) {
          idParts.vendorID = id.vendorID;
          idParts.deviceID = id.deviceID;
          idParts.revisionID = id.revision;
          idParts.subSystemID = id.subSysID;
          idParts.subVendorID = 0;
        }
      }

      // TODO: Get hardware properties given these ID parts and decide what to do with them
      hrId = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
      char driverDescription[256];
      if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
        hrId = adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(driverDescription), &driverDescription);
      }

      // Is this a GPU or NPU
      OrtHardwareDeviceType kind = OrtHardwareDeviceType::OrtHardwareDeviceType_NPU;
      if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
        kind = OrtHardwareDeviceType::OrtHardwareDeviceType_GPU;
      }

      // Insert the device into the set - if not a duplicate
      OrtHardwareDevice found_device(kind, idParts.vendorID, "", idParts.deviceID);
      found_devices.insert(found_device);
    }
  }

  return found_devices;
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices = GetInferencingDevices();
  return devices;
}
#else  // !defined(ORT_MINIMAL_BUILD)
std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  return {};
}
#endif
}  // namespace onnxruntime
