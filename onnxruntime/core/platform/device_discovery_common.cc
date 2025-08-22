// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains platform-agnostic device discovery implementation.

#include "core/platform/device_discovery.h"

#include <sstream>

#include "core/common/cpuid_info.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

const std::unordered_set<OrtHardwareDevice>& DeviceDiscovery::GetDevices() {
  // assumption: devices don't change. we assume the machine must be shutdown to change cpu/gpu/npu devices.
  // technically someone could disable/enable a device in a running OS. we choose not to add complexity to support
  // that scenario.
  static std::unordered_set<OrtHardwareDevice> devices = []() {
    auto discovered_devices = DiscoverDevicesForPlatform();

    // log discovered devices
    for (const auto& ortdevice : discovered_devices) {
      std::ostringstream oss;
      oss << "Discovered OrtHardwareDevice {vendor_id:0x" << std::hex << ortdevice.vendor_id
          << ", device_id:0x" << ortdevice.device_id
          << ", vendor:" << ortdevice.vendor
          << ", type:" << std::dec << static_cast<int>(ortdevice.type)
          << ", metadata: [";
      for (auto& [key, value] : ortdevice.metadata.Entries()) {
        oss << key << "=" << value << ", ";
      }
      oss << "]}";
      LOGS_DEFAULT(INFO) << oss.str();
    }

    return discovered_devices;
  }();

  return devices;
}

OrtHardwareDevice DeviceDiscovery::GetCpuDeviceFromCPUIDInfo() {
  const auto& cpuid_info = CPUIDInfo::GetCPUIDInfo();

  OrtHardwareDevice cpu_device{};
  cpu_device.vendor = cpuid_info.GetCPUVendor();
  cpu_device.vendor_id = cpuid_info.GetCPUVendorId();
  cpu_device.device_id = 0;
  cpu_device.type = OrtHardwareDeviceType_CPU;

  return cpu_device;
}

}  // namespace onnxruntime
