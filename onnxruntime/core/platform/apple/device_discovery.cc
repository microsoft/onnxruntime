// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include "core/common/cpuid_info.h"

namespace onnxruntime {

namespace {

OrtHardwareDevice GetCpuDevice() {
  const auto& cpuid_info = CPUIDInfo::GetCPUIDInfo();

  OrtHardwareDevice cpu_device{};
  cpu_device.vendor = cpuid_info.GetCPUVendor();
  cpu_device.vendor_id = cpuid_info.GetCPUVendorId();
  cpu_device.device_id = 0;
  cpu_device.type = OrtHardwareDeviceType_CPU;

  return cpu_device;
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices;

  // get CPU devices
  devices.insert(GetCpuDevice());

  // TODO

  // get GPU devices

  // get NPU devices

  return devices;
}
}  // namespace onnxruntime
