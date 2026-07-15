// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

namespace onnxruntime {

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  // This is a default implementation.
  // We assume that there is a CPU device and do not attempt to discover anything else.

  std::unordered_set<OrtHardwareDevice> devices{};

  devices.emplace(GetCpuDeviceFromCPUIDInfo());

  return devices;
}

}  // namespace onnxruntime
