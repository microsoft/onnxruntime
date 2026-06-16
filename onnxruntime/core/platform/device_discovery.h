// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "core/session/abi_devices.h"

namespace onnxruntime {

class DeviceDiscovery {
 public:
  static const std::unordered_set<OrtHardwareDevice>& GetDevices();

 private:
  DeviceDiscovery() = default;

  // platform specific code implements this method
  static std::unordered_set<OrtHardwareDevice> DiscoverDevicesForPlatform();

  // Gets a CPU device by querying `CPUIDInfo`.
  static OrtHardwareDevice GetCpuDeviceFromCPUIDInfo();
};

}  // namespace onnxruntime
