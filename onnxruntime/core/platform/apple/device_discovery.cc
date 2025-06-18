// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

namespace onnxruntime {
std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices;

  // TODO implement

  // get CPU devices

  // get GPU devices

  // get NPU devices

  return devices;
}
}  // namespace onnxruntime
