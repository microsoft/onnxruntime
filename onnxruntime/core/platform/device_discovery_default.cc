// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

namespace onnxruntime {

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  // This is a default implementation which does not try to discover anything.
  return {};
}

}  // namespace onnxruntime
