// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <sstream>

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
      oss << "Adding OrtHardwareDevice {vendor_id:0x" << std::hex << ortdevice.vendor_id
          << ", device_id:0x" << ortdevice.device_id
          << ", vendor:" << ortdevice.vendor
          << ", type:" << std::dec << static_cast<int>(ortdevice.type)
          << ", metadata: [";
      for (auto& [key, value] : ortdevice.metadata.entries) {
        oss << key << "=" << value << ", ";
      }
      oss << "]}";
      LOGS_DEFAULT(INFO) << oss.str();
    }

    return discovered_devices;
  }();

  return devices;
}

}  // namespace onnxruntime
