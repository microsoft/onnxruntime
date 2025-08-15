// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include "gtest/gtest.h"

namespace onnxruntime::test {

namespace {

std::vector<OrtHardwareDevice> GetDevicesByType(OrtHardwareDeviceType device_type) {
  std::vector<OrtHardwareDevice> result{};
  const auto& devices = DeviceDiscovery::GetDevices();
  std::copy_if(devices.begin(), devices.end(), std::back_inserter(result),
               [device_type](const OrtHardwareDevice& device) {
                 return device.type == device_type;
               });
  return result;
}

}  // namespace

TEST(DeviceDiscoveryTest, HasCpuDevice) {
  const auto cpu_devices = GetDevicesByType(OrtHardwareDeviceType_CPU);
  ASSERT_GT(cpu_devices.size(), 0);

#if !defined(__wasm__)
  ASSERT_NE(cpu_devices[0].vendor_id, 0);
#endif  // !defined(__WASM__)
}

}  // namespace onnxruntime::test
