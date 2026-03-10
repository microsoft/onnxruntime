// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include "gtest/gtest.h"

#if !defined(ORT_MINIMAL_BUILD) && !defined(_GAMING_XBOX)
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

#if defined(CPUINFO_SUPPORTED)
  ASSERT_NE(cpu_devices[0].vendor_id, 0);
#endif  // defined(CPUINFO_SUPPORTED)
}

TEST(DeviceDiscoveryTest, GpuDevicesHaveValidProperties) {
  const auto gpu_devices = GetDevicesByType(OrtHardwareDeviceType_GPU);

  // GPU detection should not crash. If GPUs are present, validate their properties.
  for (const auto& gpu_device : gpu_devices) {
    EXPECT_NE(gpu_device.vendor_id, 0u);
    // Note: device_id may be 0 on some platforms (e.g., Apple Silicon) where it is not populated.
  }
}

}  // namespace onnxruntime::test
#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(_GAMING_XBOX)
