// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <sys/utsname.h>
#include <TargetConditionals.h>

#include "core/common/logging/logging.h"

namespace onnxruntime {

namespace {

constexpr auto kApplePciVendorId = 0x106B;
constexpr auto kAppleVendorName = "Apple";

std::vector<OrtHardwareDevice> GetGpuDevices() {
  std::vector<OrtHardwareDevice> result{};

  // For now, we assume the existence of one GPU if it is a Mac with Apple Silicon.
  // TODO support iOS
  // TODO support Intel Macs which may have more than one GPU
#if TARGET_OS_OSX && TARGET_CPU_ARM64
  {
    OrtHardwareDevice gpu_device{};
    gpu_device.type = OrtHardwareDeviceType_GPU;
    gpu_device.vendor_id = kApplePciVendorId;
    gpu_device.vendor = kAppleVendorName;

    result.emplace_back(std::move(gpu_device));
  }
#endif  // TARGET_OS_OSX && TARGET_CPU_ARM64

  return result;
}

bool HasAppleNeuralEngine() {
  // Copied from onnxruntime/core/providers/coreml/builders/helper.cc:HasNeuralEngine().
  bool has_apple_neural_engine = false;

  struct utsname system_info;
  uname(&system_info);
  LOGS_DEFAULT(VERBOSE) << "Current Apple hardware info: " << system_info.machine;

#if TARGET_OS_IPHONE
  // utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
  // Since Neural Engine is only available for use on A12 and later, major device version in the
  // identifier is checked for these models:
  // A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
  // A12X: iPad Pro - 3rd Gen (8,1)
  // For more information, see https://www.theiphonewiki.com/wiki/Models
  size_t str_len = strnlen(system_info.machine, onnxruntime::kMaxStrLen);
  if (str_len > 4 && strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    has_apple_neural_engine = major_version >= 8;  // There are no device between iPad 8 and 11.
  } else if (str_len > 6 && strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    has_apple_neural_engine = major_version >= 11;
  }
#elif TARGET_OS_OSX && TARGET_CPU_ARM64
  // Only Mac with arm64 CPU (Apple Silicon) has ANE.
  has_apple_neural_engine = true;
#endif  // #if TARGET_OS_IPHONE

  return has_apple_neural_engine;
}

std::vector<OrtHardwareDevice> GetNpuDevices() {
  std::vector<OrtHardwareDevice> result{};

  if (HasAppleNeuralEngine()) {
    OrtHardwareDevice npu_device{};
    npu_device.type = OrtHardwareDeviceType_NPU;
    npu_device.vendor_id = kApplePciVendorId;
    npu_device.vendor = kAppleVendorName;

    result.emplace_back(std::move(npu_device));
  }

  return result;
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices;

  // get CPU devices
  devices.insert(GetCpuDeviceFromCPUIDInfo());

  // get GPU devices
  {
    auto gpu_devices = GetGpuDevices();
    devices.insert(gpu_devices.begin(), gpu_devices.end());
  }

  // get NPU devices
  {
    auto npu_devices = GetNpuDevices();
    devices.insert(npu_devices.begin(), npu_devices.end());
  }

  return devices;
}
}  // namespace onnxruntime
