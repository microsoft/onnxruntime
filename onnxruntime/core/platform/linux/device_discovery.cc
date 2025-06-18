// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

#include "core/common/common.h"
#include "core/common/cpuid_info.h"
#include "core/common/narrow.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

namespace fs = std::filesystem;

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

bool ParseGpuSysfsPath(const fs::path& sysfs_path, size_t& idx) {
  const auto filename = sysfs_path.filename();
  const auto filename_str = std::string_view{filename.native()};

  // Look for a filename matching "cardN". N is a number.
  constexpr std::string_view prefix = "card";
  if (filename_str.find(prefix) != 0) {
    return false;
  }

  size_t parsed_idx{};
  if (!TryParseStringWithClassicLocale<size_t>(filename_str.substr(prefix.size()), parsed_idx)) {
    return false;
  }

  idx = parsed_idx;
  return true;
}

std::string ReadFileContents(const fs::path& file_path) {
  std::ifstream file{file_path};
  ORT_ENFORCE(file, "Failed to open file: ", file_path);
  std::istreambuf_iterator<char> file_begin{file}, file_end{};
  std::string contents(file_begin, file_end);
  return contents;
}

OrtHardwareDevice GetGpuDevice(const fs::path& sysfs_path, size_t idx) {
  OrtHardwareDevice gpu_device{};

  // vendor id
  {
    const auto vendor_file_path = sysfs_path / "device" / "vendor";
    const auto vendor_id_text = utils::TrimString(ReadFileContents(vendor_file_path));
    gpu_device.vendor_id = ParseStringWithClassicLocale<uint32_t>(vendor_id_text);
  }

  // TODO metadata["Discrete"]

  gpu_device.device_id = narrow<uint32_t>(idx);
  gpu_device.type = OrtHardwareDeviceType_GPU;

  return gpu_device;
}

std::vector<OrtHardwareDevice> GetGpuDevices() {
  std::vector<OrtHardwareDevice> gpu_devices{};

  const auto sysfs_class_drm_path = "/sys/class/drm";

  if (!fs::exists(sysfs_class_drm_path)) {
    return gpu_devices;
  }

  for (const auto& dir_item : fs::directory_iterator{sysfs_class_drm_path}) {
    const auto& dir_item_path = dir_item.path();

    if (size_t idx{}; ParseGpuSysfsPath(dir_item_path, idx)) {
      auto gpu_device = GetGpuDevice(dir_item_path, idx);
      gpu_devices.emplace_back(std::move(gpu_device));
    }
  }

  return gpu_devices;
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices;

  // get CPU devices
  devices.emplace(GetCpuDevice());

  // get GPU devices
  {
    auto gpu_devices = GetGpuDevices();
    devices.insert(std::make_move_iterator(gpu_devices.begin()),
                   std::make_move_iterator(gpu_devices.end()));
  }

  // get NPU devices

  return devices;
}
}  // namespace onnxruntime
