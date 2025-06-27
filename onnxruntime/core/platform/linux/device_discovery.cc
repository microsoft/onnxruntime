// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

#include "core/common/common.h"
#include "core/common/cpuid_info.h"
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

std::vector<fs::path> DetectGpuSysfsPaths() {
  const fs::path sysfs_class_drm_path = "/sys/class/drm";

  if (!fs::exists(sysfs_class_drm_path)) {
    return {};
  }

  const auto detect_card_path = [](const fs::path& sysfs_path, size_t& idx) -> bool {
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
  };

  std::vector<fs::path> gpu_sysfs_paths{};
  for (const auto& dir_item : fs::directory_iterator{sysfs_class_drm_path}) {
    auto dir_item_path = dir_item.path();

    if (size_t idx{}; detect_card_path(dir_item_path, idx)) {
      gpu_sysfs_paths.emplace_back(std::move(dir_item_path));
    }
  }

  return gpu_sysfs_paths;
}

std::string ReadFileContents(const fs::path& file_path) {
  std::ifstream file{file_path};
  ORT_ENFORCE(file, "Failed to open file: ", file_path);
  std::istreambuf_iterator<char> file_begin{file}, file_end{};
  std::string contents(file_begin, file_end);
  return contents;
}

template <typename ValueType>
ValueType ReadValueFromFile(const fs::path& file_path) {
  const auto file_text = utils::TrimString(ReadFileContents(file_path));
  return ParseStringWithClassicLocale<ValueType>(file_text);
}

OrtHardwareDevice GetGpuDevice(const fs::path& sysfs_path) {
  OrtHardwareDevice gpu_device{};

  // vendor id
  {
    const auto vendor_id_path = sysfs_path / "device" / "vendor";
    gpu_device.vendor_id = ReadValueFromFile<uint32_t>(vendor_id_path);
  }

  // TODO vendor name

  // device id
  {
    const auto device_id_path = sysfs_path / "device" / "device";
    gpu_device.device_id = ReadValueFromFile<uint32_t>(device_id_path);
  }

  // TODO metadata? e.g., is the device discrete?

  gpu_device.type = OrtHardwareDeviceType_GPU;

  return gpu_device;
}

std::vector<OrtHardwareDevice> GetGpuDevices() {
  const auto gpu_sysfs_paths = DetectGpuSysfsPaths();
  std::vector<OrtHardwareDevice> gpu_devices{};
  gpu_devices.reserve(gpu_sysfs_paths.size());

  for (const auto& gpu_sysfs_path : gpu_sysfs_paths) {
    auto gpu_device = GetGpuDevice(gpu_sysfs_path);
    gpu_devices.emplace_back(std::move(gpu_device));
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
  // TODO figure out how to discover these

  return devices;
}
}  // namespace onnxruntime
