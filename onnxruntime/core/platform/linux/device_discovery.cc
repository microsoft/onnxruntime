// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string_view>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

namespace fs = std::filesystem;

namespace onnxruntime {

namespace {

Status ErrorCodeToStatus(const std::error_code& ec) {
  if (!ec) {
    return Status::OK();
  }

  return Status{common::StatusCategory::ONNXRUNTIME, common::StatusCode::FAIL,
                MakeString("Error: std::error_code with category name: ", ec.category().name(),
                           ", value: ", ec.value(), ", message: ", ec.message())};
}

struct GpuSysfsPathInfo {
  size_t card_idx;
  fs::path path;
};

Status DetectGpuSysfsPaths(std::vector<GpuSysfsPathInfo>& gpu_sysfs_paths_out) {
  std::error_code error_code{};
  const fs::path sysfs_class_drm_path = "/sys/class/drm";
  const bool sysfs_class_drm_path_exists = fs::exists(sysfs_class_drm_path, error_code);
  ORT_RETURN_IF_ERROR(ErrorCodeToStatus(error_code));

  if (!sysfs_class_drm_path_exists) {
    gpu_sysfs_paths_out = std::vector<GpuSysfsPathInfo>{};
    return Status::OK();
  }

  const auto detect_card_path = [](const fs::path& sysfs_path, size_t& card_idx) -> bool {
    const auto filename = sysfs_path.filename();
    const auto filename_str = std::string_view{filename.native()};

    // Look for a filename matching "cardN". N is a number.
    constexpr std::string_view prefix = "card";
    if (filename_str.find(prefix) != 0) {
      return false;
    }

    size_t parsed_card_idx{};
    if (!TryParseStringWithClassicLocale<size_t>(filename_str.substr(prefix.size()), parsed_card_idx)) {
      return false;
    }

    card_idx = parsed_card_idx;
    return true;
  };

  std::vector<GpuSysfsPathInfo> gpu_sysfs_paths{};

  auto dir_iterator = fs::directory_iterator{sysfs_class_drm_path, error_code};
  ORT_RETURN_IF_ERROR(ErrorCodeToStatus(error_code));

  for (const auto& dir_item : dir_iterator) {
    const auto& dir_item_path = dir_item.path();

    if (size_t card_idx{}; detect_card_path(dir_item_path, card_idx)) {
      GpuSysfsPathInfo path_info{};
      path_info.card_idx = card_idx;
      path_info.path = dir_item_path;
      gpu_sysfs_paths.emplace_back(std::move(path_info));
    }
  }

  gpu_sysfs_paths_out = std::move(gpu_sysfs_paths);
  return Status::OK();
}

Status ReadFileContents(const fs::path& file_path, std::string& contents) {
  std::ifstream file{file_path};
  ORT_RETURN_IF_NOT(file, "Failed to open file: ", file_path);
  std::istreambuf_iterator<char> file_begin{file}, file_end{};
  contents.assign(file_begin, file_end);
  return Status::OK();
}

template <typename ValueType>
Status ReadValueFromFile(const fs::path& file_path, ValueType& value) {
  std::string file_text{};
  ORT_RETURN_IF_ERROR(ReadFileContents(file_path, file_text));
  file_text = utils::TrimString(file_text);
  return ParseStringWithClassicLocale<ValueType>(file_text, value);
}

Status GetGpuDeviceFromSysfs(const GpuSysfsPathInfo& path_info, OrtHardwareDevice& gpu_device_out) {
  OrtHardwareDevice gpu_device{};
  const auto& sysfs_path = path_info.path;

  // vendor id
  {
    const auto vendor_id_path = sysfs_path / "device" / "vendor";
    ORT_RETURN_IF_ERROR(ReadValueFromFile(vendor_id_path, gpu_device.vendor_id));
  }

  // TODO vendor name

  // device id
  {
    const auto device_id_path = sysfs_path / "device" / "device";
    ORT_RETURN_IF_ERROR(ReadValueFromFile(device_id_path, gpu_device.device_id));
  }

  // metadata
  gpu_device.metadata.Add("card_idx", MakeString(path_info.card_idx));
  // TODO is card discrete?

  gpu_device.type = OrtHardwareDeviceType_GPU;

  gpu_device_out = std::move(gpu_device);
  return Status::OK();
}

Status GetGpuDevices(std::vector<OrtHardwareDevice>& gpu_devices_out) {
  std::vector<GpuSysfsPathInfo> gpu_sysfs_path_infos{};
  ORT_RETURN_IF_ERROR(DetectGpuSysfsPaths(gpu_sysfs_path_infos));

  std::vector<OrtHardwareDevice> gpu_devices{};
  gpu_devices.reserve(gpu_sysfs_path_infos.size());

  for (const auto& gpu_sysfs_path_info : gpu_sysfs_path_infos) {
    OrtHardwareDevice gpu_device{};
    ORT_RETURN_IF_ERROR(GetGpuDeviceFromSysfs(gpu_sysfs_path_info, gpu_device));
    gpu_devices.emplace_back(std::move(gpu_device));
  }

  gpu_devices_out = std::move(gpu_devices);
  return Status::OK();
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices;

  // get CPU devices
  devices.emplace(GetCpuDeviceFromCPUIDInfo());

  // get GPU devices
  {
    std::vector<OrtHardwareDevice> gpu_devices{};
    Status gpu_device_discovery_status = GetGpuDevices(gpu_devices);
    if (gpu_device_discovery_status.IsOK()) {
      devices.insert(std::make_move_iterator(gpu_devices.begin()),
                     std::make_move_iterator(gpu_devices.end()));
    } else {
      LOGS_DEFAULT(WARNING) << "GPU device discovery failed: " << gpu_device_discovery_status.ErrorMessage();
    }
  }

  // get NPU devices
  // TODO figure out how to discover these

  return devices;
}
}  // namespace onnxruntime
