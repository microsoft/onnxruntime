// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header exposes Linux NPU device discovery internals for testing.

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "core/common/status.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
namespace npu_device_discovery {

struct NpuSysfsPathInfo {
  size_t accel_idx;
  std::filesystem::path path;
};

// Scans the given sysfs accel directory for NPU accel devices.
Status DetectNpuSysfsPaths(const std::filesystem::path& sysfs_accel_path,
                           std::vector<NpuSysfsPathInfo>& npu_sysfs_paths_out);

// Reads vendor/device IDs and populates an OrtHardwareDevice from an accel sysfs path.
Status GetNpuDeviceFromSysfs(const NpuSysfsPathInfo& path_info,
                             OrtHardwareDevice& npu_device_out);

}  // namespace npu_device_discovery
}  // namespace onnxruntime
