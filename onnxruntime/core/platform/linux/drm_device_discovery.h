// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header exposes Linux DRM device discovery internals for testing.

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

#include "core/common/status.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
namespace drm_device_discovery {

struct GpuSysfsPathInfo {
  size_t card_idx{};
  std::filesystem::path path;
  bool is_nvidia_platform_gpu{};
};

Status DetectGpuSysfsPaths(const std::filesystem::path& sysfs_class_drm_path,
                           std::vector<GpuSysfsPathInfo>& gpu_sysfs_paths_out);

Status GetGpuDeviceFromSysfs(const GpuSysfsPathInfo& path_info,
                             OrtHardwareDevice& gpu_device_out);

}  // namespace drm_device_discovery
}  // namespace onnxruntime
