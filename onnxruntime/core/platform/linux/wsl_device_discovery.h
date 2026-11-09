// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GPU discovery for WSL2, where the GPU is paravirtualized through /dev/dxg and has no
// /sys/class/drm or /sys/bus/pci entry for the sysfs-based discovery paths to find.

#pragma once

#include <cstdint>
#include <vector>

#include "core/common/status.h"

namespace onnxruntime {
namespace wsl_device_discovery {

struct WslGpuInfo {
  uint16_t vendor_id;
  uint16_t device_id;
  uint64_t luid;
};

// Enumerates GPU adapters through the D3DKMT entry points exported by libdxcore.so.
// Yields an empty vector without failing when not running under WSL2.
Status GetGpuDevices(std::vector<WslGpuInfo>& gpu_devices_out);

}  // namespace wsl_device_discovery
}  // namespace onnxruntime
