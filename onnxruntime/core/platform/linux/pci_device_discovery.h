// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header exposes Linux PCI device discovery internals for testing.

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/common/status.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
namespace pci_device_discovery {

struct GpuPciPathInfo {
  std::filesystem::path path;
  std::string pci_bus_id;
};

// Scans the given sysfs PCI devices directory for GPU devices.
// Filters by PCI class codes: 0x0300 (VGA) and 0x0302 (3D controller).
Status DetectGpuPciPaths(const std::filesystem::path& sysfs_pci_devices_path,
                         std::vector<GpuPciPathInfo>& gpu_pci_paths_out);

// Reads vendor/device IDs and populates an OrtHardwareDevice from a PCI device sysfs path.
Status GetGpuDeviceFromPci(const GpuPciPathInfo& path_info,
                           OrtHardwareDevice& gpu_device_out);

}  // namespace pci_device_discovery
}  // namespace onnxruntime
