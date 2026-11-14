// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GPU discovery for WSL2, where the GPU is paravirtualized through /dev/dxg and has no
// /sys/class/drm or /sys/bus/pci entry for the sysfs-based discovery paths to find.
//
// The D3DKMT subset below is the libdxcore.so calling convention. It is declared here,
// rather than included from the Windows SDK/WDK, so a Linux build needs no Windows headers
// and tests can supply the thunks without a GPU-PV guest.

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "core/common/status.h"
#include "core/session/abi_devices.h"

namespace onnxruntime {
namespace wsl_device_discovery {

struct WslGpuInfo {
  uint16_t vendor_id;
  uint16_t device_id;
  uint64_t luid;
};

using D3DKMT_HANDLE = uint32_t;

struct D3DKMT_LUID {
  uint32_t LowPart;
  int32_t HighPart;
};

struct D3DKMT_ADAPTERINFO {
  D3DKMT_HANDLE hAdapter;
  D3DKMT_LUID AdapterLuid;
  uint32_t NumOfSources;
  int32_t bPrecisePresentRegionsPreferred;
};

struct D3DKMT_ENUMADAPTERS2 {
  uint32_t NumAdapters;
  D3DKMT_ADAPTERINFO* pAdapters;
};

struct D3DKMT_ENUMADAPTERS3 {
  uint64_t Filter;
  uint32_t NumAdapters;
  D3DKMT_ADAPTERINFO* pAdapters;
};

struct D3DKMT_QUERYADAPTERINFO {
  D3DKMT_HANDLE hAdapter;
  uint32_t Type;
  void* pPrivateDriverData;
  uint32_t PrivateDriverDataSize;
};

struct D3DKMT_DEVICE_IDS {
  uint32_t VendorID;
  uint32_t DeviceID;
  uint32_t SubVendorID;
  uint32_t SubSystemID;
  uint32_t RevisionID;
  uint32_t BusType;
};

struct D3DKMT_QUERY_DEVICE_IDS {
  uint32_t PhysicalAdapterIndex;
  D3DKMT_DEVICE_IDS DeviceIds;
};

struct D3DKMT_CLOSEADAPTER {
  D3DKMT_HANDLE hAdapter;
};

// KMTQUERYADAPTERINFOTYPE value.
inline constexpr uint32_t kQueryTypePhysicalAdapterDeviceIds = 31;

// D3DKMT_ENUMADAPTERS_FILTER bit. Compute-only adapters are excluded by default.
inline constexpr uint64_t kEnumAdaptersFilterIncludeComputeOnly = 1;

// Function pointers from libdxcore.so. Tests supply their own; production loads them
// after confirming the dxg device node exists.
struct DxCoreThunks {
  using Thunk = int32_t (*)(void* args);

  Thunk enum_adapters2 = nullptr;
  Thunk enum_adapters3 = nullptr;
  Thunk query_adapter_info = nullptr;
  Thunk close_adapter = nullptr;

  // Close is required: EnumAdapters* opens a handle per adapter. An incomplete library
  // degrades to the sysfs/PCI discovery paths instead of leaking those handles.
  bool IsAvailable() const {
    return close_adapter != nullptr && query_adapter_info != nullptr &&
           (enum_adapters3 != nullptr || enum_adapters2 != nullptr);
  }
};

// Enumerates adapters through the supplied thunks. An incomplete set yields an empty
// list without failing.
Status GetGpuDevices(const DxCoreThunks& thunks, std::vector<WslGpuInfo>& gpu_devices_out);

// Production entry. When `dxg_device` is absent this returns an empty list without
// loading libdxcore.so. The default node is /dev/dxg.
Status GetGpuDevices(std::vector<WslGpuInfo>& gpu_devices_out,
                     const std::filesystem::path& dxg_device = "/dev/dxg");

// `wsl_gpu_infos` empty leaves `gpu_devices` unchanged, so a sysfs/PCI result is kept
// when D3DKMT enumeration yields nothing. Otherwise Microsoft vendor 0x1414 placeholders
// are removed and the WSL adapters are appended.
void ApplyWslGpuDevices(std::vector<OrtHardwareDevice>& gpu_devices,
                        const std::vector<WslGpuInfo>& wsl_gpu_infos);

}  // namespace wsl_device_discovery
}  // namespace onnxruntime
