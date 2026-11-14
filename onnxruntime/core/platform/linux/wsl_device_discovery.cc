// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/wsl_device_discovery.h"

#include <cstddef>
#include <system_error>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace wsl_device_discovery {

namespace {

constexpr int32_t kStatusSuccess = 0;

// Layouts mirror d3dkmthk.h as it expands under LP64, which is also how the LX_DX* ioctl
// structs in the WSL2 kernel's uapi/misc/d3dkmthk.h are laid out. WSL2 is 64-bit only, and
// GetGpuDevices returns before these thunks are called when the dxg node is absent.
//
// libdxcore.so is not a thin wrapper over those ioctls: it serves most query types by
// calling into the host GPU driver over VM bus, so a query type absent from the kernel
// uapi header still works here.
//
// libdxcore.so marshals these by offset, so the implicit padding before each pointer
// matches the explicit reserved field the kernel header spells out.
static_assert(sizeof(D3DKMT_ADAPTERINFO) == 20);
static_assert(offsetof(D3DKMT_ADAPTERINFO, AdapterLuid) == 4);
static_assert(offsetof(D3DKMT_ADAPTERINFO, NumOfSources) == 12);

static_assert(sizeof(D3DKMT_ENUMADAPTERS2) == 16);
static_assert(offsetof(D3DKMT_ENUMADAPTERS2, pAdapters) == 8);

static_assert(sizeof(D3DKMT_ENUMADAPTERS3) == 24);
static_assert(offsetof(D3DKMT_ENUMADAPTERS3, NumAdapters) == 8);
static_assert(offsetof(D3DKMT_ENUMADAPTERS3, pAdapters) == 16);

static_assert(sizeof(D3DKMT_QUERYADAPTERINFO) == 24);
static_assert(offsetof(D3DKMT_QUERYADAPTERINFO, pPrivateDriverData) == 8);
static_assert(offsetof(D3DKMT_QUERYADAPTERINFO, PrivateDriverDataSize) == 16);

static_assert(sizeof(D3DKMT_QUERY_DEVICE_IDS) == 28);
static_assert(offsetof(D3DKMT_QUERY_DEVICE_IDS, DeviceIds) == 4);

DxCoreThunks LoadDxCoreThunks() {
  DxCoreThunks thunks{};
  const Env& env = Env::Default();

  // Never unloaded: libdxcore.so registers process-wide state with /dev/dxg that does
  // not survive an unload, and the thunks are needed for the process lifetime anyway.
  void* library = nullptr;
  if (const Status status = env.LoadDynamicLibrary("libdxcore.so", /*global_symbols*/ false, &library);
      !status.IsOK()) {
    LOGS_DEFAULT(VERBOSE) << "libdxcore.so could not be loaded: " << status.ErrorMessage();
    return thunks;
  }

  // A missing symbol leaves the thunk null. IsAvailable() decides whether the set is
  // usable, and EnumerateAdapters picks EnumAdapters3 over EnumAdapters2.
  const auto load = [&env, library](const char* name) -> DxCoreThunks::Thunk {
    void* symbol = nullptr;
    if (!env.GetSymbolFromLibrary(library, name, &symbol).IsOK()) {
      return nullptr;
    }
    return reinterpret_cast<DxCoreThunks::Thunk>(symbol);
  };

  thunks.enum_adapters2 = load("D3DKMTEnumAdapters2");
  thunks.enum_adapters3 = load("D3DKMTEnumAdapters3");
  thunks.query_adapter_info = load("D3DKMTQueryAdapterInfo");
  thunks.close_adapter = load("D3DKMTCloseAdapter");
  return thunks;
}

const DxCoreThunks& LoadedThunks() {
  static const DxCoreThunks thunks = LoadDxCoreThunks();
  return thunks;
}

// Runs the two-pass enumeration: the first call reports the adapter count, the second
// fills the caller-provided array.
bool EnumerateAdapters(const DxCoreThunks& thunks, std::vector<D3DKMT_ADAPTERINFO>& adapters_out) {
  if (thunks.enum_adapters3 != nullptr) {
    D3DKMT_ENUMADAPTERS3 args{};
    args.Filter = kEnumAdaptersFilterIncludeComputeOnly;
    if (thunks.enum_adapters3(&args) == kStatusSuccess) {
      if (args.NumAdapters == 0) {
        adapters_out.clear();
        return true;
      }

      std::vector<D3DKMT_ADAPTERINFO> adapters(args.NumAdapters);
      args.pAdapters = adapters.data();
      if (thunks.enum_adapters3(&args) == kStatusSuccess) {
        adapters.resize(args.NumAdapters);
        adapters_out = std::move(adapters);
        return true;
      }
    }
  }

  if (thunks.enum_adapters2 == nullptr) {
    return false;
  }

  // EnumAdapters2 hides compute-only adapters, so it is only a fallback.
  D3DKMT_ENUMADAPTERS2 args{};
  if (thunks.enum_adapters2(&args) != kStatusSuccess) {
    return false;
  }

  if (args.NumAdapters == 0) {
    adapters_out.clear();
    return true;
  }

  std::vector<D3DKMT_ADAPTERINFO> adapters(args.NumAdapters);
  args.pAdapters = adapters.data();
  if (thunks.enum_adapters2(&args) != kStatusSuccess) {
    return false;
  }

  adapters.resize(args.NumAdapters);
  adapters_out = std::move(adapters);
  return true;
}

}  // namespace

Status GetGpuDevices(const DxCoreThunks& thunks, std::vector<WslGpuInfo>& gpu_devices_out) {
  gpu_devices_out = {};

  if (!thunks.IsAvailable()) {
    return Status::OK();
  }

  std::vector<D3DKMT_ADAPTERINFO> adapters{};
  if (!EnumerateAdapters(thunks, adapters)) {
    LOGS_DEFAULT(WARNING) << "Failed to enumerate WSL GPU adapters through libdxcore.so.";
    return Status::OK();
  }

  std::vector<WslGpuInfo> gpu_devices{};
  gpu_devices.reserve(adapters.size());

  for (const auto& adapter : adapters) {
    // Index 0 identifies the whole adapter; the remaining physical adapters of a linked
    // display adapter chain are not separate devices from the runtime's point of view.
    D3DKMT_QUERY_DEVICE_IDS device_ids{};
    D3DKMT_QUERYADAPTERINFO query{};
    query.hAdapter = adapter.hAdapter;
    query.Type = kQueryTypePhysicalAdapterDeviceIds;
    query.pPrivateDriverData = &device_ids;
    query.PrivateDriverDataSize = sizeof(device_ids);

    const int32_t status = thunks.query_adapter_info(&query);

    // IsAvailable() requires close_adapter, so this is always safe to call.
    D3DKMT_CLOSEADAPTER close_args{};
    close_args.hAdapter = adapter.hAdapter;
    thunks.close_adapter(&close_args);

    if (status != kStatusSuccess) {
      LOGS_DEFAULT(WARNING) << "D3DKMTQueryAdapterInfo failed for WSL adapter " << adapter.hAdapter
                            << " with status " << status << ".";
      continue;
    }

    WslGpuInfo gpu_device{};
    gpu_device.vendor_id = static_cast<uint16_t>(device_ids.DeviceIds.VendorID);
    gpu_device.device_id = static_cast<uint16_t>(device_ids.DeviceIds.DeviceID);
    gpu_device.luid = (static_cast<uint64_t>(static_cast<uint32_t>(adapter.AdapterLuid.HighPart)) << 32) |
                      adapter.AdapterLuid.LowPart;
    gpu_devices.emplace_back(gpu_device);
  }

  gpu_devices_out = std::move(gpu_devices);
  return Status::OK();
}

Status GetGpuDevices(std::vector<WslGpuInfo>& gpu_devices_out, const std::filesystem::path& dxg_device) {
  gpu_devices_out = {};

  std::error_code error_code{};
  if (!std::filesystem::exists(dxg_device, error_code)) {
    return Status::OK();
  }

  return GetGpuDevices(LoadedThunks(), gpu_devices_out);
}

}  // namespace wsl_device_discovery
}  // namespace onnxruntime
