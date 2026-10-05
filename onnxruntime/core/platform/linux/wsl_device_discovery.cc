// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/linux/wsl_device_discovery.h"

#include <filesystem>
#include <system_error>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace wsl_device_discovery {

namespace {

// Minimal subset of the D3DKMT interface that libdxcore.so exports on WSL2. It is
// declared here instead of included from the Windows SDK/WDK so that a Linux build
// needs no Windows headers. Layouts mirror d3dkmthk.h as it expands under LP64, which
// is also how the LX_DX* ioctl structs in the WSL2 kernel's uapi/misc/d3dkmthk.h are
// laid out. Nothing here is reachable on a non-LP64 target because WSL2 is 64-bit only
// and the /dev/dxg check below fails first.
//
// libdxcore.so is not a thin wrapper over those ioctls: it serves most query types by
// calling into the host GPU driver over VM bus, so a query type absent from the kernel
// uapi header still works here.

using NTSTATUS = int32_t;
constexpr NTSTATUS kStatusSuccess = 0;

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
constexpr uint32_t kQueryTypePhysicalAdapterDeviceIds = 31;

// D3DKMT_ENUMADAPTERS_FILTER bit. Compute-only adapters are excluded by default.
constexpr uint64_t kEnumAdaptersFilterIncludeComputeOnly = 1;

class DxCoreThunks {
 public:
  static const DxCoreThunks& Instance() {
    static const DxCoreThunks instance{};
    return instance;
  }

  bool IsAvailable() const {
    return query_adapter_info != nullptr && (enum_adapters3 != nullptr || enum_adapters2 != nullptr);
  }

  NTSTATUS (*enum_adapters2)(void* args) = nullptr;
  NTSTATUS (*enum_adapters3)(void* args) = nullptr;
  NTSTATUS (*query_adapter_info)(void* args) = nullptr;
  NTSTATUS (*close_adapter)(void* args) = nullptr;

 private:
  DxCoreThunks() {
    std::error_code error_code{};
    if (!std::filesystem::exists("/dev/dxg", error_code)) {
      return;
    }

    const Env& env = Env::Default();

    // Never unloaded: libdxcore.so registers process-wide state with /dev/dxg that does
    // not survive an unload, and the thunks are needed for the process lifetime anyway.
    void* library = nullptr;
    if (const Status status = env.LoadDynamicLibrary("libdxcore.so", /*global_symbols*/ false, &library);
        !status.IsOK()) {
      LOGS_DEFAULT(VERBOSE) << "/dev/dxg is present but libdxcore.so could not be loaded: "
                            << status.ErrorMessage();
      return;
    }

    // A missing symbol leaves the thunk null; IsAvailable() and the EnumAdapters3
    // fallback decide what is still usable.
    const auto load = [&env, library](const char* name) {
      void* symbol = nullptr;
      if (!env.GetSymbolFromLibrary(library, name, &symbol).IsOK()) {
        return static_cast<NTSTATUS (*)(void*)>(nullptr);
      }
      return reinterpret_cast<NTSTATUS (*)(void*)>(symbol);
    };

    enum_adapters2 = load("D3DKMTEnumAdapters2");
    enum_adapters3 = load("D3DKMTEnumAdapters3");
    query_adapter_info = load("D3DKMTQueryAdapterInfo");
    close_adapter = load("D3DKMTCloseAdapter");
  }
};

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

Status GetGpuDevices(std::vector<WslGpuInfo>& gpu_devices_out) {
  gpu_devices_out = {};

  const auto& thunks = DxCoreThunks::Instance();
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

    const NTSTATUS status = thunks.query_adapter_info(&query);

    if (thunks.close_adapter != nullptr) {
      D3DKMT_CLOSEADAPTER close_args{};
      close_args.hAdapter = adapter.hAdapter;
      thunks.close_adapter(&close_args);
    }

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

}  // namespace wsl_device_discovery
}  // namespace onnxruntime
