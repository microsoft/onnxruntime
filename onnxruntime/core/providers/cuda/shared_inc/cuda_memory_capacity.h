#pragma once

#include <cuda.h>

#include "device_memory_capacity.h"

namespace onnxruntime {
namespace cuda {

struct MemoryCapacityDriverApi {
  using Device = CUdevice;
  using Context = CUcontext;

  CUresult Init() noexcept { return cuInit(0); }
  CUresult DeviceGet(Device* device, int ordinal) noexcept { return cuDeviceGet(device, ordinal); }
  CUresult DeviceTotalMem(size_t* total, Device device) noexcept { return cuDeviceTotalMem(total, device); }
  CUresult PrimaryCtxRetain(Context* context, Device device) noexcept {
    return cuDevicePrimaryCtxRetain(context, device);
  }
  CUresult CtxPushCurrent(Context context) noexcept { return cuCtxPushCurrent(context); }
  CUresult MemGetInfo(size_t* available, size_t* total) noexcept { return cuMemGetInfo(available, total); }
  CUresult CtxPopCurrent(Context* context) noexcept { return cuCtxPopCurrent(context); }
  CUresult PrimaryCtxRelease(Device device) noexcept { return cuDevicePrimaryCtxRelease(device); }
  const char* GetErrorName(int code) noexcept {
    const char* name = nullptr;
    cuGetErrorName(static_cast<CUresult>(code), &name);
    return name;
  }
  const char* GetErrorString(int code) noexcept {
    const char* description = nullptr;
    cuGetErrorString(static_cast<CUresult>(code), &description);
    return description;
  }
};

}  // namespace cuda
}  // namespace onnxruntime