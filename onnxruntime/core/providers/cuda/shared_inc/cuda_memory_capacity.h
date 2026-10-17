#pragma once

#include <cuda.h>

#include "device_memory_capacity.h"

namespace onnxruntime {
namespace cuda {

struct MemoryCapacityDriverApi {
  using Device = CUdevice;
  using Context = CUcontext;

  bool Init() noexcept { return cuInit(0) == CUDA_SUCCESS; }
  bool DeviceGet(Device* device, int ordinal) noexcept { return cuDeviceGet(device, ordinal) == CUDA_SUCCESS; }
  bool DeviceTotalMem(size_t* total, Device device) noexcept { return cuDeviceTotalMem(total, device) == CUDA_SUCCESS; }
  bool PrimaryCtxRetain(Context* context, Device device) noexcept {
    return cuDevicePrimaryCtxRetain(context, device) == CUDA_SUCCESS;
  }
  bool CtxPushCurrent(Context context) noexcept { return cuCtxPushCurrent(context) == CUDA_SUCCESS; }
  bool MemGetInfo(size_t* available, size_t* total) noexcept { return cuMemGetInfo(available, total) == CUDA_SUCCESS; }
  bool CtxPopCurrent(Context* context) noexcept { return cuCtxPopCurrent(context) == CUDA_SUCCESS; }
  bool PrimaryCtxRelease(Device device) noexcept { return cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS; }
};

}  // namespace cuda
}  // namespace onnxruntime