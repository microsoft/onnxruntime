#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace cuda {

template <typename DriverApi>
const char* QueryDeviceMemoryCapacity(DriverApi& driver, int device_id,
                                      uint64_t& total_memory_bytes,
                                      uint64_t& available_memory_bytes) noexcept {
  typename DriverApi::Device device{};
  typename DriverApi::Context context{};
  if (!driver.Init()) {
    return "cuInit failed";
  }
  if (!driver.DeviceGet(&device, device_id)) {
    return "cuDeviceGet failed";
  }
  size_t total{};
  if (!driver.DeviceTotalMem(&total, device)) {
    return "cuDeviceTotalMem failed";
  }
  if (!driver.PrimaryCtxRetain(&context, device)) {
    return "cuDevicePrimaryCtxRetain failed";
  }
  if (!driver.CtxPushCurrent(context)) {
    driver.PrimaryCtxRelease(device);
    return "cuCtxPushCurrent failed";
  }

  size_t available{};
  size_t context_total{};
  const bool query_ok = driver.MemGetInfo(&available, &context_total);
  typename DriverApi::Context popped_context{};
  const bool restore_ok = driver.CtxPopCurrent(&popped_context);
  const bool release_ok = driver.PrimaryCtxRelease(device);
  if (!restore_ok) {
    return "cuCtxPopCurrent failed to restore the calling thread's CUDA context";
  }
  if (!release_ok) {
    return "cuDevicePrimaryCtxRelease failed";
  }
  if (!query_ok) {
    return "cuMemGetInfo failed";
  }

  total_memory_bytes = static_cast<uint64_t>(total);
  available_memory_bytes = static_cast<uint64_t>(available);
  return nullptr;
}

}  // namespace cuda
}  // namespace onnxruntime