#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace onnxruntime {
namespace cuda {

struct MemoryCapacityError {
  const char* operation = nullptr;
  int code = 0;

  explicit operator bool() const noexcept { return operation != nullptr; }
};

template <typename DriverApi, size_t BufferSize>
const char* FormatMemoryCapacityError(DriverApi& driver, MemoryCapacityError error,
                                    char (&buffer)[BufferSize]) noexcept {
  const char* name = driver.GetErrorName(error.code);
  const char* description = driver.GetErrorString(error.code);
  std::snprintf(buffer, BufferSize, "%s: CUDA error %d (%s): %s", error.operation, error.code,
                name ? name : "unknown", description ? description : "no description available");
  return buffer;
}

template <typename DriverApi>
MemoryCapacityError QueryDeviceMemoryCapacity(DriverApi& driver, int device_id,
                                            uint64_t& total_memory_bytes,
                                            uint64_t& available_memory_bytes) noexcept {
  typename DriverApi::Device device{};
  typename DriverApi::Context context{};
  if (const auto result = driver.Init(); result != 0) {
    return {"cuInit failed", static_cast<int>(result)};
  }
  if (const auto result = driver.DeviceGet(&device, device_id); result != 0) {
    return {"cuDeviceGet failed", static_cast<int>(result)};
  }
  size_t total{};
  if (const auto result = driver.DeviceTotalMem(&total, device); result != 0) {
    return {"cuDeviceTotalMem failed", static_cast<int>(result)};
  }
  if (const auto result = driver.PrimaryCtxRetain(&context, device); result != 0) {
    return {"cuDevicePrimaryCtxRetain failed", static_cast<int>(result)};
  }
  if (const auto result = driver.CtxPushCurrent(context); result != 0) {
    driver.PrimaryCtxRelease(device);
    return {"cuCtxPushCurrent failed", static_cast<int>(result)};
  }

  size_t available{};
  size_t context_total{};
  const auto query_result = driver.MemGetInfo(&available, &context_total);
  typename DriverApi::Context popped_context{};
  const auto restore_result = driver.CtxPopCurrent(&popped_context);
  const auto release_result = driver.PrimaryCtxRelease(device);
  if (restore_result != 0) {
    return {"cuCtxPopCurrent failed to restore the calling thread's CUDA context", static_cast<int>(restore_result)};
  }
  if (release_result != 0) {
    return {"cuDevicePrimaryCtxRelease failed", static_cast<int>(release_result)};
  }
  if (query_result != 0) {
    return {"cuMemGetInfo failed", static_cast<int>(query_result)};
  }

  total_memory_bytes = static_cast<uint64_t>(total);
  available_memory_bytes = static_cast<uint64_t>(available);
  return {};
}

}  // namespace cuda
}  // namespace onnxruntime