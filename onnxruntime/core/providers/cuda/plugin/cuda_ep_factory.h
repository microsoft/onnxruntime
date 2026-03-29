// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"
#include "cuda_allocator_plugin.h"
#include "cuda_data_transfer_plugin.h"
#include "cuda_stream_plugin.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

class CudaEp;

/// CUDA EP factory implementing OrtEpFactory.
/// Manages device enumeration, allocator creation, data transfer, and stream creation.
class CudaEpFactory : public OrtEpFactory {
 public:
  CudaEpFactory(const OrtApi& ort_api, const OrtEpApi& ep_api,
                const OrtLogger& default_logger);
  ~CudaEpFactory();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const std::string& GetEpName() const { return ep_name_; }

  /// Get or create the shared kernel registry for this factory.
  /// Lazily created on first call; subsequent calls return the cached instance.
  /// Thread-safe: protected by registry_mutex_.
  OrtStatus* GetKernelRegistryForEp(CudaEp& ep,
                                    const OrtKernelRegistry** out_kernel_registry);

 private:
  // OrtEpFactory callback implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices, size_t num_devices,
      OrtEpDevice** ep_devices, size_t max_ep_devices,
      size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      const OrtKeyValuePairs* const* ep_metadata,
      size_t num_devices,
      const OrtSessionOptions* session_options,
      const OrtLogger* logger,
      OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryInfo* memory_info,
      const OrtKeyValuePairs* allocator_options,
      OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr,
                                                OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr,
      OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtLogger& default_logger_;

  const std::string ep_name_{"CudaPluginExecutionProvider"};
  const std::string vendor_{"NVIDIA"};
  const uint32_t vendor_id_ = 0x10DE;  // NVIDIA PCI vendor ID
  const std::string ep_version_{"1.0.0"};

  struct DeviceCacheEntry {
    int cuda_device_id{-1};
    Ort::MemoryInfo device_memory_info{nullptr};
    Ort::MemoryInfo pinned_memory_info{nullptr};
  };

  struct HardwareDeviceKey {
    OrtHardwareDeviceType type{OrtHardwareDeviceType::OrtHardwareDeviceType_CPU};
    uint32_t vendor_id{0};
    uint32_t device_id{0};

    bool operator==(const HardwareDeviceKey&) const = default;
  };

  struct HardwareDeviceKeyHasher {
    size_t operator()(const HardwareDeviceKey& key) const noexcept {
      size_t hash = static_cast<size_t>(key.type);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.vendor_id);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.device_id);
      return hash;
    }
  };

  static HardwareDeviceKey MakeDeviceKey(const OrtApi& ort_api,
                                         const OrtHardwareDevice& device);

  // Stable per-device cache keyed by public hardware-device properties instead
  // of the transient OrtHardwareDevice* pointer received during enumeration.
  std::mutex device_cache_mutex_;
  std::unordered_map<HardwareDeviceKey, DeviceCacheEntry, HardwareDeviceKeyHasher> device_cache_;

  // Kernel registry (cached, shared across EP instances)
  OrtKernelRegistry* kernel_registry_ = nullptr;
  std::mutex registry_mutex_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
