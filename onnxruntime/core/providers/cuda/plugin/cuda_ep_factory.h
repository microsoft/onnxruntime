// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"
#include "cuda_allocator_plugin.h"
#include "cuda_arena.h"
#include "cuda_mempool_allocator_plugin.h"
#include "cuda_data_transfer_plugin.h"
#include "cuda_stream_plugin.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers.h"

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

  /// Get the device arena allocator for the given CUDA ordinal, or nullptr if none.
  CudaArenaAllocator* GetDeviceArenaForDevice(int device_id);

  /// Reset arena chunk-to-stream assignments for a device while holding the arena lock.
  /// This avoids the use-after-free risk of calling GetDeviceArenaForDevice() and then
  /// using the raw pointer after the arena_mutex is released.
  OrtStatus* ResetDeviceArenaChunksUsingStream(int device_id, const OrtSyncStreamImpl* stream_impl);

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
  const std::string ep_version_{ORT_PLUGIN_EP_VERSION};

  struct DeviceCacheEntry {
    int cuda_device_id{-1};
    Ort::MemoryInfo device_memory_info{nullptr};
    Ort::MemoryInfo pinned_memory_info{nullptr};

    // Arena members
    std::mutex arena_mutex;
    std::unique_ptr<CudaArenaAllocator> device_arena;
    std::unique_ptr<CudaArenaAllocator> pinned_arena;
    std::unique_ptr<CudaMempoolOrtAllocator> mempool_allocator;
    int num_device_arena_users = 0;
    int num_pinned_arena_users = 0;
    int num_mempool_users = 0;
  };

  struct HardwareDeviceKey {
    OrtHardwareDeviceType type{OrtHardwareDeviceType::OrtHardwareDeviceType_CPU};
    uint32_t vendor_id{0};
    uint32_t device_id{0};  // PCI device ID — identifies the hardware model, NOT a unique device
    int cuda_ordinal{-1};   // CUDA ordinal — unique per physical GPU on this host

    bool operator==(const HardwareDeviceKey&) const = default;
  };

  struct HardwareDeviceKeyHasher {
    size_t operator()(const HardwareDeviceKey& key) const noexcept {
      size_t hash = static_cast<size_t>(key.type);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.vendor_id);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.device_id);
      hash = (hash * 1315423911u) ^ static_cast<size_t>(key.cuda_ordinal);
      return hash;
    }
  };

  static HardwareDeviceKey MakeDeviceKey(const OrtApi& ort_api,
                                         const OrtHardwareDevice& device,
                                         int cuda_ordinal);

  // Per-physical-device cache. The key includes the CUDA ordinal to distinguish
  // identical GPUs (same PCI vendor/device ID) on multi-GPU hosts.
  std::mutex device_cache_mutex_;
  std::unordered_map<HardwareDeviceKey, DeviceCacheEntry, HardwareDeviceKeyHasher> device_cache_;

  // Ordinal-to-HardwareDeviceKey mapping built during GetSupportedDevicesImpl.
  InlinedHashMap<int, HardwareDeviceKey> ordinal_to_device_key_;

  /// Find the DeviceCacheEntry for a given CUDA ordinal.
  /// Returns nullptr if the ordinal has not been registered.
  DeviceCacheEntry* FindDeviceCacheEntryByOrdinal(int cuda_ordinal);

  /// Same as FindDeviceCacheEntryByOrdinal but assumes device_cache_mutex_ is already held.
  DeviceCacheEntry* FindDeviceCacheEntryByOrdinalLocked(int cuda_ordinal);

  // Kernel registry (cached, shared across EP instances)
  OrtKernelRegistry* kernel_registry_ = nullptr;
  std::mutex registry_mutex_;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
