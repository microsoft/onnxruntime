// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "ep.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

/// <summary>
/// A bridge class between the EP API and the WebGPU EP Factory implementation.
/// </summary>
class Factory : public OrtEpFactory {
 public:
  struct Config {
    // Allow the WebGPU EP to be advertised against a CPU hardware device when no GPU hardware device is available.
    // This only makes the WebGPU EP selectable as an EP device; Dawn selects and configures the software adapter
    // independently.
    bool allow_software_adapter{false};

    // Allow advertising an additional virtual GPU device for device-free compile-only sessions.
    bool allow_virtual_devices{false};
  };

  explicit Factory(Config config);
  ~Factory();

 private:
  // Static C API implementations
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      OrtEpDevice** ep_devices,
      size_t max_ep_devices,
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

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(
      OrtEpFactory* this_ptr,
      OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(
      OrtEpFactory* this_ptr,
      const OrtMemoryDevice* memory_device,
      const OrtKeyValuePairs* stream_options,
      OrtSyncStreamImpl** stream) noexcept;

  const Config config_;

  Ort::MemoryInfo default_memory_info_;
  Ort::MemoryInfo readonly_memory_info_;  // used for initializers

  // Owned virtual GPU hardware device created when the environment allows virtual devices (OrtEnv config
  // "allow_virtual_devices"=1). Released in the destructor. nullptr when virtual devices are disabled.
  OrtHardwareDevice* virtual_hw_device_ = nullptr;
};

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
