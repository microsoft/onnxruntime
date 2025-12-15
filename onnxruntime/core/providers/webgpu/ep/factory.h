// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <type_traits>

namespace onnxruntime {
namespace webgpu {
namespace ep {

/// <summary>
/// A bridge class between the EP API and the WebGPU EP Factory implementation.
/// </summary>
class Factory : public OrtEpFactory {
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

  static OrtStatus* ORT_API_CALL ValidateCompiledModelCompatibilityInfoImpl(
      OrtEpFactory* this_ptr,
      const OrtHardwareDevice* const* devices,
      size_t num_devices,
      const char* compatibility_info,
      OrtCompiledModelCompatibility* model_compatibility) noexcept;

  static OrtStatus* ORT_API_CALL SetEnvironmentOptionsImpl(
      OrtEpFactory* this_ptr,
      const OrtKeyValuePairs* options) noexcept;

 public:
  Factory();
  ~Factory() = default;
};

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
