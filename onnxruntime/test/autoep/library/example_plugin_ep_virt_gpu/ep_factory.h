// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

/// <summary>
/// EP factory that creates an OrtEp instance that supports a virtual GPU OrtHardwareDevice
/// created by the factory itself (not ORT).
/// </summary>
class EpFactoryVirtualGpu : public OrtEpFactory {
 public:
  EpFactoryVirtualGpu(const OrtApi& ort_api, const OrtEpApi& ep_api, const OrtModelEditorApi& model_editor_api,
                      const OrtLogger& default_logger);
  ~EpFactoryVirtualGpu();

  const OrtApi& GetOrtApi() const { return ort_api_; }
  const OrtEpApi& GetEpApi() const { return ep_api_; }
  const OrtModelEditorApi& GetModelEditorApi() const { return model_editor_api_; }
  const std::string& GetEpName() const { return ep_name_; }

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
  static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices) noexcept;

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              const OrtHardwareDevice* const* /*devices*/,
                                              const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger,
                                              OrtEp** ep) noexcept;

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* /*allocator_options*/,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** stream) noexcept;

  static OrtStatus* ORT_API_CALL SetEnvironmentOptionsImpl(OrtEpFactory* this_ptr,
                                                           const OrtKeyValuePairs* options) noexcept;

  const OrtApi& ort_api_;
  const OrtEpApi& ep_api_;
  const OrtModelEditorApi& model_editor_api_;
  bool allow_virtual_devices_{false};
  const OrtLogger& default_logger_;
  OrtHardwareDevice* virtual_hw_device_{};
  const std::string ep_name_{"EpVirtualGpu"};
  const std::string vendor_{"Contoso2"};   // EP vendor name
  const uint32_t vendor_id_{0xB358};       // EP vendor ID
  const std::string ep_version_{"0.1.0"};  // EP version
};
