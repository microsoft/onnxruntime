// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>

#include "ep_arena.h"
#include "ep_data_transfer.h"
#include "example_plugin_ep_utils.h"

/// <summary>
/// Example EP factory that can create an OrtEp and return information about the supported hardware devices.
/// </summary>
class ExampleEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  ExampleEpFactory(const char* ep_name, ApiPtrs apis, const OrtLogger& default_logger);

  OrtDataTransferImpl* GetDataTransfer() const {
    return data_transfer_impl_.get();
  }

  // Get the shared arena allocator if created.
  ArenaAllocator* GetArenaAllocator() const {
    return arena_allocator_.get();
  }

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

  const OrtLogger& default_logger_;        // default logger for the EP factory
  const std::string ep_name_;              // EP name
  const std::string vendor_{"Contoso"};    // EP vendor name
  const uint32_t vendor_id_{0xB357};       // EP vendor ID
  const std::string ep_version_{"0.1.0"};  // EP version

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr default_memory_info_;
  MemoryInfoUniquePtr readonly_memory_info_;  // used for initializers

  bool arena_allocator_using_default_settings_{true};
  std::unique_ptr<ArenaAllocator> arena_allocator_;  // shared device allocator that uses an arena
  uint32_t num_arena_users_{0};
  std::mutex mutex_;  // mutex to protect arena_allocator_ and num_arena_users_

  std::unique_ptr<ExampleDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};
