// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#include "utils.h"

struct ExampleEpFactory : OrtEpFactory, ApiPtrs {
  ExampleEpFactory(const char* ep_name, ApiPtrs apis);

  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr);

  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr);

  static OrtStatus* ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                                         const OrtHardwareDevice* const* devices,
                                                         size_t num_devices,
                                                         OrtEpDevice** ep_devices,
                                                         size_t max_ep_devices,
                                                         size_t* p_num_ep_devices);

  static OrtStatus* ORT_API_CALL CreateEpImpl(OrtEpFactory* this_ptr,
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep);

  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep);

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(_In_ OrtEpFactory* this_ptr,
                                                     _In_ const OrtMemoryInfo* memory_info,
                                                     _In_ const OrtKeyValuePairs* /*allocator_options*/,
                                                     _Outptr_ OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(_In_ OrtEpFactory* /*this*/, _In_ OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(_In_ OrtEpFactory* this_ptr,
                                                        _Outptr_ OrtDataTransferImpl** data_transfer) noexcept;

  static void ORT_API_CALL ReleaseDataTransferImpl(_In_ OrtEpFactory* /*this_ptr*/,
                                                   _In_ OrtDataTransferImpl* data_transfer) noexcept;

  const std::string ep_name_;            // EP name
  const std::string vendor_{"Contoso"};  // EP vendor name

 private:
  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr cpu_memory_info_;

  // for example purposes. if the EP used GPU, and pinned/shared memory was required for data transfer, these are the
  // OrtMemoryInfo instance required for that.
  MemoryInfoUniquePtr default_gpu_memory_info_;
  MemoryInfoUniquePtr pinned_gpu_memory_info_;
};
