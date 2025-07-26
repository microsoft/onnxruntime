// Copyright (c) Microsoft Corporation. All rights reserved.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "core/providers/cuda/cuda_pch.h"
#include "core/providers/cuda/plugin_ep/data_transfer.h"
#include "core/session/onnxruntime_c_api.h"

namespace cuda_plugin_ep {

struct CudaEpFactory : OrtEpFactory {
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

  CudaEpFactory();

  static const char* GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;

  static uint32_t GetVendorIdImpl(const OrtEpFactory* this_ptr) noexcept;

  static const char* ORT_API_CALL GetVersionImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* GetSupportedDevicesImpl(OrtEpFactory* this_ptr,
                                            const OrtHardwareDevice* const* devices,
                                            size_t num_devices,
                                            OrtEpDevice** ep_devices,
                                            size_t max_ep_devices,
                                            size_t* p_num_ep_devices) noexcept;
  static OrtStatus* CreateEpImpl(OrtEpFactory* this_ptr,
                                 _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                 _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata,
                                 _In_ size_t num_devices,
                                 _In_ const OrtSessionOptions* session_options,
                                 _In_ const OrtLogger* logger,
                                 _Out_ OrtEp** ep) noexcept;

  static void ReleaseEpImpl(OrtEpFactory* this_ptr, OrtEp* ep) noexcept;

  static OrtStatus* ORT_API_CALL CreateAllocatorImpl(OrtEpFactory* this_ptr,
                                                     const OrtMemoryInfo* memory_info,
                                                     const OrtKeyValuePairs* allocator_options,
                                                     OrtAllocator** allocator) noexcept;

  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* this_ptr, OrtAllocator* allocator) noexcept;

  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory* this_ptr,
                                                               const OrtMemoryDevice* memory_device,
                                                               const OrtKeyValuePairs* stream_options,
                                                               OrtSyncStreamImpl** ort_stream) noexcept;

 private:
  OrtStatus* CreateMemoryInfoForDevices(int num_devices);

  const std::string ep_name{"CUDAExecutionProvider"};  // EP name
  const std::string vendor{"Microsoft"};               // EP vendor name
  uint32_t vendor_id{0x1414};                          // Microsoft vendor ID

  // per-device memory info
  std::vector<MemoryInfoUniquePtr> gpu_memory_infos;
  std::vector<MemoryInfoUniquePtr> host_accessible_memory_infos;

  // we use a shared instance for the OrtDataTransferImpl instead of creating a new one on every call to
  // CreateDataTransferImpl.
  CudaDataTransferImpl data_transfer_impl;

  CudaEpFactory(const CudaEpFactory&) = delete;
  CudaEpFactory& operator=(const CudaEpFactory&) = delete;

  CudaEpFactory(CudaEpFactory&&) = default;
  CudaEpFactory& operator=(CudaEpFactory&&) = default;
};
}  // namespace cuda_plugin_ep
