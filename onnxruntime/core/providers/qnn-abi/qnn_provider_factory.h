// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/framework/error_code_helper.h"
#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/qnn_execution_provider.h"

namespace onnxruntime {

class QnnEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  QnnEpFactory(const char* ep_name,
               ApiPtrs ort_api_in,
               std::unordered_map<OrtHardwareDeviceType, std::string> supported_backends);

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
                                              _In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata*/,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ OrtEp** ep) noexcept;
  static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept;
  static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory* /*this*/, OrtAllocator* allocator) noexcept;
  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;
  static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory* this_ptr) noexcept;

  // const OrtApi& ort_api;
  const std::string ep_name_;              // EP name
  const std::string vendor_{"Qualcomm"};   // EP vendor name
  const std::string ep_version_{"0.1.0"};  // EP version

  // Qualcomm vendor ID. Refer to the ACPI ID registry (search Qualcomm): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id_{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  // Supported OrtHardwareDeviceTypes and their QNN backend paths.
  const std::unordered_map<OrtHardwareDeviceType, std::string> supported_backends_;

  // CPU allocator so we can control the arena behavior. optional as ORT always provides a CPU allocator if needed.
  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;
  MemoryInfoUniquePtr host_accessible_memory_info_;

  std::vector<OrtEpDevice*> ep_devices_;
};

}  // namespace onnxruntime
