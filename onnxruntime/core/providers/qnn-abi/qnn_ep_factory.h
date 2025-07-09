// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_c_api.h"
// #include "qnn_ep.h"
#include "test/autoep/library/ep_data_transfer.h"
#include "test/autoep/library/example_plugin_ep_utils.h"

#if !BUILD_QNN_EP_STATIC_LIB
#include "core/framework/error_code_helper.h"
#include "qnn_ep_data_transfer.h"
#include "qnn_ep.h"

namespace onnxruntime {
// OrtEpApi infrastructure to be able to use the QNN EP as an OrtEpFactory for auto EP selection.
struct QnnEpFactory : public OrtEpFactory, public ApiPtrs {
 public:
  QnnEpFactory(const char* ep_name,
               const ApiPtrs& ort_api_in,
               OrtHardwareDeviceType hw_type,
               const char* qnn_backend_type);

 private:
  static const char* ORT_API_CALL GetNameImpl(const OrtEpFactory* this_ptr) noexcept;
  static const char* ORT_API_CALL GetVendorImpl(const OrtEpFactory* this_ptr) noexcept;
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
  static OrtStatus* ORT_API_CALL CreateDataTransferImpl(OrtEpFactory* this_ptr,
                                                        OrtDataTransferImpl** data_transfer) noexcept;

  // const OrtApi& ort_api;
  const std::string ep_name_;             // EP name
  const std::string vendor_{"Qualcomm"};  // EP vendor name

  // Qualcomm vendor ID. Refer to the ACPI ID registry (search Qualcomm): https://uefi.org/ACPI_ID_List
  const uint32_t vendor_id_{'Q' | ('C' << 8) | ('O' << 16) | ('M' << 24)};
  const OrtHardwareDeviceType ort_hw_device_type_;  // Supported OrtHardwareDevice
  const std::string qnn_backend_type_;              // QNN backend type for OrtHardwareDevice

  using MemoryInfoUniquePtr = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>;

  // If the EP used NPU, and pinned/shared memory was required for data transfer, these are the
  // OrtMemoryInfo instance required for that.
  MemoryInfoUniquePtr default_npu_memory_info_;
  MemoryInfoUniquePtr host_accessible_npu_memory_info_;

  std::unique_ptr<QnnDataTransfer> data_transfer_impl_;  // data transfer implementation for this factory
};

}  // namespace onnxruntime

#endif  // !BUILD_QNN_EP_STATIC_LIB
