// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/framework/execution_provider.h"
#include "core/framework/provider_options.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
class EpFactoryInternal;
struct SessionOptions;

// class with virtual methods that are implemented for each internal EP
class EpFactoryInternalImpl {
 public:
  EpFactoryInternalImpl(const std::string& ep_name, const std::string& vendor, uint32_t vendor_id)
      : ep_name_(ep_name), vendor_(vendor), vendor_id_(vendor_id) {
  }

  const char* GetName() const noexcept { return ep_name_.c_str(); }
  const char* GetVendor() const noexcept { return vendor_.c_str(); }
  uint32_t GetVendorId() const noexcept { return vendor_id_; }
  const char* GetVersion() const noexcept;

  virtual OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                         _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                         _In_ size_t num_devices,
                                         _Inout_ OrtEpDevice** ep_devices,
                                         _In_ size_t max_ep_devices,
                                         _Out_ size_t* num_ep_devices) noexcept = 0;

  virtual OrtStatus* CreateIExecutionProvider(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                              _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
                                              _In_ size_t num_devices,
                                              _In_ const OrtSessionOptions* session_options,
                                              _In_ const OrtLogger* logger,
                                              _Out_ std::unique_ptr<IExecutionProvider>* ep) = 0;

  virtual OrtStatus* CreateAllocator(_In_ const OrtMemoryInfo* /*memory_info*/,
                                     _In_opt_ const OrtKeyValuePairs* /*allocator_options*/,
                                     _Outptr_ OrtAllocator** allocator) noexcept {
    // default implementation does not add OrtMemoryInfo to OrtEpDevice instances returned
    // so this should never be called
    *allocator = nullptr;
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "CreateAllocator is not implemented for this EP factory.");
  }

  virtual void ReleaseAllocator(_In_ OrtAllocator* /*allocator*/) noexcept {
    // we don't create any allocators so we don't need to release any
  }

  virtual OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;  // Default implementation does nothing
  }

  virtual bool IsStreamAware() const noexcept {
    return false;
  }

   virtual OrtStatus* ValidateCompiledModelCompatibilityInfo(_In_ const char* compatibility_info,
                                                             _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept {
    ORT_UNUSED_PARAMETER(compatibility_info);
    // Default implementation: mark as not applicable
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return nullptr;
  }

  virtual OrtStatus* CreateSyncStreamForDevice(_In_ const OrtMemoryDevice* /*memory_device*/,
                                               _In_opt_ const OrtKeyValuePairs* /*stream_options*/,
                                               _Outptr_result_maybenull_ OrtSyncStreamImpl** stream) noexcept {
    *stream = nullptr;
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED,
                                 "CreateSyncStreamForDevice is not implemented for this EP factory.");
  }

  // Function ORT calls to release an EP instance.
  void ReleaseEp(OrtEp* ep);

  virtual ~EpFactoryInternalImpl() = default;

 protected:
  ProviderOptions GetOptionsFromSessionOptions(const SessionOptions& session_options) const;

 private:
  const std::string ep_name_;  // EP name library was registered with
  const std::string vendor_;   // EP vendor name
  const uint32_t vendor_id_;   // EP vendor ID
};
}  // namespace onnxruntime
