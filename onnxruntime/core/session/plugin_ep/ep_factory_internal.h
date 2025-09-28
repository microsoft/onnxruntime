// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/common/common.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"
#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "onnxruntime_config.h"  // for ORT_VERSION

namespace onnxruntime {
struct SessionOptions;
class EpFactoryInternalImpl;

// this class can't have any virtual methods as they break using it as an OrtEpFactory* in OrtEpDevice.
class EpFactoryInternal : public OrtEpFactory {
 public:
  EpFactoryInternal(std::unique_ptr<EpFactoryInternalImpl> impl);

  const char* GetName() const noexcept { return impl_->GetName(); }
  const char* GetVendor() const noexcept { return impl_->GetVendor(); }
  uint32_t GetVendorId() const noexcept { return impl_->GetVendorId(); }
  const char* GetVersion() const noexcept { return ORT_VERSION; }

  OrtStatus* GetSupportedDevices(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                 _In_ size_t num_devices,
                                 _Inout_ OrtEpDevice** ep_devices,
                                 _In_ size_t max_ep_devices,
                                 _Out_ size_t* num_ep_devices) noexcept {
    return impl_->GetSupportedDevices(*this, devices, num_devices, ep_devices, max_ep_devices, num_ep_devices);
  }

  // we don't implement this. CreateIExecutionProvider should be used.
  OrtStatus* CreateEp(_In_reads_(num_devices) const OrtHardwareDevice* const* /*devices*/,
                      _In_reads_(num_devices) const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                      _In_ size_t /*num_devices*/,
                      _In_ const OrtSessionOptions* /*session_options*/,
                      _In_ const OrtLogger* /*logger*/,
                      _Out_ OrtEp** /*ep*/) {
    ORT_THROW("Internal error. CreateIExecutionProvider should be used for EpFactoryInternal.");
  }

  // same input args as CreateEp in case we need something from device or ep_metadata_pairs in the future.
  OrtStatus* CreateIExecutionProvider(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                      _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      _In_ size_t num_devices,
                                      _In_ const OrtSessionOptions* session_options,
                                      _In_ const OrtLogger* logger,
                                      _Out_ std::unique_ptr<IExecutionProvider>* ep) {
    return impl_->CreateIExecutionProvider(devices, ep_metadata_pairs, num_devices, session_options, logger, ep);
  }

  OrtStatus* CreateAllocator(_In_ const OrtMemoryInfo* memory_info,
                             _In_opt_ const OrtKeyValuePairs* allocator_options,
                             _Outptr_ OrtAllocator** allocator) noexcept {
    return impl_->CreateAllocator(memory_info, allocator_options, allocator);
  }

  void ReleaseAllocator(_In_ OrtAllocator* allocator) noexcept {
    return impl_->ReleaseAllocator(allocator);
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept {
    return impl_->CreateDataTransfer(data_transfer);
  }

  bool IsStreamAware() const noexcept {
    return impl_->IsStreamAware();
  }

  OrtStatus* CreateSyncStreamForDevice(_In_ const OrtMemoryDevice* memory_device,
                                       _In_opt_ const OrtKeyValuePairs* stream_options,
                                       _Outptr_result_maybenull_ OrtSyncStreamImpl** stream) noexcept {
    return impl_->CreateSyncStreamForDevice(memory_device, stream_options, stream);
  }

  OrtStatus* ValidateCompiledModelCompatibilityInfo(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                                    _In_ size_t num_devices,
                                                    _In_ const char* compatibility_info,
                                                    _Out_ OrtCompiledModelCompatibility* model_compatibility) noexcept {
    return impl_->ValidateCompiledModelCompatibilityInfo(devices, num_devices, compatibility_info, model_compatibility);
  }

  // Function ORT calls to release an EP instance.
  void ReleaseEp(OrtEp* /*ep*/) noexcept {
    // we never create an OrtEp so we should never be trying to release one
  }

 private:
  std::unique_ptr<EpFactoryInternalImpl> impl_;
};

// IExecutionProviderFactory for EpFactoryInternal that is required for SessionOptionsAppendExecutionProvider_V2
struct InternalExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  InternalExecutionProviderFactory(EpFactoryInternal& ep_factory, gsl::span<const OrtEpDevice* const> ep_devices);

  std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                     const OrtLogger& session_logger) override;

  std::unique_ptr<IExecutionProvider> CreateProvider() override {
    ORT_NOT_IMPLEMENTED("CreateProvider without parameters is not supported.");
  }

 private:
  EpFactoryInternal& ep_factory_;
  std::vector<const OrtHardwareDevice*> devices_;
  std::vector<const OrtKeyValuePairs*> ep_metadata_;
};

}  // namespace onnxruntime
