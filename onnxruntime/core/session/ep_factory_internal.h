// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
class EpFactoryInternal;
class EpLibraryInternal;
struct SessionOptions;

// class with virtual methods that are implemented for each internal EP
class EpFactoryInternalImpl {
 public:
  EpFactoryInternalImpl(const std::string& ep_name, const std::string& vendor)
      : ep_name_(ep_name), vendor_(vendor) {
  }

  const char* GetName() const noexcept { return ep_name_.c_str(); }
  const char* GetVendor() const noexcept { return vendor_.c_str(); }

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
                                     _In_ const OrtKeyValuePairs* /*allocator_options*/,
                                     _Outptr_ OrtAllocator** allocator) noexcept {
    // default implementation does not add OrtMemoryInfo to OrtEpDevice instances returned
    // so this should never be called
    *allocator = nullptr;
    return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "CreateAllocator is not implemented for this EP factory.");
  }

  virtual void ReleaseAllocator(_In_ OrtAllocator* /*allocator*/) noexcept {
    // we don't create any allocators so we don't need to release any
  }

  virtual OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) {
    *data_transfer = nullptr;
    return nullptr;  // Default implementation does nothing
  }

  virtual bool IsStreamAware() const {
    return false;
  }

  virtual OrtStatus* CreateSyncStreamForDevice(_In_ const OrtMemoryDevice* /*memory_device*/,
                                               _Outptr_result_maybenull_ OrtSyncStreamImpl** stream) {
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
};

// this class can't have any virtual methods as we hook it up to the OrtEpDevice
class EpFactoryInternal : public OrtEpFactory {
 public:
  EpFactoryInternal(std::unique_ptr<EpFactoryInternalImpl> impl);

  const char* GetName() const noexcept { return impl_->GetName(); }
  const char* GetVendor() const noexcept { return impl_->GetVendor(); }

  OrtStatus* GetSupportedDevices(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                 _In_ size_t num_devices,
                                 _Inout_ OrtEpDevice** ep_devices,
                                 _In_ size_t max_ep_devices,
                                 _Out_ size_t* num_ep_devices) noexcept {
    return impl_->GetSupportedDevices(*this, devices, num_devices, ep_devices, max_ep_devices, num_ep_devices);
  }

  // we don't implement this. CreateIExecutionProvider should be used.
  OrtStatus* CreateEp(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                      _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
                      _In_ size_t num_devices,
                      _In_ const OrtSessionOptions* session_options,
                      _In_ const OrtLogger* logger, _Out_ OrtEp** ep);

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
                             _In_ const OrtKeyValuePairs* allocator_options,
                             _Outptr_ OrtAllocator** allocator) noexcept {
    return impl_->CreateAllocator(memory_info, allocator_options, allocator);
  }

  void ReleaseAllocator(_In_ OrtAllocator* allocator) noexcept {
    return impl_->ReleaseAllocator(allocator);
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) {
    return impl_->CreateDataTransfer(data_transfer);
  }

  bool IsStreamAware() const {
    return impl_->IsStreamAware();
  }

  OrtStatus* CreateSyncStreamForDevice(_In_ const OrtMemoryDevice* memory_device,
                                       _Outptr_result_maybenull_ OrtSyncStreamImpl** stream) {
    return impl_->CreateSyncStreamForDevice(memory_device, stream);
  }

  // Function ORT calls to release an EP instance.
  void ReleaseEp(OrtEp* /*ep*/) {
    // we never create an OrtEp so we should never be trying to release one
    ORT_THROW("Internal error. No ReleaseEp call is required for EpFactoryInternal.");
  }

 private:
  std::unique_ptr<EpFactoryInternalImpl> impl_;
  // std::vector<std::unique_ptr<EpFactoryInternal>> eps_;  // EP instances created by this factory
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
