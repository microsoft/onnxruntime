// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <vector>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "core/providers/providers.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class EpLibraryInternal;
struct SessionOptions;

class EpFactoryInternal : public OrtEpFactory {
 public:
  // factory is non-const as a pointer to the factory is added to OrtEpDevice and needs to be non-const
  // to provide flexibility to the CreateEp call to modify internal state.
  using GetSupportedFunc = std::function<OrtStatus*(OrtEpFactory* factory,
                                                    const OrtHardwareDevice* const* devices,
                                                    size_t num_devices,
                                                    OrtEpDevice** ep_devices,
                                                    size_t max_ep_devices,
                                                    size_t* num_ep_devices)>;

  using CreateFunc = std::function<OrtStatus*(OrtEpFactory* factory,
                                              const OrtHardwareDevice* const* devices,
                                              const OrtKeyValuePairs* const* ep_metadata_pairs,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger, std::unique_ptr<IExecutionProvider>* ep)>;

  EpFactoryInternal(const std::string& ep_name, const std::string& vendor, uint32_t vendor_id,
                    GetSupportedFunc&& get_supported_func,
                    CreateFunc&& create_func);

  const char* GetName() const noexcept { return ep_name_.c_str(); }
  const char* GetVendor() const noexcept { return vendor_.c_str(); }
  uint32_t GetVendorId() const noexcept { return vendor_id_; }
  const char* GetVersion() const noexcept;

  OrtStatus* GetSupportedDevices(_In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                 _In_ size_t num_devices,
                                 _Inout_ OrtEpDevice** ep_devices,
                                 _In_ size_t max_ep_devices,
                                 _Out_ size_t* num_ep_devices) noexcept;

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
                                      _In_ const OrtLogger* logger, _Out_ std::unique_ptr<IExecutionProvider>* ep);

  // Function ORT calls to release an EP instance.
  void ReleaseEp(OrtEp* ep);

 private:
  const std::string ep_name_;                  // EP name library was registered with
  const std::string vendor_;                   // EP vendor name
  const uint32_t vendor_id_;                   // EP vendor ID
  const GetSupportedFunc get_supported_func_;  // function to return supported devices
  const CreateFunc create_func_;               // function to create the EP instance

  std::vector<std::unique_ptr<EpFactoryInternal>> eps_;  // EP instances created by this factory
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
