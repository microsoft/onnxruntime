// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
  using IsSupportedFunc = std::function<bool(const OrtHardwareDevice* device,
                                             OrtKeyValuePairs** ep_metadata,
                                             OrtKeyValuePairs** ep_options)>;

  using CreateFunc = std::function<OrtStatus*(const OrtHardwareDevice* const* devices,
                                              const OrtKeyValuePairs* const* ep_metadata_pairs,
                                              size_t num_devices,
                                              const OrtSessionOptions* session_options,
                                              const OrtLogger* logger, std::unique_ptr<IExecutionProvider>* ep)>;

  EpFactoryInternal(const std::string& ep_name, const std::string& vendor,
                    IsSupportedFunc&& is_supported_func,
                    CreateFunc&& create_func);

  const char* GetName() const { return ep_name_.c_str(); }
  const char* GetVendor() const { return vendor_.c_str(); }

  bool GetDeviceInfoIfSupported(_In_ const OrtHardwareDevice* device,
                                _Out_ OrtKeyValuePairs** ep_device_metadata,
                                _Out_ OrtKeyValuePairs** ep_options_for_device) const;

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
  const std::string ep_name_;                // EP name library was registered with
  const std::string vendor_;                 // EP vendor name
  const IsSupportedFunc is_supported_func_;  // function to check if the device is supported
  const CreateFunc create_func_;             // function to create the EP instance

  std::vector<std::unique_ptr<EpFactoryInternal>> eps_;  // EP instances created by this factory
};

// IExecutionProviderFactory for EpFactoryInternal that is required for SessionOptionsAppendExecutionProvider_V2
struct InternalExecutionProviderFactory : public IExecutionProviderFactory {
 public:
  InternalExecutionProviderFactory(EpFactoryInternal& ep_factory, const std::vector<const OrtEpDevice*>& ep_devices);

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
