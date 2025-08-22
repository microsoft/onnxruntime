// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <optional>
#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/provider_bridge_library.h"

namespace onnxruntime {
class ProviderBridgeEpFactory : public EpFactoryInternalImpl {
 public:
  ProviderBridgeEpFactory(OrtEpFactory& ep_factory, ProviderLibrary& provider_library,
                          std::optional<std::filesystem::path> library_path = std::nullopt)
      : EpFactoryInternalImpl(ep_factory.GetName(&ep_factory),
                              ep_factory.GetVendor(&ep_factory),
                              ep_factory.GetVendorId(&ep_factory)),
        ep_factory_{ep_factory},
        provider_library_{provider_library},
        library_path_{std::move(library_path)} {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* num_ep_devices) noexcept override;

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                      const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override;

  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator) noexcept override {
    return ep_factory_.CreateAllocator(&ep_factory_, memory_info, allocator_options, allocator);
  }

  void ReleaseAllocator(OrtAllocator* allocator) noexcept override {
    ep_factory_.ReleaseAllocator(&ep_factory_, allocator);
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept override {
    return ep_factory_.CreateDataTransfer(&ep_factory_, data_transfer);
  }

  bool IsStreamAware() const noexcept override {
    return ep_factory_.IsStreamAware(&ep_factory_);
  }

  OrtStatus* CreateSyncStreamForDevice(const OrtMemoryDevice* device,
                                       const OrtKeyValuePairs* stream_options,
                                       OrtSyncStreamImpl** stream) noexcept override {
    return ep_factory_.CreateSyncStreamForDevice(&ep_factory_, device, stream_options, stream);
  }

  OrtEpFactory& ep_factory_;           // OrtEpFactory from the provider bridge EP
  ProviderLibrary& provider_library_;  // ProviderLibrary from the provider bridge EP
  std::optional<std::filesystem::path> library_path_;  // Library path for EP metadata
};

}  // namespace onnxruntime
