// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_provider_bridge.h"

#include "core/common/status.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/session_options.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_factory_internal.h"

namespace onnxruntime {
class ProviderBridgeEpFactory : public EpFactoryInternalImpl {
 public:
  ProviderBridgeEpFactory(OrtEpFactory& ep_factory, ProviderLibrary& provider_library)
      : EpFactoryInternalImpl(ep_factory.GetName(&ep_factory), ep_factory.GetVendor(&ep_factory)),
        ep_factory_{ep_factory},
        provider_library_{provider_library} {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* num_ep_devices) noexcept override {
    ORT_API_RETURN_IF_ERROR(ep_factory_.GetSupportedDevices(&ep_factory_, devices, num_devices, ep_devices,
                                                            max_ep_devices, num_ep_devices));

    // add the EpFactoryInternal layer back in so that we can redirect to CreateIExecutionProvider.
    for (size_t i = 0; i < *num_ep_devices; ++i) {
      auto* ep_device = ep_devices[i];
      if (ep_device) {
        ep_device->ep_factory = &ep_factory;
      }
    }

    return nullptr;
  }

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                      const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override {
    // get the provider specific options
    auto ep_options = GetOptionsFromSessionOptions(session_options->value);
    auto& provider = provider_library_.Get();

    auto status = provider.CreateIExecutionProvider(devices, ep_metadata_pairs, num_devices,
                                                    ep_options, *session_options, *session_logger, *ep);

    return ToOrtStatus(status);
  }

  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtEp* ep,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator) noexcept override {
    return ep_factory_.CreateAllocator(&ep_factory_, memory_info, ep, allocator_options, allocator);
  }

  void ReleaseAllocator(OrtAllocator* allocator) noexcept override {
    ep_factory_.ReleaseAllocator(&ep_factory_, allocator);
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) override {
    return ep_factory_.CreateDataTransfer(&ep_factory_, data_transfer);
  }

  bool IsStreamAware() const noexcept override {
    return ep_factory_.IsStreamAware(&ep_factory_);
  }

  OrtStatus* CreateSyncStreamForDevice(const OrtMemoryDevice* device,
                                       const OrtKeyValuePairs* stream_options,
                                       OrtSyncStreamImpl** stream) noexcept override {
    return ep_factory_.CreateSyncStreamForDevice(&ep_factory_, device, /*OrtEp*/ nullptr, stream_options,
                                                 stream);
  }

  OrtEpFactory& ep_factory_;           // OrtEpFactory from the provider bridge EP
  ProviderLibrary& provider_library_;  // ProviderLibrary from the provider bridge EP
};

Status EpLibraryProviderBridge::Load() {
  std::lock_guard<std::mutex> lock{mutex_};

  if (!factories_.empty()) {
    // already loaded
    return Status::OK();
  }

  // if we have been unloaded we can't just be reloaded.
  if (!ep_library_plugin_ || !provider_library_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "EpLibraryProviderBridge has been unloaded. "
                           "Please create a new instance using LoadPluginOrProviderBridge.");
  }

  // wrap the EpLibraryPlugin factories that were created via calling CreateEpFactories in the library.
  // use GetSupportedDevices from the library's factory.
  // to do this we need to capture `factory` and plug it in to is_supported_fn and create_fn.
  // we also need to update any returned OrtEpDevice instances to swap the wrapper EpFactoryInternal in so that we can
  // call Provider::CreateIExecutionProvider in EpFactoryInternal::CreateIExecutionProvider.
  for (const auto& factory : ep_library_plugin_->GetFactories()) {
    auto factory_impl = std::make_unique<ProviderBridgeEpFactory>(*factory, *provider_library_);
    auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(factory_impl));

    factory_ptrs_.push_back(internal_factory.get());
    internal_factory_ptrs_.push_back(internal_factory.get());
    factories_.push_back(std::move(internal_factory));
  }

  return Status::OK();
}

Status EpLibraryProviderBridge::Unload() {
  std::lock_guard<std::mutex> lock{mutex_};

  internal_factory_ptrs_.clear();
  factory_ptrs_.clear();
  factories_.clear();

  // we loaded ep_library_plugin_ after provider_library_ in LoadPluginOrProviderBridge so do the reverse order here.
  ORT_RETURN_IF_ERROR(ep_library_plugin_->Unload());
  ep_library_plugin_ = nullptr;

  provider_library_->Unload();
  provider_library_ = nullptr;

  return Status::OK();
}

}  // namespace onnxruntime
