// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_provider_bridge.h"

#include "core/framework/session_options.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/shared_library/provider_host_api.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_factory_internal.h"

namespace onnxruntime {

std::unique_ptr<EpFactoryInternal> EpLibraryProviderBridge::CreateCudaEpFactory(Provider& provider) {
  // Using the name that SessionOptionsAppendExecutionProvider uses to identify the EP as that matches the
  // expected name in the configuration options. must be static to be valid for the lambdas
  static const std::string ep_name = "CUDA";

  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU &&
        device->vendor_id == 0x10de) {
      return true;
    }

    return false;
  };

  const auto create_cuda_ep = [&provider](const OrtSessionOptions& session_options,
                                          const OrtLogger& session_logger) {
    OrtCUDAProviderOptionsV2 options;
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto ep_options = GetOptionsFromSessionOptions(ep_name, so);
    provider.UpdateProviderOptions(&options, ep_options);

    auto ep_factory = provider.CreateExecutionProviderFactory(&options);
    auto ep = ep_factory->CreateProvider(session_options, session_logger);

    return ep;
  };

  auto factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_cuda_ep);
  return factory;
}

Status EpLibraryProviderBridge::Load() {
  auto& provider = provider_library_.Get();
  if (registration_name_ == "CUDA") {
    auto ep_factory = CreateCudaEpFactory(provider);
    factory_ptrs_.push_back(ep_factory.get());
    internal_factory_ptrs_.push_back(ep_factory.get());
    factories_.push_back(std::move(ep_factory));
  } else {
    ORT_NOT_IMPLEMENTED("Execution provider library is not supported: ", library_path_);
  }

  return Status::OK();
}

Status EpLibraryProviderBridge::Unload() {
  provider_library_.Unload();
  return Status::OK();
}

}  // namespace onnxruntime
