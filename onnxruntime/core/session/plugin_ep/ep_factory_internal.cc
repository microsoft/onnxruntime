// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_factory_internal.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/forward_to_factory_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
using Forward = ForwardToFactoryImpl<EpFactoryInternal>;

EpFactoryInternal::EpFactoryInternal(std::unique_ptr<EpFactoryInternalImpl> impl)
    : impl_{std::move(impl)} {
  ort_version_supported = ORT_API_VERSION;

  OrtEpFactory::GetName = Forward::GetFactoryName;
  OrtEpFactory::GetVendor = Forward::GetVendor;
  OrtEpFactory::GetVendorId = Forward::GetVendorId;
  OrtEpFactory::GetVersion = Forward::GetVersion;
  OrtEpFactory::GetSupportedDevices = Forward::GetSupportedDevices;
  OrtEpFactory::CreateEp = Forward::CreateEp;
  OrtEpFactory::ReleaseEp = Forward::ReleaseEp;
  OrtEpFactory::CreateAllocator = Forward::CreateAllocator;
  OrtEpFactory::ReleaseAllocator = Forward::ReleaseAllocator;
  OrtEpFactory::CreateDataTransfer = Forward::CreateDataTransfer;
  OrtEpFactory::IsStreamAware = Forward::IsStreamAware;
  OrtEpFactory::CreateSyncStreamForDevice = Forward::CreateSyncStreamForDevice;
}

InternalExecutionProviderFactory::InternalExecutionProviderFactory(EpFactoryInternal& ep_factory,
                                                                   gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory} {
  devices_.reserve(ep_devices.size());
  ep_metadata_.reserve(ep_devices.size());

  for (const auto* ep_device : ep_devices) {
    devices_.push_back(ep_device->device);
    ep_metadata_.push_back(&ep_device->ep_metadata);
  }
}

std::unique_ptr<IExecutionProvider>
InternalExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                                 const OrtLogger& session_logger) {
  std::unique_ptr<IExecutionProvider> ep;
  auto status = ToStatusAndRelease(ep_factory_.CreateIExecutionProvider(devices_.data(), ep_metadata_.data(),
                                                                        devices_.size(), &session_options,
                                                                        &session_logger, &ep));
  if (!status.IsOK()) {
    ORT_THROW("Error creating execution provider: ", status);
  }

  return ep;
}
}  // namespace onnxruntime
