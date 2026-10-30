// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_factory_internal.h"

#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/forward_to_factory_impl.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {
using Forward = ForwardToFactoryImpl<EpFactoryInternal>;

EpFactoryInternal::EpFactoryInternal(std::unique_ptr<EpFactoryInternalImpl> impl)
    : OrtEpFactory{},  // Ensure optional functions are default initialized to nullptr
      impl_{std::move(impl)} {
  ort_version_supported = ORT_API_VERSION;

  OrtEpFactory::GetName = Forward::GetFactoryName;
  OrtEpFactory::GetVendor = Forward::GetVendor;
  OrtEpFactory::GetVendorId = Forward::GetVendorId;
  OrtEpFactory::GetVersion = Forward::GetVersion;
  OrtEpFactory::GetSupportedDevices = Forward::GetSupportedDevices;
  OrtEpFactory::CreateEp = Forward::CreateEp;
  OrtEpFactory::ReleaseEp = Forward::ReleaseEp;
  OrtEpFactory::ValidateCompiledModelCompatibilityInfo = Forward::ValidateCompiledModelCompatibilityInfo;
  OrtEpFactory::CreateAllocator = Forward::CreateAllocator;
  OrtEpFactory::ReleaseAllocator = Forward::ReleaseAllocator;
  OrtEpFactory::CreateDataTransfer = Forward::CreateDataTransfer;
  OrtEpFactory::IsStreamAware = Forward::IsStreamAware;
  OrtEpFactory::CreateSyncStreamForDevice = Forward::CreateSyncStreamForDevice;
  OrtEpFactory::CreateExternalResourceImporterForDevice = Forward::CreateExternalResourceImporterForDevice;
  OrtEpFactory::GetHardwareDeviceIncompatibilityDetails = Forward::GetHardwareDeviceIncompatibilityDetails;
  OrtEpFactory::InitGraphicsInterop = Forward::InitGraphicsInterop;
  OrtEpFactory::DeinitGraphicsInterop = Forward::DeinitGraphicsInterop;
  OrtEpFactory::SelectBestModelCandidate = Forward::SelectBestModelCandidate;
}

OrtStatus* EpFactoryInternal::CreateIExecutionProvider(gsl::span<const OrtEpDevice* const> ep_devices,
                                                       _In_ const OrtSessionOptions* session_options,
                                                       _In_ const OrtLogger* logger,
                                                       _Out_ std::unique_ptr<IExecutionProvider>* ep) {
  InlinedVector<const OrtHardwareDevice*> hardware_devices;
  InlinedVector<const OrtKeyValuePairs*> ep_metadata;
  hardware_devices.reserve(ep_devices.size());
  ep_metadata.reserve(ep_devices.size());

  for (const OrtEpDevice* ep_device : ep_devices) {
    hardware_devices.push_back(ep_device->device);
    ep_metadata.push_back(&ep_device->ep_metadata);
  }

  ORT_API_RETURN_IF_ERROR(impl_->CreateIExecutionProvider(hardware_devices.data(), ep_metadata.data(),
                                                          ep_devices.size(), session_options, logger, ep));

  if (*ep != nullptr) {
    (*ep)->SetEpDevices(ep_devices);
  }

  return nullptr;
}

InternalExecutionProviderFactory::InternalExecutionProviderFactory(EpFactoryInternal& ep_factory,
                                                                   gsl::span<const OrtEpDevice* const> ep_devices)
    : ep_factory_{ep_factory}, ep_devices_{ep_devices.begin(), ep_devices.end()} {
}

std::unique_ptr<IExecutionProvider>
InternalExecutionProviderFactory::CreateProvider(const OrtSessionOptions& session_options,
                                                 const OrtLogger& session_logger) {
  std::unique_ptr<IExecutionProvider> ep;
  auto status = ToStatusAndRelease(ep_factory_.CreateIExecutionProvider(ep_devices_, &session_options,
                                                                        &session_logger, &ep));
  if (!status.IsOK()) {
    ORT_THROW("Error creating execution provider: ", status);
  }

  return ep;
}
}  // namespace onnxruntime
