// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_factory_internal.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ep_api_utils.h"
#include "core/session/ort_apis.h"
#include "onnxruntime_config.h"  // for ORT_VERSION

namespace onnxruntime {

using Forward = ForwardToFactory<EpFactoryInternal>;

EpFactoryInternal::EpFactoryInternal(const std::string& ep_name, const std::string& vendor,
                                     GetSupportedFunc&& get_supported_func,
                                     CreateFunc&& create_func)
    : ep_name_{ep_name},
      vendor_{vendor},
      get_supported_func_{std::move(get_supported_func)},
      create_func_{create_func} {
  ort_version_supported = ORT_API_VERSION;

  OrtEpFactory::GetName = Forward::GetFactoryName;
  OrtEpFactory::GetVendor = Forward::GetVendor;
  OrtEpFactory::GetVersion = Forward::GetVersion;
  OrtEpFactory::GetSupportedDevices = Forward::GetSupportedDevices;
  OrtEpFactory::CreateEp = Forward::CreateEp;
  OrtEpFactory::ReleaseEp = Forward::ReleaseEp;
}

const char* EpFactoryInternal::GetVersion() const noexcept {
  return ORT_VERSION;
}

OrtStatus* EpFactoryInternal::GetSupportedDevices(const OrtHardwareDevice* const* devices,
                                                  size_t num_devices,
                                                  OrtEpDevice** ep_devices,
                                                  size_t max_ep_devices,
                                                  size_t* num_ep_devices) noexcept {
  return get_supported_func_(this, devices, num_devices, ep_devices, max_ep_devices, num_ep_devices);
}

OrtStatus* EpFactoryInternal::CreateEp(const OrtHardwareDevice* const* /*devices*/,
                                       const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                       size_t /*num_devices*/,
                                       const OrtSessionOptions* /*api_session_options*/,
                                       const OrtLogger* /*api_logger*/,
                                       OrtEp** /*ep*/) {
  ORT_THROW("Internal error. CreateIExecutionProvider should be used for EpFactoryInternal.");
}

OrtStatus* EpFactoryInternal::CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                                       const OrtKeyValuePairs* const* ep_metadata_pairs,
                                                       size_t num_devices,
                                                       const OrtSessionOptions* session_options,
                                                       const OrtLogger* session_logger,
                                                       std::unique_ptr<IExecutionProvider>* ep) {
  *ep = nullptr;

  if (num_devices != 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpFactoryInternal currently only supports one device at a time.");
  }

  return create_func_(this, devices, ep_metadata_pairs, num_devices, session_options, session_logger, ep);
}

void EpFactoryInternal::ReleaseEp(OrtEp* /*ep*/) {
  // we never create an OrtEp so we should never be trying to release one
  ORT_THROW("Internal error. No ReleaseEp call is required for EpFactoryInternal.");
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
