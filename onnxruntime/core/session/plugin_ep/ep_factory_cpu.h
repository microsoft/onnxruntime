// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "core/graph/constants.h"

namespace onnxruntime {

class CpuEpFactory : public EpFactoryInternalImpl {
 public:
  CpuEpFactory() : EpFactoryInternalImpl(kCpuExecutionProvider, "Microsoft", OrtDevice::VendorIds::MICROSOFT) {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override;

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override;
};
}  // namespace onnxruntime
