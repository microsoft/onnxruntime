// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_DML)

#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "core/graph/constants.h"

namespace onnxruntime {

class DmlEpFactory : public EpFactoryInternalImpl {
 public:
  DmlEpFactory() : EpFactoryInternalImpl(kDmlExecutionProvider, "Microsoft", OrtDevice::VendorIds::MICROSOFT) {
  }

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override;

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                      const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override;

  std::vector<std::unique_ptr<OrtMemoryInfo>> device_memory_infos;  // memory info for each device
  std::vector<std::unique_ptr<OrtAllocator>> device_allocators;     // allocators for each device
};

}  // namespace onnxruntime

#endif  // USE_DML
