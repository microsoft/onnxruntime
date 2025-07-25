// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_DML)

#include "core/session/plugin_ep/ep_factory_internal_impl.h"

// #include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
// #include "core/providers/dml/dml_provider_factory_creator.h"
// #include "core/session/abi_devices.h"
// #include "core/session/abi_logger.h"
// #include "core/session/abi_session_options_impl.h"
// #include "core/session/plugin_ep/ep_api.h"
// #include "core/session/ort_apis.h"

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

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                      const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override;

  OrtStatus* CreateAllocator(const OrtMemoryInfo* /*memory_info*/,
                             const OrtKeyValuePairs* /*allocator_options*/,
                             OrtAllocator** allocator) noexcept override;

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept override;

  std::vector<std::unique_ptr<OrtMemoryInfo>> device_memory_infos;  // memory info for each device
  std::vector<std::unique_ptr<OrtAllocator>> device_allocators;     // allocators for each device
};

}  // namespace onnxruntime

#endif  // USE_DML
