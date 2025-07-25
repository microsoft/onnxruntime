// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_WEBGPU)
#include "core/session/plugin_ep/ep_factory_internal_impl.h"

// #include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
// #include "core/providers/webgpu/webgpu_provider_factory_creator.h"
// #include "core/session/abi_devices.h"
// #include "core/session/abi_logger.h"
// #include "core/session/abi_session_options_impl.h"
// #include "core/session/plugin_ep/ep_api.h"
// #include "core/session/ort_apis.h"

namespace onnxruntime {

class WebGpuEpFactory : public EpFactoryInternalImpl {
 public:
  WebGpuEpFactory() : EpFactoryInternalImpl(kWebGpuExecutionProvider, "Microsoft", OrtDevice::VendorIds::MICROSOFT) {
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

  /* TODO: Implement CreateAllocator and CreateDataTransfer to support shared allocators and data transfer outside of
           an InferenceSession.
  OrtStatus* CreateAllocator(const OrtMemoryInfo* memory_info,
                             const OrtKeyValuePairs* allocator_options,
                             OrtAllocator** allocator) noexcept override {
    *allocator = device_allocators[memory_info->device.Id()].get();
  }

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) override {
    // TODO: Wrap the IDataTransfer implementation so we can copy to device using OrtApi CopyTensors.
    *data_transfer = nullptr;
    return nullptr;
  }
  */
};
}  // namespace onnxruntime

#endif  // USE_WEBGPU
