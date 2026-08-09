// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "core/graph/constants.h"

namespace onnxruntime {

class WebGpuEpFactory : public EpFactoryInternalImpl {
 public:
  // allow_virtual_devices is captured at construction (from the OrtEnv config "allow_virtual_devices")
  // rather than queried from the OrtEnv singleton inside GetSupportedDevices: internal EPs are
  // registered while the OrtEnv creation mutex is already held on this thread, so querying the singleton
  // there would self-deadlock.
  explicit WebGpuEpFactory(bool allow_virtual_devices)
      : EpFactoryInternalImpl(kWebGpuExecutionProvider, "Microsoft", OrtDevice::VendorIds::MICROSOFT),
        allow_virtual_devices_{allow_virtual_devices} {
  }

  ~WebGpuEpFactory() override;

 private:
  OrtStatus* GetSupportedDevices(EpFactoryInternal& ep_factory,
                                 const OrtHardwareDevice* const* devices,
                                 size_t num_devices,
                                 OrtEpDevice** ep_devices,
                                 size_t max_ep_devices,
                                 size_t* p_num_ep_devices) noexcept override;

  const bool allow_virtual_devices_;

  // Owned virtual GPU hardware device created on demand by GetSupportedDevices when the environment
  // allows virtual devices (OrtEnv config "allow_virtual_devices"=1) and no real GPU device was
  // discovered -- e.g. in a sandboxed process where OS device enumeration is unavailable (such as
  // Win32k lockdown), so ORT device discovery is skipped. Released in the destructor. nullptr when
  // no virtual device was made.
  OrtHardwareDevice* virtual_hw_device_ = nullptr;

  OrtStatus* CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                      const OrtKeyValuePairs* const* ep_metadata_pairs,
                                      size_t num_devices,
                                      const OrtSessionOptions* session_options,
                                      const OrtLogger* session_logger,
                                      std::unique_ptr<IExecutionProvider>* ep) noexcept override;

  OrtStatus* CreateDataTransfer(_Outptr_result_maybenull_ OrtDataTransferImpl** data_transfer) noexcept override;
};
}  // namespace onnxruntime

#endif  // defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
