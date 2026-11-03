// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// CoreML can be built on non-Apple platforms to test model conversion. Such builds report Core ML 7 but cannot
// execute models, even when device discovery finds GPUs. Require __APPLE__ so the factory cannot advertise those
// GPUs. On Apple platforms, the factory accepts any discovered GPU regardless of vendor. GPU discovery currently
// supports only Apple Silicon Macs. Support for iOS and Intel Macs remains TODO in
// core/platform/apple/device_discovery.cc.
#if defined(USE_COREML) && defined(__APPLE__)

#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "core/common/common.h"
#include "core/framework/ortdevice.h"
#include "core/graph/constants.h"

namespace onnxruntime {

class CoreMLEpFactory : public EpFactoryInternalImpl {
 public:
  // Report "Microsoft" and VendorIds::MICROSOFT as the factory vendor identifiers, like the other internal
  // factories (CPU, DML, and WebGPU): the factory vendor identifies who provides the EP implementation, not the
  // hardware vendor. Discovered Apple hardware devices keep their own Apple vendor identifiers.
  // Because the factory vendor does not match the Apple hardware vendor, device ordering gives CoreML no vendor
  // affinity. When multiple EPs target the same NPU or GPU with equal vendor affinity, they are ordered by
  // EP name.
  CoreMLEpFactory()
      : EpFactoryInternalImpl(kCoreMLExecutionProvider, "Microsoft", OrtDevice::VendorIds::MICROSOFT) {}

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CoreMLEpFactory);

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
                                      std::unique_ptr<IExecutionProvider>* ep) override;
};

}  // namespace onnxruntime

#endif  // defined(USE_COREML) && defined(__APPLE__)
