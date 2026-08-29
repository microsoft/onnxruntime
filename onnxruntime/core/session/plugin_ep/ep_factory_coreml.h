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

#include <algorithm>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"
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

  // Validates that 'devices' contains exactly one NPU, one GPU, or one NPU plus one GPU. Otherwise, returns
  // ORT_INVALID_ARGUMENT. The CPU device is never part of a valid selection as the factory does not claim it.
  // If 'options' does not contain MLComputeUnits, derives the default from the selection:
  // CPUAndNeuralEngine for NPU, CPUAndGPU for GPU, and ALL for NPU plus GPU.
  // A caller-provided value may disable selected accelerators, as CPUOnly does, but it must not enable
  // an accelerator that was not selected.
  // Unrecognized values are left for CoreMLOptions to validate during provider creation.
  static OrtStatus* ValidateDeviceSelectionAndResolveMLComputeUnits(const OrtHardwareDevice* const* devices,
                                                                    size_t num_devices,
                                                                    ProviderOptions& options);

  // Returns whether the factory advertises an OrtEpDevice for a hardware device of this type at this Core ML
  // version. No devices are advertised below Core ML 5 (MINIMUM_COREML_VERSION, the EP's minimum). NPU devices
  // additionally require MINIMUM_COREML_VERSION_FOR_NEURAL_ENGINE_SELECTION. CPU devices are never claimed.
  static bool CanClaimDeviceType(OrtHardwareDeviceType device_type, int32_t coreml_version);

  // Selects the hardware devices for the factory to advertise. The result contains only devices accepted by
  // CanClaimDeviceType, at most one device of each type, and no more than max_ep_devices in total.
  // Core ML's MLComputeUnits selects a device class, not a specific physical device, so the CoreML EP cannot
  // honor selection of a particular GPU or NPU when multiple devices of that type exist. Apple device discovery
  // currently reports at most one device of each type. Additional same-type devices are intentionally ignored.
  static InlinedVector<const OrtHardwareDevice*, 2> SelectDevicesToClaim(
      gsl::span<const OrtHardwareDevice* const> devices, int32_t coreml_version, size_t max_ep_devices);

  // Creates one OrtEpDevice for each selected hardware device. The results are published only after all creations
  // succeed. If a creation fails, the function releases all devices created by earlier calls and leaves ep_devices
  // and p_num_ep_devices unchanged, following ORT's C API convention for failed calls. A successful call always
  // writes the count, including zero when no devices were selected.
  // GetSupportedDevices supplies callbacks for CreateEpDevice and ReleaseEpDevice. On success, create_ep_device must
  // return nullptr and store a valid OrtEpDevice pointer in its output parameter. On failure, it must return an error
  // status without modifying the output parameter.
  // Neither callback may throw: create_ep_device must report failures through its returned status, and
  // release_ep_device must complete cleanup for every device passed to it.
  // selected.size() must not exceed the number of entries available in ep_devices. GetSupportedDevices ensures this
  // by limiting SelectDevicesToClaim to max_ep_devices.
  template <typename CreateEpDeviceFn, typename ReleaseEpDeviceFn>
  static OrtStatus* CreateAndPublishEpDevices(gsl::span<const OrtHardwareDevice* const> selected,
                                              const CreateEpDeviceFn& create_ep_device,
                                              const ReleaseEpDeviceFn& release_ep_device,
                                              OrtEpDevice** ep_devices,
                                              size_t* p_num_ep_devices) {
    InlinedVector<OrtEpDevice*, 2> created;
    // Reserve before creating any devices. When DISABLE_ABSEIL maps InlinedVector to std::vector, this ensures that
    // push_back cannot allocate after a device has been created.
    created.reserve(selected.size());

    for (const OrtHardwareDevice* device : selected) {
      OrtEpDevice* ep_device = nullptr;
      OrtStatus* status = create_ep_device(*device, &ep_device);
      if (status != nullptr) {
        for (OrtEpDevice* created_device : created) {
          release_ep_device(created_device);
        }

        return status;
      }

      created.push_back(ep_device);
    }

    std::copy(created.begin(), created.end(), ep_devices);
    *p_num_ep_devices = created.size();

    return nullptr;
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
};

}  // namespace onnxruntime

#endif  // defined(USE_COREML) && defined(__APPLE__)
