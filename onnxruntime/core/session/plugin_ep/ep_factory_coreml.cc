// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CoreML can be built on non-Apple platforms to test model conversion. Such builds report Core ML 7 but cannot
// execute models, even when device discovery finds GPUs. Require __APPLE__ so the factory cannot advertise those
// GPUs. On Apple platforms, the factory accepts any discovered GPU regardless of vendor. GPU discovery currently
// supports only Apple Silicon Macs. Support for iOS and Intel Macs remains TODO in
// core/platform/apple/device_discovery.cc.
#if defined(USE_COREML) && defined(__APPLE__)

#include "core/session/plugin_ep/ep_factory_coreml.h"

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/providers/coreml/coreml_provider_factory.h"
#include "core/providers/coreml/coreml_provider_factory_creator.h"
#include "core/providers/coreml/model/host_utils.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

bool CoreMLEpFactory::CanClaimDeviceType(OrtHardwareDeviceType device_type, int32_t coreml_version) {
  // The factory advertises no devices below Core ML 5 (macOS 12 / iOS 15), the minimum version supported by
  // the current CoreML EP. Although earlier Core ML versions can use accelerators, CoreMLExecutionProvider
  // rejects those systems during construction. Advertising a device on those systems would cause session creation
  // to fail after selection instead of allowing the policy to select another EP.
  if (coreml_version < MINIMUM_COREML_VERSION) {
    return false;
  }

  switch (device_type) {
    case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
      return true;

    case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
      // The NPU is claimed only on Core ML 6 or later (iOS 16 / macOS 13): earlier Core ML has no MLComputeUnits
      // mode that enables the Neural Engine without also enabling the GPU (CPUAndNeuralEngine requires Core ML 6),
      // so an NPU selection could not be honored. Core ML can still use the ANE through ALL on those systems.
      // The factory therefore does not advertise the NPU on Core ML 5 systems: iOS 15 devices with an ANE and
      // Apple Silicon Macs running macOS 12.
      return coreml_version >= MINIMUM_COREML_VERSION_FOR_NEURAL_ENGINE_SELECTION;

    default:
      // The CPU device is deliberately not claimed. DEFAULT and PREFER_CPU select a single preferred EP for the CPU
      // device. The NPU- and GPU-preferring policies first select an EP for the accelerator and then use the same CPU
      // selection as a fallback.
      // If CoreML claimed the CPU, it would compete with the ORT CPU EP and other CPU-based EPs to be selected as the
      // primary CPU EP.
      // CoreML-on-CPU remains available through an explicit MLComputeUnits=CPUOnly option. Core ML may also use its
      // internal CPU path for operations unsupported by the selected compute units.
      return false;
  }
}

InlinedVector<const OrtHardwareDevice*, 2> CoreMLEpFactory::SelectDevicesToClaim(
    gsl::span<const OrtHardwareDevice* const> devices, int32_t coreml_version, size_t max_ep_devices) {
  InlinedVector<const OrtHardwareDevice*, 2> selected;
  bool npu_claimed = false;
  bool gpu_claimed = false;

  for (size_t i = 0; i < devices.size() && selected.size() < max_ep_devices; ++i) {
    const OrtHardwareDevice* device = devices[i];

    // CanClaimDeviceType checks the Core ML version and excludes CPU devices.
    if (!CanClaimDeviceType(device->type, coreml_version)) {
      continue;
    }

    // Core ML selects compute-unit classes, not individual physical devices, so advertise at most one NPU and one GPU.
    // Apple device discovery currently reports at most one device of each type. If it later reports several devices
    // of the same type, this function keeps the first one. Additional devices are intentionally ignored because
    // Core ML cannot target a specific physical device.
    // CanClaimDeviceType currently accepts only NPU and GPU devices, so this logic treats every accepted non-NPU
    // device as a GPU. If support for another device type is added, update this selection logic as well.
    const bool is_npu = device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_NPU;
    bool& claimed = is_npu ? npu_claimed : gpu_claimed;

    if (claimed) {
      continue;
    }

    claimed = true;

    selected.push_back(device);
  }

  return selected;
}

OrtStatus* CoreMLEpFactory::GetSupportedDevices(EpFactoryInternal& ep_factory,
                                                const OrtHardwareDevice* const* devices,
                                                size_t num_devices,
                                                OrtEpDevice** ep_devices,
                                                size_t max_ep_devices,
                                                size_t* p_num_ep_devices) noexcept {
  OrtStatus* status = nullptr;

  ORT_TRY {
    const auto selected =
        SelectDevicesToClaim(gsl::make_span(devices, num_devices), coreml::util::CoreMLVersion(), max_ep_devices);

    // Do not attach MLComputeUnits to individual OrtEpDevice instances. Per-device defaults would assign different
    // values to the NPU and GPU, creating a conflict when both are selected. CreateIExecutionProvider instead computes
    // a single value from all selected devices. Any MLComputeUnits value already present in the session options was
    // provided by the caller and is preserved if it is compatible with the selection.
    const auto create_ep_device =
        [&ep_factory](const OrtHardwareDevice& device, OrtEpDevice** ep_device) -> OrtStatus* {
      return OrtExecutionProviderApi::CreateEpDevice(&ep_factory, &device, nullptr, nullptr, ep_device);
    };

    const auto release_ep_device = [](OrtEpDevice* ep_device) {
      OrtExecutionProviderApi::ReleaseEpDevice(ep_device);
    };

    status = CreateAndPublishEpDevices(
        selected,
        create_ep_device,
        release_ep_device,
        ep_devices, p_num_ep_devices);
  }
  ORT_CATCH(const std::exception& ex) {
    // Convert local allocation failures to OrtStatus because this method is noexcept. Device selection and storage
    // reservation finish before CreateEpDevice is called, so this exception path cannot leak an OrtEpDevice.
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, ex.what());
    });
  }

  return status;
}

OrtStatus* CoreMLEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* devices,
                                                     const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                                     size_t num_devices,
                                                     const OrtSessionOptions* session_options,
                                                     const OrtLogger* session_logger,
                                                     std::unique_ptr<IExecutionProvider>* ep) noexcept {
  OrtStatus* status = nullptr;
  *ep = nullptr;

  ORT_TRY {
    ProviderOptions options = GetOptionsFromSessionOptions(session_options->value);

    // OrtEpDevice instances carry no MLComputeUnits default, so any existing session option can only come from
    // the caller. Validate the device selection, derive MLComputeUnits when absent, and reject recognized values
    // that enable an unselected accelerator.
    // CoreMLOptions performs the remaining option validation during provider creation.
    ORT_API_RETURN_IF_ERROR(ValidateDeviceSelectionAndResolveMLComputeUnits(devices, num_devices, options));

    const auto provider_factory = CoreMLProviderFactoryCreator::Create(options);
    *ep = provider_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, ex.what());
    });
  }

  return status;
}

OrtStatus* CoreMLEpFactory::ValidateDeviceSelectionAndResolveMLComputeUnits(
    const OrtHardwareDevice* const* devices, size_t num_devices, ProviderOptions& options) {
  if (num_devices == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CoreML EP factory requires at least one device to be selected.");
  }

  if (devices == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "CoreML device list is null.");
  }

  size_t num_npu = 0;
  size_t num_gpu = 0;
  for (size_t i = 0; i < num_devices; ++i) {
    if (devices[i] == nullptr) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "CoreML device is null.");
    }

    switch (devices[i]->type) {
      case OrtHardwareDeviceType::OrtHardwareDeviceType_NPU:
        ++num_npu;
        break;
      case OrtHardwareDeviceType::OrtHardwareDeviceType_GPU:
        ++num_gpu;
        break;
      default:
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "CoreML EP factory only supports NPU and GPU devices.");
    }
  }

  // Reject multiple devices of the same type. Core ML's MLComputeUnits selects a device class, not a specific
  // physical device, so the CoreML EP cannot honor a selection containing multiple NPUs or GPUs.
  // With current device discovery, such entries can only be duplicates. Future discovery could expose multiple
  // physical devices of one type.
  if (num_npu > 1 || num_gpu > 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CoreML EP factory supports one NPU, one GPU, or one NPU plus one GPU. "
                                 "At most one device of each type can be selected.");
  }

  // This helper does not recheck the Core ML version. CreateIExecutionProvider receives only devices previously
  // advertised by GetSupportedDevices, which uses CanClaimDeviceType to enforce the version requirements.
  // Therefore, a selected NPU implies Core ML 6 or later, so CPUAndNeuralEngine is available.
  // MLComputeUnits is the only option derived from the selection. ModelFormat is not device-specific and changes
  // only when the caller explicitly sets it.
  const auto compute_units_it = options.find(kCoremlProviderOption_MLComputeUnits);
  if (compute_units_it == options.end()) {
    options[kCoremlProviderOption_MLComputeUnits] =
        (num_npu > 0 && num_gpu > 0) ? kCoremlProviderOption_MLComputeUnits_ALL
        : num_npu > 0                ? kCoremlProviderOption_MLComputeUnits_CPUAndNeuralEngine
                                     : kCoremlProviderOption_MLComputeUnits_CPUAndGPU;
    return nullptr;
  }

  // A user-provided MLComputeUnits value may disable selected accelerators (e.g. CPUOnly), but it must not enable
  // an unselected accelerator. Otherwise, PREFER_NPU could be silently redirected to the GPU. This check only
  // enforces compatibility with the selected devices. CoreMLOptions::ValidateAndParseProviderOption later validates
  // the option value itself. If that method is extended to accept a new value that enables an accelerator, update
  // the classification below accordingly.
  const std::string& compute_units = compute_units_it->second;
  const bool enables_npu =
      compute_units == kCoremlProviderOption_MLComputeUnits_CPUAndNeuralEngine ||
      compute_units == kCoremlProviderOption_MLComputeUnits_ALL;
  const bool enables_gpu =
      compute_units == kCoremlProviderOption_MLComputeUnits_CPUAndGPU ||
      compute_units == kCoremlProviderOption_MLComputeUnits_ALL;

  if ((enables_npu && num_npu == 0) || (enables_gpu && num_gpu == 0)) {
    const std::string message = MakeString("MLComputeUnits value '", compute_units,
                                           "' enables a device type that was not selected for the CoreML EP.");

    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
  }

  return nullptr;
}

}  // namespace onnxruntime

#endif  // defined(USE_COREML) && defined(__APPLE__)
