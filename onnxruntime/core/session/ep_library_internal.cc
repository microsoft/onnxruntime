// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_internal.h"

#include "core/framework/session_options.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_apis.h"

#if defined(USE_DML)
#include "core/providers/dml/dml_provider_factory_creator.h"
#endif

#if defined(USE_WEBGPU)
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#endif

namespace onnxruntime {
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateCpuEp() {
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      return true;
    }

    return false;
  };

  const auto create_cpu_ep = [](const OrtHardwareDevice* const* /*devices*/,
                                const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                size_t num_devices,
                                const OrtSessionOptions* session_options,
                                const OrtLogger* session_logger,
                                std::unique_ptr<IExecutionProvider>* ep) -> OrtStatus* {
    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "CPU EP factory currently only supports one device at a time.");
    }

    CPUExecutionProviderInfo epi{session_options->value.enable_cpu_mem_arena};
    *ep = std::make_unique<CPUExecutionProvider>(epi);
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  };

  std::string ep_name = kCpuExecutionProvider;
  auto cpu_factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_cpu_ep);
  return std::make_unique<EpLibraryInternal>(std::move(cpu_factory));
}

#if defined(USE_DML)
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateDmlEp() {
  static const std::string ep_name = kDmlExecutionProvider;
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** ep_options) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // TODO: Should we ignore a user provided 'device_id' when they select an OrtEpDevice as that is associated with
      //       a specific device.
      //       How would we know what options should not allow user overrides if set in OrtEpDevice?
      if (auto it = device->metadata.entries.find("DxgiAdapterNumber"); it != device->metadata.entries.end()) {
        auto options = std::make_unique<OrtKeyValuePairs>();
        options->Add("device_id", it->second.c_str());
        *ep_options = options.release();
      }

      return true;
    }

    return false;
  };

  const auto create_dml_ep = [](const OrtHardwareDevice* const* /*devices*/,
                                const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                size_t num_devices,
                                const OrtSessionOptions* session_options,
                                const OrtLogger* session_logger,
                                std::unique_ptr<IExecutionProvider>* ep) -> OrtStatus* {
    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "DML EP factory currently only supports one device at a time.");
    }

    auto ep_options = GetOptionsFromSessionOptions(ep_name, session_options->value);
    auto dml_ep_factory = DMLProviderFactoryCreator::CreateFromProviderOptions(session_options->value.config_options,
                                                                               ep_options);

    *ep = dml_ep_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  };

  auto dml_factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_dml_ep);

  return std::make_unique<EpLibraryInternal>(std::move(dml_factory));
}
#endif

#if defined(USE_WEBGPU)
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateWebGpuEp() {
  static const std::string ep_name = kWebGpuExecutionProvider;

  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // What is the correct behavior here to match the device if there are multiple GPUs?
      // Should WebGPU default to picking the GPU with HighPerformanceIndex of 0?
      // Or should we be setting the 'deviceId', 'webgpuInstance' and 'webgpuDevice' options for each GPU?
      return true;
    }

    return false;
  };

  const auto create_webgpu_ep = [](const OrtHardwareDevice* const* /*devices*/,
                                   const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                   size_t num_devices,
                                   const OrtSessionOptions* session_options,
                                   const OrtLogger* session_logger,
                                   std::unique_ptr<IExecutionProvider>* ep) -> OrtStatus* {
    if (num_devices != 1) {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "WebGPU EP factory currently only supports one device at a time.");
    }

    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(session_options->value.config_options);
    *ep = webgpu_ep_factory->CreateProvider();
    (*ep)->SetLogger(session_logger->ToInternal());

    return nullptr;
  };

  auto webgpu_factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_webgpu_ep);

  return std::make_unique<EpLibraryInternal>(std::move(webgpu_factory));
}
#endif

std::vector<std::unique_ptr<EpLibraryInternal>> EpLibraryInternal::CreateInternalEps() {
  std::vector<std::unique_ptr<EpLibraryInternal>> internal_eps;
  internal_eps.reserve(4);

  // CPU EP
  internal_eps.push_back(CreateCpuEp());

#if defined(USE_DML)
  internal_eps.push_back(CreateDmlEp());
#endif

#if defined(USE_WEBGPU)
  internal_eps.push_back(CreateWebGpuEp());
#endif

  return internal_eps;
}

}  // namespace onnxruntime
