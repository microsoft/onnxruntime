// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/ep_library_internal.h"

#include "core/framework/session_options.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"

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

  const auto create_cpu_ep = [](const OrtSessionOptions& session_options,
                                const OrtLogger& session_logger) {
    CPUExecutionProviderInfo epi{session_options.value.enable_cpu_mem_arena};
    auto ep = std::make_unique<CPUExecutionProvider>(epi);
    ep->SetLogger(session_logger.ToInternal());
    return ep;
  };

  std::string ep_name = "CPU";
  auto cpu_factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_cpu_ep);
  return std::make_unique<EpLibraryInternal>(std::move(cpu_factory));
}

#if defined(USE_DML)
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateDmlEp() {
  static const std::string ep_name = "DML";
  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // does anything need to be added here?
      // is it possible to get the PCI bus number from OrtHardwareDevice? Is that 1:1 with 'device_id'
      return true;
    }

    return false;
  };

  const auto create_dml_ep = [](const OrtSessionOptions& session_options,
                                const OrtLogger& session_logger) {
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto ep_options = GetOptionsFromSessionOptions(ep_name, so);
    auto dml_ep_factory = DMLProviderFactoryCreator::CreateFromProviderOptions(so.config_options,
                                                                               ep_options);

    auto dml_ep = dml_ep_factory->CreateProvider();
    dml_ep->SetLogger(session_logger.ToInternal());
    return dml_ep;
  };

  auto dml_factory = std::make_unique<EpFactoryInternal>(ep_name, "Microsoft", is_supported, create_dml_ep);

  return std::make_unique<EpLibraryInternal>(std::move(dml_factory));
}
#endif

#if defined(USE_WEBGPU)
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateWebGpuEp() {
  static const std::string ep_name = "WebGPU";

  const auto is_supported = [](const OrtHardwareDevice* device,
                               OrtKeyValuePairs** /*ep_metadata*/,
                               OrtKeyValuePairs** /*ep_options*/) -> bool {
    if (device->type == OrtHardwareDeviceType::OrtHardwareDeviceType_GPU) {
      // does anything need to be added here?
      return true;
    }

    return false;
  };

  const auto create_webgpu_ep = [](const OrtSessionOptions& session_options,
                                   const OrtLogger& session_logger) {
    const SessionOptions& so = session_options.existing_value ? **session_options.existing_value
                                                              : session_options.value;

    auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(so.config_options);
    auto webgpu_ep = webgpu_ep_factory->CreateProvider();
    webgpu_ep->SetLogger(session_logger.ToInternal());
    return webgpu_ep;
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
