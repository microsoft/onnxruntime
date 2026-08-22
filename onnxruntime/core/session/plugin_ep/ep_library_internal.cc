// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_library_internal.h"
#include "core/session/plugin_ep/ep_factory_cpu.h"
#include "core/session/plugin_ep/ep_factory_dml.h"
#include "core/session/plugin_ep/ep_factory_webgpu.h"

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#endif

namespace onnxruntime {

std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateCpuEp() {
  auto cpu_factory_impl = std::make_unique<CpuEpFactory>();
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(cpu_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}

#if defined(USE_DML)

std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateDmlEp() {
  auto dml_factory_impl = std::make_unique<DmlEpFactory>();
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(dml_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}
#endif

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
std::unique_ptr<EpLibraryInternal> EpLibraryInternal::CreateWebGpuEp(
    bool allow_virtual_devices,
    const webgpu::WebGpuDeviceConfig& device_config) {
  auto webgpu_factory_impl = std::make_unique<WebGpuEpFactory>(allow_virtual_devices, device_config);
  auto internal_factory = std::make_unique<EpFactoryInternal>(std::move(webgpu_factory_impl));
  return std::make_unique<EpLibraryInternal>(std::move(internal_factory));
}
#endif

std::vector<std::unique_ptr<EpLibraryInternal>> EpLibraryInternal::CreateInternalEps(
    bool allow_virtual_devices,
    const OrtKeyValuePairs& environment_options) {
  std::vector<std::unique_ptr<EpLibraryInternal>> internal_eps;
  internal_eps.reserve(4);

  // CPU EP
  internal_eps.push_back(CreateCpuEp());

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  internal_eps.push_back(
      CreateWebGpuEp(allow_virtual_devices, ParseWebGpuDeviceConfig(environment_options)));
#else
  ORT_UNUSED_PARAMETER(allow_virtual_devices);
  ORT_UNUSED_PARAMETER(environment_options);
#endif

#if defined(USE_DML)
  internal_eps.push_back(CreateDmlEp());
#endif

  return internal_eps;
}

}  // namespace onnxruntime
