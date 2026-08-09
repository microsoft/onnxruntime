// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_factory_cpu.h"

#include "core/framework/error_code_helper.h"
#include "core/graph/constants.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_logger.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_factory_internal.h"
#include "core/session/ort_apis.h"

namespace onnxruntime {

OrtStatus* CpuEpFactory::GetSupportedDevices(EpFactoryInternal& ep_factory,
                                             const OrtHardwareDevice* const* devices,
                                             size_t num_devices,
                                             OrtEpDevice** ep_devices,
                                             size_t max_ep_devices,
                                             size_t* p_num_ep_devices) noexcept {
  size_t& num_ep_devices = *p_num_ep_devices;
  num_ep_devices = 0;

  for (size_t i = 0; i < num_devices && num_ep_devices < max_ep_devices; ++i) {
    const OrtHardwareDevice& device = *devices[i];
    if (device.type == OrtHardwareDeviceType::OrtHardwareDeviceType_CPU) {
      ORT_API_RETURN_IF_ERROR(
          OrtExecutionProviderApi::CreateEpDevice(&ep_factory, &device, nullptr, nullptr,
                                                  &ep_devices[num_ep_devices++]));
    }
  }

  return nullptr;
}

OrtStatus* CpuEpFactory::CreateIExecutionProvider(const OrtHardwareDevice* const* /*devices*/,
                                                  const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                                                  size_t num_devices,
                                                  const OrtSessionOptions* session_options,
                                                  const OrtLogger* session_logger,
                                                  std::unique_ptr<IExecutionProvider>* ep) noexcept {
  if (num_devices != 1) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CPU EP factory currently only supports one device at a time.");
  }

  CPUExecutionProviderInfo epi{session_options->value.enable_cpu_mem_arena};
  *ep = std::make_unique<CPUExecutionProvider>(epi);
  (*ep)->SetLogger(session_logger->ToInternal());

  return nullptr;
}
}  // namespace onnxruntime
