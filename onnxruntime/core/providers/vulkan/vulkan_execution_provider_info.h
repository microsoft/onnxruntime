// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ortdevice.h"
#include "core/framework/provider_options.h"

namespace onnxruntime {

struct VulkanExecutionProviderInfo {
  VulkanExecutionProviderInfo() = default;

  VulkanExecutionProviderInfo(const ProviderOptions&) {
    // future: parse ProviderOptions
  }

  OrtDevice::DeviceId device_id{0};

  //static VulkanExecutionProviderInfo FromProviderOptions(const ProviderOptions& options);
  //static ProviderOptions ToProviderOptions(const VulkanExecutionProviderInfo& info);
  //static ProviderOptions ToProviderOptions(const OrtVulkanProviderOptions& info);
};

}  // namespace onnxruntime