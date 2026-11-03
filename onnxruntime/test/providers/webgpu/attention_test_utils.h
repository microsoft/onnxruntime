// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "test/util/include/default_providers.h"

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/webgpu_context.h"
#endif

namespace onnxruntime::test {

inline bool WebGpuSupportsFp16() {
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  auto ep = DefaultWebGpuExecutionProvider();
  return ep && webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId())
                   .DeviceHasFeature(wgpu::FeatureName::ShaderF16);
#else
  // The plugin EP interface cannot report Dawn device features.
  return false;
#endif
}

inline bool SupportsQualcommCooperativePrefill() {
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    return false;
  }
  const auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
  return context.AdapterInfo().vendor == std::string_view{"qualcomm"} &&
         context.DeviceHasFeature(wgpu::FeatureName::ShaderF16) &&
         context.DeviceHasFeature(wgpu::FeatureName::Subgroups) &&
         context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl) &&
         context.Instance().HasWGSLLanguageFeature(wgpu::WGSLLanguageFeatureName::SubgroupId) &&
         context.AdapterInfo().subgroupMinSize <= 128 && context.AdapterInfo().subgroupMaxSize >= 128;
#else
  return false;
#endif
}

}  // namespace onnxruntime::test
