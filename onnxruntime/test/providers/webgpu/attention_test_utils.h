// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <string>
#include <string_view>

#include "gtest/gtest.h"

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

inline bool SupportsCooperativeDispatchObservation() {
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  if (SupportsQualcommCooperativePrefill()) {
    auto ep = DefaultWebGpuExecutionProvider();
    return webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId())
        .DeviceHasFeature(wgpu::FeatureName::TimestampQuery);
  }
#endif
  return false;
}

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
inline void ExpectCooperativeDispatch(const profiling::Events& events) {
  int attention_dispatches = 0;
  for (const auto& event : events) {
    if (!std::string_view{event.name}.ends_with("&FlashAttention")) {
      continue;
    }
    ++attention_dispatches;
    const auto key = event.args.find("cache_key");
    ASSERT_NE(key, event.args.end());
    EXPECT_EQ(key->second.find("FlashAttention[SubgroupId]"), 0u) << key->second;
    // Cache keys include labels in Debug builds.
    EXPECT_TRUE(key->second.find(":256,1,1:128:") != std::string::npos ||
                key->second.find(":WorkgroupSize=256,1,1:SubgroupSize=128:") != std::string::npos)
        << key->second;
  }
  EXPECT_GT(attention_dispatches, 0) << "No cooperative attention dispatch was observed";
}
#endif

template <typename Run>
void RunWithCooperativeDispatchCheck(bool verify_dispatch, Run&& run) {
  if (!verify_dispatch) {
    run();
    return;
  }
  ASSERT_TRUE(SupportsCooperativeDispatchObservation());
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
  auto ep = DefaultWebGpuExecutionProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(ep->GetDeviceId());
  profiling::Events events;
  context.SetProfilingStartTime(std::chrono::high_resolution_clock::now());
  context.StartProfiling();
  {
    auto stop_profiling = gsl::finally([&] { context.CollectProfilingData(events); });
    run();
  }
  ExpectCooperativeDispatch(events);
#endif
}

}  // namespace onnxruntime::test
