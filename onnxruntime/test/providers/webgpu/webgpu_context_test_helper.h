// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "dawn/native/DawnNative.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

template <typename TestBody>
void RunWithExternalCompletionContext(TestBody test_body, bool timed_wait = true) {
  ConfigOptions options;
  auto owned_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(owned_ep, nullptr);

  const auto instance_feature = wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatureCount = timed_wait ? 1 : 0;
  instance_desc.requiredFeatures = timed_wait ? &instance_feature : nullptr;
  std::atomic<size_t> uncaptured_errors{0};
  dawn::native::Instance instance{&instance_desc};
  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = static_cast<wgpu::BackendType>(webgpu::WebGpuContextConfig{}.backend_type);
  auto adapters = instance.EnumerateAdapters(&adapter_options);
  ASSERT_FALSE(adapters.empty());

  const auto device_feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
  auto lost = std::make_shared<bool>(false);
  wgpu::DeviceDescriptor device_desc{};
  device_desc.requiredFeatureCount = 1;
  device_desc.requiredFeatures = &device_feature;
  device_desc.SetDeviceLostCallback(
      wgpu::CallbackMode::WaitAnyOnly,
      [lost](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView) noexcept { *lost = true; });
  device_desc.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView, std::atomic<size_t>* count) noexcept {
        count->fetch_add(1, std::memory_order_relaxed);
      },
      &uncaptured_errors);
  auto device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_desc));
  ASSERT_NE(device, nullptr);

  ConfigOptions external_options;
  ASSERT_STATUS_OK(external_options.AddConfigEntry(webgpu::options::kDeviceId, "1"));
  // Keep Dawn validation enabled, without session/kernel scopes masking the checked boundary.
  ASSERT_STATUS_OK(external_options.AddConfigEntry(webgpu::options::kValidationMode,
                                                   webgpu::options::kValidationMode_wgpuOnly));
  ASSERT_STATUS_OK(external_options.AddConfigEntry(
      webgpu::options::kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(instance.Get())).c_str()));
  ASSERT_STATUS_OK(external_options.AddConfigEntry(
      webgpu::options::kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));
  auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
  ASSERT_NE(external_ep, nullptr);
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*external_ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(1);
  test_body(context, webgpu_ep, lost, external_ep);
  // ForceLoss itself reports an uncaptured internal error before the checked call.
  if (!*lost) {
    EXPECT_EQ(uncaptured_errors.load(std::memory_order_relaxed), 0u);
  }
}

}  // namespace onnxruntime::test
