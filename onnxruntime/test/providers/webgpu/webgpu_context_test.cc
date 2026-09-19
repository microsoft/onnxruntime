// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/run_options.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/asserts.h"

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/DawnNative.h"
#endif

namespace onnxruntime {
namespace test {
namespace {

using namespace webgpu::options;

ConfigOptions RobustnessOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kEnableRobustness, value));
  return options;
}

ConfigOptions KvCacheQuantizationOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kKvCacheQuantizationBits, value));
  return options;
}

bool DeviceToggleIsEnabled(const webgpu::WebGpuContext& context, std::string_view toggle_name) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  const auto toggles = dawn::native::GetTogglesUsed(context.Device().Get());
  return std::any_of(toggles.begin(), toggles.end(), [toggle_name](const char* toggle) {
    return std::string_view{toggle} == toggle_name;
  });
#else
  ORT_UNUSED_PARAMETER(context);
  ORT_UNUSED_PARAMETER(toggle_name);
  return false;
#endif
}

bool DisableRobustnessToggleIsEnabled(const webgpu::WebGpuContext& context) {
  return DeviceToggleIsEnabled(context, "disable_robustness");
}

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
wgpu::Device CreateExternalDevice(webgpu::WebGpuContext& context, bool enable_synchronization) {
  auto adapter = std::make_shared<wgpu::Adapter>();
  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = context.AdapterInfo().backendType;
  ORT_THROW_IF_ERROR(context.Wait(context.Instance().RequestAdapter(
      &adapter_options, wgpu::CallbackMode::WaitAnyOnly,
      [adapter](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView) {
        if (status == wgpu::RequestAdapterStatus::Success) {
          *adapter = std::move(result);
        }
      })));
  ORT_ENFORCE(*adapter, "Failed to request the external device's adapter.");

  wgpu::DawnTogglesDescriptor toggles{};
  const char* disabled_toggles[] = {"skip_validation"};
  toggles.disabledToggles = disabled_toggles;
  toggles.disabledToggleCount = std::size(disabled_toggles);
  wgpu::FeatureName synchronization = wgpu::FeatureName::ImplicitDeviceSynchronization;
  wgpu::DeviceDescriptor descriptor{};
  descriptor.nextInChain = &toggles;
  descriptor.requiredFeatureCount = enable_synchronization ? 1 : 0;
  descriptor.requiredFeatures = enable_synchronization ? &synchronization : nullptr;
  auto device = std::make_shared<wgpu::Device>();
  ORT_THROW_IF_ERROR(context.Wait(adapter->RequestDevice(
      &descriptor, wgpu::CallbackMode::WaitAnyOnly,
      [device](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView) {
        if (status == wgpu::RequestDeviceStatus::Success) {
          *device = std::move(result);
        }
      })));
  ORT_ENFORCE(*device, "Failed to create an external device.");
  return std::move(*device);
}

ConfigOptions ExternalDeviceOptions(const webgpu::WebGpuContext& context, const wgpu::Device& device,
                                    const char* context_id) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kDeviceId, context_id));
  ORT_THROW_IF_ERROR(options.AddConfigEntry(
      kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(context.Instance().Get())).c_str()));
  ORT_THROW_IF_ERROR(options.AddConfigEntry(
      kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kValidationMode, kValidationMode_basic));
  return options;
}
#endif

std::array<uint32_t, 16> ReadBufferWithExternalCommandEncoder(webgpu::WebGpuContext& context,
                                                              WGPUBuffer buffer) {
  constexpr size_t kBufferSize = sizeof(std::array<uint32_t, 16>);
  wgpu::BufferDescriptor readback_desc{};
  readback_desc.size = kBufferSize;
  readback_desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  auto readback_buffer = context.Device().CreateBuffer(&readback_desc);

  auto external_encoder = context.Device().CreateCommandEncoder();
  external_encoder.CopyBufferToBuffer(buffer, 0, readback_buffer, 0, kBufferSize);
  auto external_commands = external_encoder.Finish();
  context.Device().GetQueue().Submit(1, &external_commands);

  wgpu::MapAsyncStatus map_status{};
  ORT_THROW_IF_ERROR(context.Wait(readback_buffer.MapAsync(
      wgpu::MapMode::Read, 0, kBufferSize, wgpu::CallbackMode::WaitAnyOnly,
      [](wgpu::MapAsyncStatus status, wgpu::StringView /*message*/, wgpu::MapAsyncStatus* result) noexcept {
        *result = status;
      },
      &map_status)));
  ORT_ENFORCE(map_status == wgpu::MapAsyncStatus::Success);

  std::array<uint32_t, 16> result;
  const auto* mapped_data = static_cast<const uint32_t*>(readback_buffer.GetConstMappedRange());
  ORT_ENFORCE(mapped_data != nullptr);
  std::copy_n(mapped_data, result.size(), result.begin());
  readback_buffer.Unmap();
  return result;
}

TEST(WebGpuContextTest, SessionAllocatorSubmitsReusedBufferClearOutsideRun) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Bucket,
                                       webgpu::BufferCacheMode::Simple,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  webgpu::GpuBufferAllocator allocator(
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);

  const auto downloaded_data = ReadBufferWithExternalCommandEncoder(context, reused_buffer);
  ASSERT_STATUS_OK(context.Flush(buffer_manager, webgpu_ep->Recording()));
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(downloaded_data, expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, DoesNotCaptureDeviceAllocatorBufferClear) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Graph,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  std::vector<webgpu::CapturedCommandInfo> captured_commands;
  webgpu::GpuBufferAllocator allocator(
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false);

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  context.CaptureBegin(&captured_commands, buffer_manager, webgpu_ep->Recording());
  allocation = allocator.Alloc(sizeof(nonzero_data));
  if (allocation == nullptr) {
    context.CaptureEnd(webgpu_ep->Recording());
    FAIL() << "Failed to reacquire a device allocation during graph capture.";
  }
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);
  const Status flush_status = context.Flush(buffer_manager, webgpu_ep->Recording());
  context.CaptureEnd(webgpu_ep->Recording());
  if (!flush_status.IsOK()) {
    allocator.Free(allocation);
    FAIL() << flush_status.ErrorMessage();
  }

  EXPECT_TRUE(captured_commands.empty());

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, SessionAllocatorDefersReusedBufferClearDuringRun) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Bucket,
                                       webgpu::BufferCacheMode::Simple,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  webgpu::GpuBufferAllocator allocator(
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      [webgpu_ep]() -> webgpu::CommandRecordingState& { return webgpu_ep->Recording(); },
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(webgpu_ep->Recording(), nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  ASSERT_STATUS_OK(context.Flush(buffer_manager, webgpu_ep->Recording()));
  allocator.Free(allocation);

  RunOptions run_options;
  ASSERT_STATUS_OK(webgpu_ep->OnRunStart(run_options));
  allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), nonzero_data);

  ASSERT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, WebGpuExecutionProviderTracksRunActivity) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(ep.get());
  RunOptions run_options;

  EXPECT_FALSE(webgpu_ep->IsRunActive());
  ASSERT_STATUS_OK(webgpu_ep->OnRunStart(run_options));
  EXPECT_TRUE(webgpu_ep->IsRunActive());
  ASSERT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
  EXPECT_FALSE(webgpu_ep->IsRunActive());
}

TEST(WebGpuContextTest, ConcurrentValidationScopesAttributeErrorsToTheirCallingThreads) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "This test exercises the pinned Dawn native error-scope implementation.";
#else
  ConfigOptions default_options;
  auto default_ep = WebGpuProviderFactoryCreator::Create(default_options)->CreateProvider();
  ASSERT_NE(default_ep, nullptr);
  auto& default_context = webgpu::WebGpuContextFactory::GetContext(0);

  // A new device is necessary: wrapping the default Release device cannot undo skip_validation.
  auto device = CreateExternalDevice(default_context, true);
  auto options = ExternalDeviceOptions(default_context, device, "29851");
  auto first_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto second_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(first_ep, nullptr);
  ASSERT_NE(second_ep, nullptr);

  auto* first_webgpu_ep = static_cast<WebGpuExecutionProvider*>(first_ep.get());
  auto* second_webgpu_ep = static_cast<WebGpuExecutionProvider*>(second_ep.get());
  auto& context = webgpu::WebGpuContextFactory::GetContext(29851);
  ASSERT_FALSE(DeviceToggleIsEnabled(context, "skip_validation"));
  RunOptions run_options;

  auto inject_validation_error = [&] {
    wgpu::BufferDescriptor descriptor{};
    descriptor.size = 16;
    descriptor.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::Storage;
    auto invalid_buffer = device.CreateBuffer(&descriptor);
  };

  for (bool error_in_first : {true, false}) {
    ASSERT_STATUS_OK(first_webgpu_ep->OnRunStart(run_options));
    if (error_in_first) {
      inject_validation_error();
    }

    std::promise<void> second_started;
    auto second_started_future = second_started.get_future();
    std::promise<void> first_ended;
    auto first_ended_future = first_ended.get_future();
    Status second_start_status;
    Status second_end_status;
    Status second_recovery_status;
    Status second_empty_scope_status;
    std::thread second_run([&]() {
      second_start_status = second_webgpu_ep->OnRunStart(run_options);
      if (second_start_status.IsOK() && !error_in_first) {
        inject_validation_error();
      }
      second_started.set_value();
      first_ended_future.wait();
      if (second_start_status.IsOK()) {
        second_end_status = second_webgpu_ep->OnRunEnd(false, run_options);
      }
      second_recovery_status = second_webgpu_ep->OnRunStart(run_options);
      if (second_recovery_status.IsOK()) {
        second_recovery_status = second_webgpu_ep->OnRunEnd(false, run_options);
      }
      second_empty_scope_status = context.PopErrorScope();
    });

    // Force A.push -> B.push -> A.pop -> B.pop, retaining an error in exactly one scope.
    // Timing out still lets A end, so a reintroduced Run-wide lock fails instead of deadlocking.
    const auto overlap = second_started_future.wait_for(std::chrono::seconds{10});
    Status first_status = first_webgpu_ep->OnRunEnd(false, run_options);
    first_ended.set_value();
    second_run.join();

    EXPECT_EQ(overlap, std::future_status::ready);
    EXPECT_STATUS_OK(second_start_status);
    EXPECT_EQ(first_status.IsOK(), !error_in_first) << first_status;
    EXPECT_EQ(second_end_status.IsOK(), error_in_first) << second_end_status;
    const auto& error = error_in_first ? first_status : second_end_status;
    EXPECT_NE(error.ErrorMessage().find("WebGPU validation failed"), std::string::npos);
    EXPECT_STATUS_OK(second_recovery_status);
    EXPECT_FALSE(second_empty_scope_status.IsOK());
    ASSERT_STATUS_OK(first_webgpu_ep->OnRunStart(run_options));
    EXPECT_STATUS_OK(first_webgpu_ep->OnRunEnd(false, run_options));
    EXPECT_FALSE(context.PopErrorScope().IsOK());
  }
#endif
}

TEST(WebGpuContextTest, EnablesLazyClearResourceOnFirstUse) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);

  EXPECT_TRUE(DeviceToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0),
                                    "lazy_clear_resource_on_first_use"));
#endif
}

TEST(WebGpuContextTest, NativeDeviceUsesImplicitSynchronization) {
#if defined(__wasm__)
  GTEST_SKIP() << "ImplicitDeviceSynchronization is a native Dawn feature.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  EXPECT_TRUE(webgpu::WebGpuContextFactory::GetContext(0).DeviceHasFeature(
      wgpu::FeatureName::ImplicitDeviceSynchronization));
#endif
}

TEST(WebGpuContextTest, ExternalDeviceMustEnableImplicitSynchronizationAtCreation) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN) || defined(ORT_NO_EXCEPTIONS)
  GTEST_SKIP() << "This test requires Dawn native and exceptions.";
#else
  ConfigOptions default_options;
  auto default_ep = WebGpuProviderFactoryCreator::Create(default_options)->CreateProvider();
  ASSERT_NE(default_ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto unsynchronized_device = CreateExternalDevice(context, false);
  ASSERT_FALSE(unsynchronized_device.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization));
  auto invalid_options = ExternalDeviceOptions(context, unsynchronized_device, "29853");
  try {
    auto invalid_ep = WebGpuProviderFactoryCreator::Create(invalid_options)->CreateProvider();
    FAIL() << "An unsynchronized external device was accepted.";
  } catch (const OnnxRuntimeException& e) {
    EXPECT_NE(std::string_view{e.what()}.find("DeviceDescriptor.requiredFeatures"), std::string_view::npos);
  }

  auto synchronized_device = CreateExternalDevice(context, true);
  auto valid_options = ExternalDeviceOptions(context, synchronized_device, "29853");
  auto valid_ep = WebGpuProviderFactoryCreator::Create(valid_options)->CreateProvider();
  ASSERT_NE(valid_ep, nullptr);
  auto* webgpu_ep = static_cast<WebGpuExecutionProvider*>(valid_ep.get());
  RunOptions run_options;
  ASSERT_STATUS_OK(webgpu_ep->OnRunStart(run_options));
  EXPECT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
#endif
}

TEST(WebGpuContextTest, EnableRobustnessControlsDawnToggle) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  auto enabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
  ASSERT_NE(enabled_ep, nullptr);
  EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
  enabled_ep.reset();

  auto disabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
  ASSERT_NE(disabled_ep, nullptr);
  EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#endif
}

TEST(WebGpuContextTest, EnableRobustnessUsesBuildDefault) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
#ifdef NDEBUG
  EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#else
  EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#endif
#endif
}

TEST(WebGpuContextTest, EnableRobustnessRejectsInvalidValue) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(RobustnessOptions("true")), OnnxRuntimeException);
}

TEST(WebGpuContextTest, KvCacheQuantizationAcceptsSupportedBitWidths) {
  for (const auto& [value, expected_bits] :
       std::array<std::pair<const char*, uint32_t>, 3>{{{"0", 0}, {"4", 4}, {"8", 8}}}) {
    auto ep = WebGpuProviderFactoryCreator::Create(KvCacheQuantizationOptions(value))->CreateProvider();
    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(static_cast<WebGpuExecutionProvider*>(ep.get())->KvCacheQuantizationBits(), expected_bits);
  }
}

TEST(WebGpuContextTest, KvCacheQuantizationRejectsInvalidValue) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(KvCacheQuantizationOptions("3")), OnnxRuntimeException);
}

TEST(WebGpuContextTest, CompileOnlyContextDoesNotCreateDevice) {
  auto options = RobustnessOptions("0");
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

  ASSERT_NE(ep, nullptr);
  EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
}

TEST(WebGpuContextTest, EnableRobustnessIsIndependentFromValidationMode) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  auto robust_options = RobustnessOptions("1");
  ORT_THROW_IF_ERROR(robust_options.AddConfigEntry(kValidationMode, kValidationMode_Disabled));
  auto robust_ep = WebGpuProviderFactoryCreator::Create(robust_options)->CreateProvider();
  ASSERT_NE(robust_ep, nullptr);
  const auto& robust_context = webgpu::WebGpuContextFactory::GetContext(0);
  EXPECT_FALSE(DisableRobustnessToggleIsEnabled(robust_context));
  EXPECT_TRUE(DeviceToggleIsEnabled(robust_context, "skip_validation"));
  robust_ep.reset();

  auto non_robust_options = RobustnessOptions("0");
  ORT_THROW_IF_ERROR(non_robust_options.AddConfigEntry(kValidationMode, kValidationMode_full));
  auto non_robust_ep = WebGpuProviderFactoryCreator::Create(non_robust_options)->CreateProvider();
  ASSERT_NE(non_robust_ep, nullptr);
  const auto& non_robust_context = webgpu::WebGpuContextFactory::GetContext(0);
  EXPECT_TRUE(DisableRobustnessToggleIsEnabled(non_robust_context));
  EXPECT_FALSE(DeviceToggleIsEnabled(non_robust_context, "skip_validation"));
#endif
}

TEST(WebGpuContextTest, ConflictingExplicitValueWarnsAndKeepsFirstValue) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  auto first_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
  ASSERT_NE(first_ep, nullptr);

  testing::internal::CaptureStderr();
  auto second_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
  const std::string warning = testing::internal::GetCapturedStderr();

  ASSERT_NE(second_ep, nullptr);
  EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
  EXPECT_NE(warning.find("already initialized"), std::string::npos);
  EXPECT_NE(warning.find("will be ignored"), std::string::npos);
#endif
}

TEST(WebGpuContextTest, OmittedAndMatchingValuesDoNotWarn) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  auto first_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
  ASSERT_NE(first_ep, nullptr);

  ConfigOptions omitted_options;
  testing::internal::CaptureStderr();
  auto omitted_ep = WebGpuProviderFactoryCreator::Create(omitted_options)->CreateProvider();
  auto matching_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
  const std::string warning = testing::internal::GetCapturedStderr();

  ASSERT_NE(omitted_ep, nullptr);
  ASSERT_NE(matching_ep, nullptr);
  EXPECT_EQ(warning.find("enableRobustness"), std::string::npos);
#endif
}

TEST(WebGpuContextTest, ExternalDeviceValueWarnsAndIsIgnored) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  auto owned_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
  ASSERT_NE(owned_ep, nullptr);
  const auto& owned_context = webgpu::WebGpuContextFactory::GetContext(0);

  ConfigOptions external_options;
  ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kDeviceId, "1"));
  ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
      kWebGpuInstance,
      std::to_string(reinterpret_cast<uintptr_t>(owned_context.Instance().Get())).c_str()));
  ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
      kWebGpuDevice,
      std::to_string(reinterpret_cast<uintptr_t>(owned_context.Device().Get())).c_str()));
  ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kEnableRobustness, "1"));

  testing::internal::CaptureStderr();
  auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
  const std::string warning = testing::internal::GetCapturedStderr();

  ASSERT_NE(external_ep, nullptr);
  EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(1)));
  EXPECT_NE(warning.find("externally supplied WebGPU device"), std::string::npos);
  EXPECT_NE(warning.find("will be ignored"), std::string::npos);
#endif
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
