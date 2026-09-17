// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
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

void TestCopyAfterDeferredDispatch(bool upload) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& recording = static_cast<WebGpuExecutionProvider*>(ep.get())->Recording();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& buffer_manager = context.BufferManager();

  std::array<uint32_t, 16> input_data;
  input_data.fill(7);
  wgpu::BufferDescriptor buffer_desc{};
  buffer_desc.size = sizeof(input_data);
  buffer_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto input = context.Device().CreateBuffer(&buffer_desc);
  auto output = context.Device().CreateBuffer(&buffer_desc);
  auto copy = context.Device().CreateBuffer(&buffer_desc);
  buffer_manager.Upload(recording, input_data.data(), input.Get(), sizeof(input_data));

  wgpu::ShaderSourceWGSL source{};
  source.code = R"(
    @group(0) @binding(0) var<storage, read> input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> output: array<u32>;
    @compute @workgroup_size(16)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      output[id.x] = input[id.x] + 1u;
    }
  )";
  wgpu::ShaderModuleDescriptor shader_desc{};
  shader_desc.nextInChain = &source;
  wgpu::ComputePipelineDescriptor pipeline_desc{};
  pipeline_desc.compute.module = context.Device().CreateShaderModule(&shader_desc);
  pipeline_desc.compute.entryPoint = "main";
  auto pipeline = context.Device().CreateComputePipeline(&pipeline_desc);
  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].binding = 0;
  entries[0].buffer = input;
  entries[0].size = sizeof(input_data);
  entries[1].binding = 1;
  entries[1].buffer = output;
  entries[1].size = sizeof(input_data);
  wgpu::BindGroupDescriptor bind_group_desc{};
  bind_group_desc.layout = pipeline.GetBindGroupLayout(0);
  bind_group_desc.entryCount = entries.size();
  bind_group_desc.entries = entries.data();
  webgpu::CapturedCommandInfo dispatch;
  dispatch.compute_pipeline = pipeline;
  dispatch.bind_group = context.Device().CreateBindGroup(&bind_group_desc);
  recording.deferred_dispatches.push_back(std::move(dispatch));
  recording.has_unsubmitted_work = true;

  if (upload) {
    // The dispatch must consume the original input before Upload overwrites it.
    input_data.fill(42);
    buffer_manager.Upload(recording, input_data.data(), input.Get(), sizeof(input_data));
  } else {
    // MemCpy must read the dispatch result, not the output buffer's initial zeros.
    buffer_manager.MemCpy(recording, output.Get(), copy.Get(), sizeof(input_data));
  }
  EXPECT_TRUE(recording.deferred_dispatches.empty());

  std::array<uint32_t, 16> result{};
  buffer_manager.Download(recording, upload ? output.Get() : copy.Get(), result.data(), sizeof(result));
  std::array<uint32_t, 16> expected;
  expected.fill(8);
  EXPECT_EQ(result, expected);
}

TEST(WebGpuContextTest, UploadFollowsDeferredDispatch) {
  TestCopyAfterDeferredDispatch(true);
}

TEST(WebGpuContextTest, MemCpyFollowsDeferredDispatch) {
  TestCopyAfterDeferredDispatch(false);
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
      false);

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
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), expected_data);
  const Status flush_status = context.Flush(buffer_manager, webgpu_ep->Recording());
  context.CaptureEnd(webgpu_ep->Recording());
  if (!flush_status.IsOK()) {
    allocator.Free(allocation);
    FAIL() << flush_status.ErrorMessage();
  }

  EXPECT_TRUE(captured_commands.empty());

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, SessionAllocatorReusesZeroedBufferDuringRun) {
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
      false);

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
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), expected_data);
  EXPECT_FALSE(webgpu_ep->Recording().has_unsubmitted_work);

  ASSERT_STATUS_OK(webgpu_ep->OnRunEnd(false, run_options));
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer), expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, ClearsRetiredStorageAfterLastUse) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);

  for (auto mode : {webgpu::BufferCacheMode::Bucket, webgpu::BufferCacheMode::Simple}) {
    SCOPED_TRACE(mode);
    webgpu::BufferManager buffer_manager(context, mode,
                                         webgpu::BufferCacheMode::Disabled,
                                         webgpu::BufferCacheMode::Disabled,
                                         webgpu::BufferCacheMode::Disabled);
    webgpu::CommandRecordingState recording;
    std::array<uint32_t, 16> nonzero_data;
    nonzero_data.fill(7);
    constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    auto buffer = wgpu::Buffer::Acquire(buffer_manager.Create(recording, sizeof(nonzero_data), usage));
    WGPUBuffer retired_buffer = buffer.Get();
    buffer_manager.Upload(recording, nonzero_data.data(), buffer.Get(), sizeof(nonzero_data));

    wgpu::BufferDescriptor desc{};
    desc.size = sizeof(nonzero_data);
    desc.usage = usage;
    auto output = context.Device().CreateBuffer(&desc);
    buffer_manager.MemCpy(recording, buffer.Get(), output.Get(), sizeof(nonzero_data));
    buffer_manager.Release(recording, buffer.MoveToCHandle());

    auto fresh_buffer = wgpu::Buffer::Acquire(buffer_manager.Create(recording, sizeof(nonzero_data), usage));
    EXPECT_NE(fresh_buffer.Get(), retired_buffer);
    EXPECT_TRUE(recording.has_unsubmitted_work);
    EXPECT_EQ(recording.pending_buffers.size(), 1u);

    ASSERT_STATUS_OK(context.Flush(buffer_manager, recording));
    EXPECT_TRUE(recording.pending_buffers.empty());
    EXPECT_FALSE(recording.has_unsubmitted_work);
    EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, output.Get()), nonzero_data);

    auto reused_buffer = wgpu::Buffer::Acquire(buffer_manager.Create(recording, sizeof(nonzero_data), usage));
    EXPECT_EQ(reused_buffer.Get(), retired_buffer);
    const std::array<uint32_t, 16> expected_data{};
    EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer.Get()), expected_data);
    EXPECT_FALSE(recording.has_unsubmitted_work);
  }
}

TEST(WebGpuContextTest, FailedFlushDoesNotRecycleUnclearedBuffers) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Simple,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(7);
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = wgpu::Buffer::Acquire(buffer_manager.Create(recording, sizeof(nonzero_data), usage));
  WGPUBuffer retired_buffer = buffer.Get();
  buffer_manager.Upload(recording, nonzero_data.data(), buffer.Get(), sizeof(nonzero_data));
  buffer_manager.Release(recording, buffer.MoveToCHandle());

  webgpu::CapturedCommandInfo invalid_dispatch;
  invalid_dispatch.program_key = "missing-pipeline-for-recycle-test";
  recording.deferred_dispatches.push_back(std::move(invalid_dispatch));
  recording.has_unsubmitted_work = true;
  const auto status = context.Flush(buffer_manager, recording);
  ASSERT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("No cached or pending pipeline"), std::string::npos);
  EXPECT_EQ(recording.pending_buffers.size(), 1u);
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, retired_buffer), nonzero_data);

  ASSERT_STATUS_OK(context.Flush(buffer_manager, recording));
  EXPECT_TRUE(recording.pending_buffers.empty());
  auto reused_buffer = wgpu::Buffer::Acquire(buffer_manager.Create(recording, sizeof(nonzero_data), usage));
  EXPECT_EQ(reused_buffer.Get(), retired_buffer);
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, reused_buffer.Get()), expected_data);
}

TEST(WebGpuContextTest, EnablesImplicitDeviceSynchronization) {
#if defined(__wasm__)
  GTEST_SKIP() << "ImplicitDeviceSynchronization is a Dawn native feature.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  EXPECT_TRUE(webgpu::WebGpuContextFactory::GetContext(0).DeviceHasFeature(
      wgpu::FeatureName::ImplicitDeviceSynchronization));
#endif
}

TEST(WebGpuContextTest, ExternalDeviceRequiresImplicitDeviceSynchronization) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  // Initialize ORT's Dawn proc table before using externally created devices.
  ConfigOptions options;
  auto owned_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(owned_ep, nullptr);

  dawn::native::Instance instance;
  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = static_cast<wgpu::BackendType>(webgpu::WebGpuContextConfig{}.backend_type);
  auto adapters = instance.EnumerateAdapters(&adapter_options);
  ASSERT_FALSE(adapters.empty());

  for (bool enable_synchronization : {false, true}) {
    SCOPED_TRACE(enable_synchronization);
    const auto feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
    wgpu::DeviceDescriptor device_desc{};
    device_desc.requiredFeatureCount = enable_synchronization ? 1 : 0;
    device_desc.requiredFeatures = enable_synchronization ? &feature : nullptr;
    auto device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_desc));
    ASSERT_NE(device, nullptr);
    ASSERT_EQ(device.HasFeature(feature), enable_synchronization);

    ConfigOptions external_options;
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(kDeviceId, "1"));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(instance.Get())).c_str()));
    ORT_THROW_IF_ERROR(external_options.AddConfigEntry(
        kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));

    if (enable_synchronization) {
      auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
      ASSERT_NE(external_ep, nullptr);
      EXPECT_TRUE(webgpu::WebGpuContextFactory::GetContext(1).DeviceHasFeature(feature));
    } else {
      EXPECT_THAT([&]() { WebGpuProviderFactoryCreator::Create(external_options); },
                  ::testing::ThrowsMessage<OnnxRuntimeException>(::testing::HasSubstr(
                      "an externally supplied native device must enable ImplicitDeviceSynchronization "
                      "in DeviceDescriptor.requiredFeatures when it is created.")));
    }
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
