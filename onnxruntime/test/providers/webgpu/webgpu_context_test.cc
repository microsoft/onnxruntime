// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
#include "absl/flags/reflection.h"
#endif

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

template <typename TestBody>
void RunWithFreshDefaultContext(TestBody test_body, bool compile_only_parent = false) {
#if GTEST_HAS_DEATH_TEST
  // Context 0 can outlive an individual test. Re-exec instead of forking its
  // initialized Dawn device or clearing state that another EP still owns.
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  auto* death_test_style_flag = absl::FindCommandLineFlag("gtest_death_test_style");
  ASSERT_NE(death_test_style_flag, nullptr);
  const std::string previous_style = death_test_style_flag->CurrentValue();
  std::string flag_error;
  ASSERT_TRUE(death_test_style_flag->ParseFrom("threadsafe", &flag_error)) << flag_error;
#else
  const auto previous_style = GTEST_FLAG_GET(death_test_style);
  GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif

  // Exercise isolation even when running only this test, and ensure the child
  // does not disturb a live provider in the parent process.
  std::unique_ptr<IExecutionProvider> existing_provider;
  WGPUDevice existing_device = nullptr;
  if (!testing::internal::InDeathTestChild()) {
    ConfigOptions options;
    if (compile_only_parent) {
      // Keep the compile-only test runnable on hosts without a GPU. An existing
      // device context, if any, will still be retained by this provider.
      ASSERT_STATUS_OK(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));
    }
    existing_provider = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(existing_provider, nullptr);
    existing_device = webgpu::WebGpuContextFactory::GetContext(0).Device().Get();
  }

  EXPECT_EXIT(
      {
        test_body();
        std::_Exit(testing::Test::HasFailure() || testing::Test::IsSkipped() ? EXIT_FAILURE : EXIT_SUCCESS);
      },
      testing::ExitedWithCode(EXIT_SUCCESS), "");

  EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), existing_device);
#if defined(GTEST_HAS_ABSL) && !defined(GTEST_NO_ABSL_FLAGS)
  flag_error.clear();
  EXPECT_TRUE(death_test_style_flag->ParseFrom(previous_style, &flag_error)) << flag_error;
#else
  GTEST_FLAG_SET(death_test_style, previous_style);
#endif
#else
  ORT_UNUSED_PARAMETER(compile_only_parent);
  test_body();
#endif
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

void TestCopyAfterDeferredDispatch(bool upload, bool checked_completion = false) {
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
  recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);

  if (checked_completion) {
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(buffer_manager, recording));
  } else if (upload) {
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

void ExpectQueueCompleted(webgpu::WebGpuContext& context) {
  auto result = std::make_shared<wgpu::QueueWorkDoneStatus>(wgpu::QueueWorkDoneStatus::CallbackCancelled);
  const auto future = context.Device().GetQueue().OnSubmittedWorkDone(
      wgpu::CallbackMode::WaitAnyOnly,
      [result](wgpu::QueueWorkDoneStatus status, wgpu::StringView /*message*/) noexcept {
        *result = status;
      });
  EXPECT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
  EXPECT_EQ(*result, wgpu::QueueWorkDoneStatus::Success);
}

TEST(WebGpuContextTest, CheckedCompletionHandlesEmptyAndPendingEncoders) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();

  for (bool pending_encoder : {false, true}) {
    SCOPED_TRACE(pending_encoder);
    if (pending_encoder) {
      context.GetCommandEncoder(recording);
      ASSERT_TRUE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
    }
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
    EXPECT_EQ(recording.command_encoder, nullptr);
    EXPECT_FALSE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
    ExpectQueueCompleted(context);
  }
}

TEST(WebGpuContextTest, CheckedCompletionSubmitsAndWaitsForCopies) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
  std::array<uint32_t, 16> expected;
  expected.fill(42);
  wgpu::BufferDescriptor desc{};
  desc.size = sizeof(expected);
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto source = context.Device().CreateBuffer(&desc);
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  auto destination = context.Device().CreateBuffer(&desc);
  context.Device().GetQueue().WriteBuffer(source, 0, expected.data(), sizeof(expected));

  for (bool already_submitted : {false, true}) {
    SCOPED_TRACE(already_submitted);
    context.GetCommandEncoder(recording).CopyBufferToBuffer(source, 0, destination, 0, sizeof(expected));
    if (already_submitted) {
      ASSERT_STATUS_OK(context.Flush(webgpu_ep.BufferManager(), recording));
    }
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
    ExpectQueueCompleted(context);

    auto map_status = std::make_shared<wgpu::MapAsyncStatus>(wgpu::MapAsyncStatus::Error);
    ASSERT_STATUS_OK(context.Wait(destination.MapAsync(
        wgpu::MapMode::Read, 0, sizeof(expected), wgpu::CallbackMode::WaitAnyOnly,
        [map_status](wgpu::MapAsyncStatus status, wgpu::StringView /*message*/) noexcept {
          *map_status = status;
        })));
    ASSERT_EQ(*map_status, wgpu::MapAsyncStatus::Success);
    std::array<uint32_t, 16> actual;
    std::copy_n(static_cast<const uint32_t*>(destination.GetConstMappedRange()), actual.size(), actual.begin());
    EXPECT_EQ(actual, expected);
    destination.Unmap();
  }
}

TEST(WebGpuContextTest, CheckedCompletionEncodesDeferredDispatch) {
  TestCopyAfterDeferredDispatch(true, true);
}

TEST(WebGpuContextTest, CheckedCompletionDrainsAfterDeferredDispatchFailure) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
  webgpu::CapturedCommandInfo invalid_dispatch;
  invalid_dispatch.program_key = "checked-completion-missing-pipeline";
  recording.deferred_dispatches.push_back(std::move(invalid_dispatch));
  recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);

  const auto status = context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording);
  EXPECT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("No cached or pending pipeline"));
  EXPECT_TRUE(recording.deferred_dispatches.empty());
  EXPECT_FALSE(recording.has_unsubmitted_work.load(std::memory_order_relaxed));
  ExpectQueueCompleted(context);
  ASSERT_STATUS_OK(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), recording));
}

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
class PipelineCallbackTestProgram final : public webgpu::Program<PipelineCallbackTestProgram> {
 public:
  PipelineCallbackTestProgram() : Program{"PipelineCallbackTest"} {}

  Status GenerateShaderCode(webgpu::ShaderHelper&) const override {
    return Status::OK();
  }
};

Status StartPipelineCallbackTestBuild(webgpu::WebGpuContext& context, webgpu::PendingPipelineBuild& build,
                                      bool invalid_shader) {
  PipelineCallbackTestProgram program;
  program.SetWorkgroupSize(1);
  auto metadata = program.Metadata();
  const webgpu::ProgramConstant invalid_constant{"invalid identifier", uint32_t{0}};
  if (invalid_shader) {
    metadata.constants = {&invalid_constant, 1};
  }
  webgpu::ProgramManager manager{context};
  build.name = program.Name();
  build.callback_context = std::make_shared<webgpu::PipelineCallbackContext>();
  return manager.Build(program, metadata, {}, {}, program.Name(), 1, 1, 1,
                       build.bind_group_layout, build.shape_uniform_ranks, build.future, build.callback_context);
}

template <typename TestBody>
void RunWithExternalCompletionContext(TestBody test_body, bool timed_wait = true) {
  ConfigOptions options;
  auto owned_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(owned_ep, nullptr);

  const auto instance_feature = wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatureCount = timed_wait ? 1 : 0;
  instance_desc.requiredFeatures = timed_wait ? &instance_feature : nullptr;
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
  auto device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_desc));
  ASSERT_NE(device, nullptr);

  ConfigOptions external_options;
  ASSERT_STATUS_OK(external_options.AddConfigEntry(kDeviceId, "1"));
  ASSERT_STATUS_OK(external_options.AddConfigEntry(
      kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(instance.Get())).c_str()));
  ASSERT_STATUS_OK(external_options.AddConfigEntry(
      kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));
  auto external_ep = WebGpuProviderFactoryCreator::Create(external_options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*external_ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(1);
  test_body(context, webgpu_ep, lost);
}
#endif

TEST(WebGpuContextTest, SuccessfulPipelineBuildReleasesCallbackOwnership) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto&, const auto&) {
    webgpu::PendingPipelineBuild build;
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, false));
    const std::weak_ptr<webgpu::PipelineCallbackContext> weak_context = build.callback_context;
    ASSERT_STATUS_OK(context.Wait(build.future));
    ASSERT_STATUS_OK(build.callback_context->status);
    EXPECT_NE(build.callback_context->pipeline, nullptr);
    EXPECT_EQ(weak_context.use_count(), 1);
    build.callback_context.reset();
    EXPECT_TRUE(weak_context.expired());
  });
#endif
}

TEST(WebGpuContextTest, FailedPipelineWaitRetainsCallbackUntilDelivery) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&) {
    // Invalid WGSL makes the pipeline event ready synchronously, but WaitAnyOnly defers delivery.
    context.PushErrorScope();
    webgpu::CapturedCommandInfo dispatch;
    dispatch.program_key = "failed-wait-pipeline-callback";
    auto& build = dispatch.pending_build.emplace();
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, true));
    const auto future = build.future;
    const std::weak_ptr<webgpu::PipelineCallbackContext> weak_context = build.callback_context;
    auto& recording = ep.Recording();
    recording.deferred_dispatches.push_back(std::move(dispatch));

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(context.Flush(ep.BufferManager(), recording),
                                        "Failed to wait for the operation");
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    ASSERT_EQ(weak_context.use_count(), 1);

    auto callback_context = weak_context.lock();
    ASSERT_NE(callback_context, nullptr);
    ASSERT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
    EXPECT_FALSE(callback_context->status.IsOK());
    EXPECT_THAT(callback_context->status.ErrorMessage(),
                testing::HasSubstr("Failed to create a WebGPU compute pipeline"));
    EXPECT_EQ(callback_context->pipeline, nullptr);
    EXPECT_EQ(weak_context.use_count(), 1);
    callback_context.reset();
    EXPECT_TRUE(weak_context.expired());

    auto scope_error = std::make_shared<wgpu::ErrorType>(wgpu::ErrorType::NoError);
    const auto scope_future = context.Device().PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [scope_error](wgpu::PopErrorScopeStatus, wgpu::ErrorType error, wgpu::StringView) noexcept {
          *scope_error = error;
        });
    ASSERT_EQ(context.Instance().WaitAny(scope_future, 0), wgpu::WaitStatus::Success);
    EXPECT_EQ(*scope_error, wgpu::ErrorType::Validation);
  },
                                   false);
#endif
}

TEST(WebGpuContextTest, FailedPipelineWaitRetainsCallbackAfterContextDestruction) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  wgpu::Instance callback_instance;
  wgpu::Future future{};
  std::weak_ptr<webgpu::PipelineCallbackContext> weak_context;
  RunWithExternalCompletionContext([&](auto& context, auto& ep, const auto&) {
    context.PushErrorScope();
    webgpu::CapturedCommandInfo dispatch;
    dispatch.program_key = "post-context-pipeline-callback";
    auto& build = dispatch.pending_build.emplace();
    ASSERT_STATUS_OK(StartPipelineCallbackTestBuild(context, build, true));
    callback_instance = context.Instance();
    future = build.future;
    weak_context = build.callback_context;
    auto& recording = ep.Recording();
    recording.deferred_dispatches.push_back(std::move(dispatch));

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(context.Flush(ep.BufferManager(), recording),
                                        "Failed to wait for the operation");
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    EXPECT_EQ(weak_context.use_count(), 1);
    // Leave the callback undelivered until after the ORT context and manager are destroyed.
  },
                                   false);
  EXPECT_THROW(webgpu::WebGpuContextFactory::GetContext(1), OnnxRuntimeException);
  ASSERT_EQ(weak_context.use_count(), 1);
  auto callback_context = weak_context.lock();
  ASSERT_NE(callback_context, nullptr);
  EXPECT_TRUE(callback_context->status.IsOK());
  ASSERT_EQ(callback_instance.WaitAny(future, 0), wgpu::WaitStatus::Success);
  EXPECT_EQ(callback_context.use_count(), 1);
  callback_context.reset();
  EXPECT_TRUE(weak_context.expired());
#endif
}

TEST(WebGpuContextTest, CheckedCompletionPropagatesValidationAndPreservesOuterScope) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&) {
    auto& recording = ep.Recording();
    wgpu::BufferDescriptor desc{};
    desc.size = 64;
    desc.usage = wgpu::BufferUsage::CopySrc;
    auto buffer = context.Device().CreateBuffer(&desc);

    context.PushErrorScope();
    context.Device().InjectError(wgpu::ErrorType::Validation, "checked-completion-outer-scope-error");
    // Poison the encoder before the helper's scopes; finishing it must report validation too.
    context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
    const auto status = context.FlushAndWaitChecked(ep.BufferManager(), recording);
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("WebGPU error scope failed"));
    const auto outer_status = context.PopErrorScope();
    EXPECT_FALSE(outer_status.IsOK());
    EXPECT_THAT(outer_status.ErrorMessage(), testing::HasSubstr("checked-completion-outer-scope-error"));
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
    ExpectQueueCompleted(context);
  });
#endif
}

TEST(WebGpuContextTest, ErrorScopesPropagateValidationAndOutOfMemory) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native error injection is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&) {
    for (const auto filter : {wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory}) {
      SCOPED_TRACE(static_cast<uint32_t>(filter));
      const auto type = filter == wgpu::ErrorFilter::Validation ? wgpu::ErrorType::Validation
                                                                : wgpu::ErrorType::OutOfMemory;
      context.PushErrorScope(filter);
      context.Device().InjectError(type, "checked-completion-scope-error");
      const auto status = context.PopErrorScope();
      EXPECT_FALSE(status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-completion-scope-error"));
      ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
    }
  });
#endif
}

TEST(WebGpuContextTest, CheckedCompletionCleansScopesAfterWaitFailure) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device creation is unavailable.";
#else
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&) {
    const auto status = context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording());
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Failed to wait for the operation"));

    auto pop_status = std::make_shared<wgpu::PopErrorScopeStatus>(wgpu::PopErrorScopeStatus::Success);
    const auto future = context.Device().PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [pop_status](wgpu::PopErrorScopeStatus status, wgpu::ErrorType, wgpu::StringView) noexcept {
          *pop_status = status;
        });
    ASSERT_EQ(context.Instance().WaitAny(future, 0), wgpu::WaitStatus::Success);
    // No helper-owned scopes remain, even though its completion/scope waits failed.
    EXPECT_EQ(*pop_status, wgpu::PopErrorScopeStatus::Error);
  },
                                   false);
#endif
}

void TestCheckedCompletionUncapturedError(wgpu::ErrorType type) {
#if !GTEST_HAS_DEATH_TEST || defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  ORT_UNUSED_PARAMETER(type);
  GTEST_SKIP() << "Uncaptured errors require an isolated ORT-owned Dawn device.";
#else
  RunWithFreshDefaultContext([type]() {
    ConfigOptions options;
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    context.Device().InjectError(type, "checked-completion-uncaptured-error");
    const auto status = context.FlushAndWaitChecked(webgpu_ep.BufferManager(), webgpu_ep.Recording());
    EXPECT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("checked-completion-uncaptured-error"));
    EXPECT_FALSE(context.FlushAndWaitChecked(webgpu_ep.BufferManager(), webgpu_ep.Recording()).IsOK());
    context.PushErrorScope();
    EXPECT_TRUE(context.PopErrorScope().IsOK());
  });
#endif
}

TEST(WebGpuContextTest, CheckedCompletionReportsEarlierUncapturedValidation) {
  TestCheckedCompletionUncapturedError(wgpu::ErrorType::Validation);
}

TEST(WebGpuContextTest, CheckedCompletionReportsEarlierUncapturedOutOfMemory) {
  TestCheckedCompletionUncapturedError(wgpu::ErrorType::OutOfMemory);
}

TEST(WebGpuContextTest, CheckedCompletionDetectsLostExternalDevice) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native device loss injection is unavailable.";
#else
  for (bool pending_encoder : {false, true}) {
    RunWithExternalCompletionContext([pending_encoder](auto& context, auto& ep, const auto& lost) {
      auto& recording = ep.Recording();
      ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), recording));
      if (pending_encoder) {
        context.GetCommandEncoder(recording);
      }
      context.Device().ForceLoss(wgpu::DeviceLostReason::Unknown, "checked-completion-device-loss");

      // Dawn can report successful queue completion on a lost device.
      ExpectQueueCompleted(context);
      const auto status = context.FlushAndWaitChecked(ep.BufferManager(), recording);
      EXPECT_FALSE(status.IsOK());
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("WebGPU device lost"));
      EXPECT_TRUE(*lost);
      EXPECT_FALSE(context.FlushAndWaitChecked(ep.BufferManager(), recording).IsOK());
    });
  }
#endif
}

TEST(WebGpuContextTest, BufferReuseFollowsCachePolicyAndRecordingSubmission) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;

  for (auto mode : {webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::LazyRelease,
                    webgpu::BufferCacheMode::Bucket, webgpu::BufferCacheMode::Simple,
                    webgpu::BufferCacheMode::Graph, webgpu::BufferCacheMode::GraphSimple}) {
    SCOPED_TRACE(static_cast<int>(mode));
    webgpu::BufferManager manager(context, mode, webgpu::BufferCacheMode::Disabled,
                                  webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled);
    webgpu::CommandRecordingState first;
    webgpu::CommandRecordingState second;
    auto buffer = manager.Create(first, 64, usage);
    wgpu::Buffer retained{buffer};
    context.GetCommandEncoder(first).ClearBuffer(buffer, 0, 64);
    manager.Release(buffer, &first);

    auto within_batch = manager.Create(first, 64, usage);
    EXPECT_NE(within_batch, buffer);
    auto other_buffer = manager.Create(second, 64, usage);
    EXPECT_NE(other_buffer, buffer);
    context.GetCommandEncoder(second).ClearBuffer(other_buffer, 0, 64);
    ASSERT_STATUS_OK(context.Flush(manager, second));
    // A submission on another recording cannot make the first buffer reusable.
    auto after_other_submission = manager.Create(second, 64, usage);
    EXPECT_NE(after_other_submission, buffer);

    ASSERT_STATUS_OK(context.Flush(manager, first));
    auto reused = manager.Create(second, 64, usage);
    if (mode == webgpu::BufferCacheMode::Disabled || mode == webgpu::BufferCacheMode::LazyRelease) {
      EXPECT_NE(reused, buffer);
    } else {
      EXPECT_EQ(reused, buffer);
    }
    manager.Release(reused);
    manager.Release(within_batch);
    manager.Release(other_buffer);
    manager.Release(after_other_submission);
  }
}

TEST(WebGpuContextTest, BufferReleasedAfterSubmissionIsReusableWithoutAnotherFlush) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  webgpu::CommandRecordingState other;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = manager.Create(recording, 64, usage);
  wgpu::Buffer retained{buffer};
  context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
  ASSERT_STATUS_OK(context.Flush(manager, recording));

  // An allocator can outlive its last Run. Free must not wait for another submission to this recording.
  manager.Release(buffer, &recording);
  auto reused = manager.Create(other, 64, usage);
  EXPECT_EQ(reused, buffer);
  manager.Release(reused);
}

TEST(WebGpuContextTest, EmptyFlushRefreshesIdleCacheWithoutReleasingAnotherRecording) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Simple,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState idle;
  webgpu::CommandRecordingState active;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto idle_buffer = manager.Create(idle, 64, usage);
  auto active_buffer = manager.Create(active, 64, usage);
  wgpu::Buffer retained_idle{idle_buffer};
  wgpu::Buffer retained_active{active_buffer};
  context.GetCommandEncoder(active).ClearBuffer(active_buffer, 0, 64);
  manager.Release(active_buffer, &active);
  manager.Release(idle_buffer, &idle);

  ASSERT_STATUS_OK(context.Flush(manager, idle));
  auto reused = manager.Create(idle, 64, usage);
  EXPECT_EQ(reused, idle_buffer);
  auto still_pending = manager.Create(idle, 64, usage);
  EXPECT_NE(still_pending, active_buffer);

  ASSERT_STATUS_OK(context.Flush(manager, active));
  auto after_submission = manager.Create(idle, 64, usage);
  EXPECT_EQ(after_submission, active_buffer);
  manager.Release(reused);
  manager.Release(still_pending);
  manager.Release(after_submission);
}

TEST(WebGpuContextTest, IndependentClearPreservesPendingAndCapturedUniformBuffers) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Graph,
                                webgpu::BufferCacheMode::GraphSimple, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState session_recording;
  webgpu::CommandRecordingState independent_recording;
  constexpr auto uniform_usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
  constexpr auto storage_usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                                 wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Indirect;

  std::vector<webgpu::CapturedCommandInfo> captured_commands;
  context.CaptureBegin(&captured_commands, manager, session_recording);
  auto uniform_buffer = manager.Create(session_recording, 64, uniform_usage);
  wgpu::Buffer retained_uniform{uniform_buffer};
  context.GetCommandEncoder(session_recording).ClearBuffer(uniform_buffer, 0, 64);
  manager.Release(uniform_buffer, &session_recording);

  const auto submit_independent_clear = [&]() {
    // Exercise the cached clear and immediate submission used by plugin streamless allocations.
    std::array<uint32_t, 16> dirty_data;
    dirty_data.fill(0xffffffffu);
    auto dirty_buffer = manager.Create(independent_recording, sizeof(dirty_data), storage_usage);
    manager.Upload(independent_recording, dirty_data.data(), dirty_buffer, sizeof(dirty_data));
    manager.Release(dirty_buffer, &independent_recording);
    auto cleared_buffer = wgpu::Buffer::Acquire(
        manager.Create(independent_recording, sizeof(dirty_data), storage_usage, true, true));
    EXPECT_EQ(cleared_buffer.Get(), dirty_buffer);
    const std::array<uint32_t, 16> zeros{};
    EXPECT_EQ(ReadBufferWithExternalCommandEncoder(context, cleared_buffer.Get()), zeros);
    manager.Release(cleared_buffer.MoveToCHandle(), &independent_recording);
  };

  submit_independent_clear();
  auto before_submission = wgpu::Buffer::Acquire(manager.Create(independent_recording, 64, uniform_usage));
  EXPECT_NE(before_submission.Get(), uniform_buffer);
  EXPECT_TRUE(session_recording.has_unsubmitted_work.load(std::memory_order_relaxed));

  ASSERT_STATUS_OK(context.Flush(manager, session_recording));
  context.CaptureEnd(session_recording);
  submit_independent_clear();
  auto after_submission = wgpu::Buffer::Acquire(manager.Create(independent_recording, 64, uniform_usage));
  EXPECT_NE(after_submission.Get(), uniform_buffer);
}

TEST(WebGpuContextTest, GraphFlushRetiresSharedBuffersForSubmittedRecording) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kStorageBufferCacheMode, "bucket"));
    ASSERT_STATUS_OK(options.AddConfigEntry(kUniformBufferCacheMode, "simple"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    webgpu::BufferManager graph_manager(context, webgpu::BufferCacheMode::Graph,
                                        webgpu::BufferCacheMode::GraphSimple, webgpu::BufferCacheMode::Disabled,
                                        webgpu::BufferCacheMode::Disabled);

    // Exercise both the Bucket and Simple shared caches during capture and replay.
    for (auto state : {webgpu::GraphCaptureState::Capturing, webgpu::GraphCaptureState::Replaying}) {
      for (auto usage : {wgpu::BufferUsage::Storage, wgpu::BufferUsage::Uniform}) {
        SCOPED_TRACE(static_cast<int>(state));
        SCOPED_TRACE(static_cast<uint64_t>(usage));
        usage |= wgpu::BufferUsage::CopyDst;
        webgpu::CommandRecordingState session;
        webgpu::CommandRecordingState other;
        webgpu::CommandRecordingState idle;
        session.graph_capture_state = state;
        auto released = shared_manager.Create(idle, 64, usage);
        auto other_released = shared_manager.Create(idle, 64, usage);
        wgpu::Buffer retained{released};
        wgpu::Buffer other_retained{other_released};
        context.GetCommandEncoder(session).ClearBuffer(released, 0, 64);
        context.GetCommandEncoder(other).ClearBuffer(other_released, 0, 64);
        shared_manager.Release(released, &session);
        shared_manager.Release(other_released, &other);

        ASSERT_STATUS_OK(context.Flush(graph_manager, idle));
        auto before_submission = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_NE(before_submission.Get(), released);
        EXPECT_NE(before_submission.Get(), other_released);

        ASSERT_STATUS_OK(context.Flush(graph_manager, session));
        EXPECT_EQ(session.graph_capture_state, state);
        auto reused = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_EQ(reused.Get(), released);
        auto still_pending = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_NE(still_pending.Get(), other_released);

        ASSERT_STATUS_OK(context.Flush(shared_manager, other));
        auto other_reused = wgpu::Buffer::Acquire(shared_manager.Create(idle, 64, usage));
        EXPECT_EQ(other_reused.Get(), other_released);
      }
    }
  });
}

TEST(WebGpuContextTest, InitializerFlushRetiresSharedUniformBuffers) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kUniformBufferCacheMode, "simple"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    constexpr auto usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;

    // Prepacking flushes the initializer manager, while uniforms use the shared manager.
    // An empty flush must also promote idle buffers from SimpleCacheManager's pending list.
    for (bool encode_commands : {true, false}) {
      SCOPED_TRACE(encode_commands);
      webgpu::CommandRecordingState recording;
      auto released = shared_manager.Create(recording, 64, usage);
      wgpu::Buffer retained{released};
      if (encode_commands) {
        context.GetCommandEncoder(recording).ClearBuffer(released, 0, 64);
      }
      shared_manager.Release(released, &recording);
      ASSERT_STATUS_OK(context.Flush(context.InitializerBufferManager(), recording));
      auto reused = wgpu::Buffer::Acquire(shared_manager.Create(recording, 64, usage));
      EXPECT_EQ(reused.Get(), released);
    }
  });
}

TEST(WebGpuContextTest, FailedFlushDiscardsPendingBuffersFromBothManagers) {
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kStorageBufferCacheMode, "bucket"));
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    auto& context = webgpu::WebGpuContextFactory::GetContext(0);
    auto& shared_manager = context.BufferManager();
    webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                  webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                  webgpu::BufferCacheMode::Disabled);
    webgpu::CommandRecordingState recording;
    constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    auto released = manager.Create(recording, 64, usage);
    auto shared_released = shared_manager.Create(recording, 64, usage);
    wgpu::Buffer retained{released};
    wgpu::Buffer shared_retained{shared_released};
    webgpu::CapturedCommandInfo dispatch;
    dispatch.program_key = "missing_pipeline_for_failed_flush_test";
    recording.deferred_dispatches.push_back(std::move(dispatch));
    recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
    manager.Release(released, &recording);
    shared_manager.Release(shared_released, &recording);

    EXPECT_FALSE(context.Flush(manager, recording).IsOK());
    EXPECT_EQ(recording.command_encoder, nullptr);
    EXPECT_TRUE(recording.deferred_dispatches.empty());
    // Reusing the recording after failure must not revive discarded buffers in either manager.
    ASSERT_STATUS_OK(context.Flush(manager, recording));
    auto replacement = wgpu::Buffer::Acquire(manager.Create(recording, 64, usage));
    auto shared_replacement = wgpu::Buffer::Acquire(shared_manager.Create(recording, 64, usage));
    EXPECT_NE(replacement.Get(), released);
    EXPECT_NE(shared_replacement.Get(), shared_released);
  });
}

TEST(WebGpuContextTest, AbandonedRecordingDoesNotReturnBuffersToPool) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager manager(context, webgpu::BufferCacheMode::Bucket,
                                webgpu::BufferCacheMode::Disabled, webgpu::BufferCacheMode::Disabled,
                                webgpu::BufferCacheMode::Disabled);
  webgpu::CommandRecordingState recording;
  constexpr auto usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = manager.Create(recording, 64, usage);
  // Keep the handle alive so a new device allocation cannot reuse its address.
  wgpu::Buffer retained{buffer};
  context.GetCommandEncoder(recording).ClearBuffer(buffer, 0, 64);
  manager.Release(buffer, &recording);
  recording.command_encoder = nullptr;
  manager.DiscardPendingBuffers(recording);

  // Reuse the recording's address after abandoning its previous commands.
  context.GetCommandEncoder(recording);
  ASSERT_STATUS_OK(context.Flush(manager, recording));
  auto replacement = manager.Create(recording, 64, usage);
  EXPECT_NE(replacement, buffer);
  manager.Release(replacement);
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
      webgpu_ep->GetDeviceId(),
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
      webgpu_ep->GetDeviceId(),
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
      webgpu_ep->GetDeviceId(),
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

  for (bool enable_synchronization : {false, true}) {
    SCOPED_TRACE(enable_synchronization);
    auto adapters = instance.EnumerateAdapters(&adapter_options);
    ASSERT_FALSE(adapters.empty());
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
  RunWithFreshDefaultContext([]() {
    auto enabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    ASSERT_NE(enabled_ep, nullptr);
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
    enabled_ep.reset();

    auto disabled_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
    ASSERT_NE(disabled_ep, nullptr);
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
  });
#endif
}

TEST(WebGpuContextTest, EnableRobustnessUsesBuildDefault) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(ep, nullptr);
#ifdef NDEBUG
    EXPECT_TRUE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#else
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
#endif
  });
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

TEST(WebGpuContextTest, DeviceIdRejectsInvalidValueBeforeContextCreation) {
  for (const char* value : {"-1", "-32768", "32768", "2147483647", "2147483648", "", "1x", "0x", "x"}) {
    SCOPED_TRACE(value);
    ConfigOptions options;
    ASSERT_STATUS_OK(options.AddConfigEntry(kDeviceId, value));
    try {
      WebGpuProviderFactoryCreator::Create(options);
      FAIL() << "Expected deviceId to be rejected.";
    } catch (const OnnxRuntimeException& ex) {
      EXPECT_NE(std::string_view{ex.what()}.find("Invalid deviceId value"), std::string_view::npos);
    }
  }
}

TEST(WebGpuContextTest, AdapterIndexRejectsInvalidValue) {
  for (const char* value : {"-1", "1x"}) {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, value));
    EXPECT_THROW(WebGpuProviderFactoryCreator::Create(options), OnnxRuntimeException);
  }
}

TEST(WebGpuContextTest, AdapterIndexAcceptsNonNegativeInteger) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  },
                             /*compile_only_parent=*/true);
#endif
}

TEST(WebGpuContextTest, AdapterIndexSelectsPhysicalAdapter) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions options;
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_NE(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  });
#endif
}

TEST(WebGpuContextTest, AdapterIndexRejectsOutOfRangeValue) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  ConfigOptions options;
  const std::string adapter_index = std::to_string(std::numeric_limits<uint32_t>::max());
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, adapter_index.c_str()));

  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(options), OnnxRuntimeException);
#endif
}

TEST(WebGpuContextTest, AdapterIndexRejectsUnsupportedBuild) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "This build supports physical adapter enumeration.";
#else
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kAdapterIndex, "0"));
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

  try {
    WebGpuProviderFactoryCreator::Create(options);
    FAIL() << "Expected adapterIndex to be rejected by this build.";
  } catch (const OnnxRuntimeException& ex) {
    EXPECT_NE(std::string_view{ex.what()}.find("requires a native Dawn build"), std::string_view::npos);
  }
#endif
}

TEST(WebGpuContextTest, AdapterIndexRejectsConflictingSelectorOnReusedContext) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Physical adapter enumeration requires a native Dawn build.";
#else
  RunWithFreshDefaultContext([]() {
    ConfigOptions first_options;
    ORT_THROW_IF_ERROR(first_options.AddConfigEntry(kAdapterIndex, "0"));
    auto first_ep = WebGpuProviderFactoryCreator::Create(first_options)->CreateProvider();
    ASSERT_NE(first_ep, nullptr);

    ConfigOptions power_options;
    ORT_THROW_IF_ERROR(power_options.AddConfigEntry(kAdapterIndex, "0"));
    ORT_THROW_IF_ERROR(power_options.AddConfigEntry(kPowerPreference, kPowerPreference_LowPower));
    EXPECT_THROW(WebGpuProviderFactoryCreator::Create(power_options), OnnxRuntimeException);

    webgpu::WebGpuContextConfig backend_config;
    backend_config.adapter_index = 0;
    backend_config.backend_type = std::numeric_limits<int>::max();
    EXPECT_THROW(webgpu::WebGpuContextFactory::CreateContext(backend_config), OnnxRuntimeException);
  });
#endif
}

TEST(WebGpuContextTest, CompileOnlyContextDoesNotCreateDevice) {
  RunWithFreshDefaultContext([]() {
    auto options = RobustnessOptions("0");
    ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));

    auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();

    ASSERT_NE(ep, nullptr);
    EXPECT_EQ(webgpu::WebGpuContextFactory::GetContext(0).Device().Get(), nullptr);
  },
                             /*compile_only_parent=*/true);
}

TEST(WebGpuContextTest, EnableRobustnessIsIndependentFromValidationMode) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
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
  });
#endif
}

TEST(WebGpuContextTest, ConflictingExplicitValueWarnsAndKeepsFirstValue) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
    auto first_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("1"))->CreateProvider();
    ASSERT_NE(first_ep, nullptr);

    testing::internal::CaptureStderr();
    auto second_ep = WebGpuProviderFactoryCreator::Create(RobustnessOptions("0"))->CreateProvider();
    const std::string warning = testing::internal::GetCapturedStderr();

    ASSERT_NE(second_ep, nullptr);
    EXPECT_FALSE(DisableRobustnessToggleIsEnabled(webgpu::WebGpuContextFactory::GetContext(0)));
    EXPECT_NE(warning.find("already initialized"), std::string::npos);
    EXPECT_NE(warning.find("will be ignored"), std::string::npos);
  });
#endif
}

TEST(WebGpuContextTest, OmittedAndMatchingValuesDoNotWarn) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
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
  });
#endif
}

TEST(WebGpuContextTest, ExternalDeviceValueWarnsAndIsIgnored) {
#if defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Dawn native toggle inspection is unavailable.";
#else
  RunWithFreshDefaultContext([]() {
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
  });
#endif
}

#if !defined(__wasm__)
TEST(WebGpuContextTest, CanMapDeviceLocalMemory) {
  using webgpu::detail::CanMapDeviceLocalMemory;

  constexpr uint64_t kMiB = 1024 * 1024;
  constexpr auto kDeviceLocal = wgpu::HeapProperty::DeviceLocal;
  constexpr auto kHostMappable = wgpu::HeapProperty::HostVisible | wgpu::HeapProperty::HostCoherent;
  const std::array<wgpu::MemoryHeapInfo, 3> small_bar{{
      {kDeviceLocal, 20224 * kMiB},
      {kHostMappable, 64353 * kMiB},
      {kDeviceLocal | kHostMappable, 256 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 2> resizable_bar{{
      {kDeviceLocal | kHostMappable, 12216 * kMiB},
      {kHostMappable, 23781 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 1> unified_memory{{
      {kDeviceLocal | kHostMappable, 5461 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 2> larger_mappable_heap{{
      {kDeviceLocal, 256 * kMiB},
      {kDeviceLocal | kHostMappable, 16384 * kMiB},
  }};
  const std::array<wgpu::MemoryHeapInfo, 1> host_only{{
      {kHostMappable, 23781 * kMiB},
  }};

  EXPECT_FALSE(CanMapDeviceLocalMemory(small_bar));
  EXPECT_TRUE(CanMapDeviceLocalMemory(resizable_bar));
  EXPECT_TRUE(CanMapDeviceLocalMemory(unified_memory));
  EXPECT_TRUE(CanMapDeviceLocalMemory(larger_mappable_heap));
  EXPECT_FALSE(CanMapDeviceLocalMemory(host_only));
}
#endif  // !defined(__wasm__)

}  // namespace
}  // namespace test
}  // namespace onnxruntime
