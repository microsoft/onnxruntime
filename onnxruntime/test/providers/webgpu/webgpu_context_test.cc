// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
#include "core/providers/webgpu/direct_storage_external_data_loader.h"
#endif

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

ConfigOptions WeightLoadAccelerationOptions(const char* value) {
  ConfigOptions options;
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kWeightLoadAcceleration, value));
  return options;
}

ConfigOptions CompileOnlyWeightLoadAccelerationOptions(const char* value) {
  auto options = WeightLoadAccelerationOptions(value);
  ORT_THROW_IF_ERROR(options.AddConfigEntry(kOrtSessionOptionCompileOnly, "1"));
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
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);

  const auto downloaded_data = ReadBufferWithExternalCommandEncoder(context, reused_buffer);
  ASSERT_STATUS_OK(context.Flush(buffer_manager));
  const std::array<uint32_t, 16> expected_data{};
  EXPECT_EQ(downloaded_data, expected_data);

  allocator.Free(allocation);
}

TEST(WebGpuContextTest, DoesNotCaptureDeviceAllocatorBufferClear) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  webgpu::BufferManager buffer_manager(context,
                                       webgpu::BufferCacheMode::Graph,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled,
                                       webgpu::BufferCacheMode::Disabled);
  std::vector<webgpu::CapturedCommandInfo> captured_commands;
  webgpu::GpuBufferAllocator allocator(
      [&buffer_manager]() -> const webgpu::BufferManager& { return buffer_manager; },
      false);

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  allocator.Free(allocation);

  context.CaptureBegin(&captured_commands, buffer_manager);
  allocation = allocator.Alloc(sizeof(nonzero_data));
  if (allocation == nullptr) {
    context.CaptureEnd();
    FAIL() << "Failed to reacquire a device allocation during graph capture.";
  }
  WGPUBuffer reused_buffer = static_cast<WGPUBuffer>(allocation);
  EXPECT_EQ(reused_buffer, dirty_buffer);
  const Status flush_status = context.Flush(buffer_manager);
  context.CaptureEnd();
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
      false,
      [webgpu_ep]() { return !webgpu_ep->IsRunActive(); });

  std::array<uint32_t, 16> nonzero_data;
  nonzero_data.fill(0xffffffffu);
  void* allocation = allocator.Alloc(sizeof(nonzero_data));
  ASSERT_NE(allocation, nullptr);
  WGPUBuffer dirty_buffer = static_cast<WGPUBuffer>(allocation);
  buffer_manager.Upload(nonzero_data.data(), dirty_buffer, sizeof(nonzero_data));
  ASSERT_STATUS_OK(context.Flush(buffer_manager));
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

TEST(WebGpuContextTest, WeightLoadAccelerationRejectsBooleanAndUnknownModes) {
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(
                   WeightLoadAccelerationOptions("1")),
               OnnxRuntimeException);
  EXPECT_THROW(WebGpuProviderFactoryCreator::Create(
                   WeightLoadAccelerationOptions("automatic")),
               OnnxRuntimeException);
}

TEST(WebGpuContextTest, WeightLoadAccelerationOffDoesNotRequireDeviceSupport) {
  auto ep = WebGpuProviderFactoryCreator::Create(
                CompileOnlyWeightLoadAccelerationOptions(
                    kWeightLoadAcceleration_Off))
                ->CreateProvider();
  ASSERT_NE(ep, nullptr);
}

TEST(WebGpuContextTest, PreferredWeightLoadAccelerationFallsBackWithoutDeviceSupport) {
  for (const char* mode : {kWeightLoadAcceleration_Preferred,
                           kWeightLoadAcceleration_PreferredPipelined}) {
    auto ep = WebGpuProviderFactoryCreator::Create(
                  CompileOnlyWeightLoadAccelerationOptions(mode))
                  ->CreateProvider();
    ASSERT_NE(ep, nullptr);
  }
}

TEST(WebGpuContextTest, RequiredWeightLoadAccelerationFailsWithoutDeviceSupport) {
  for (const char* mode : {kWeightLoadAcceleration_Required,
                           kWeightLoadAcceleration_RequiredPipelined}) {
    auto factory = WebGpuProviderFactoryCreator::Create(
        CompileOnlyWeightLoadAccelerationOptions(mode));
#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
    auto ep = factory->CreateProvider();
    ASSERT_NE(ep, nullptr);
    auto loader = ep->GetExternalDataLoader();
    ASSERT_NE(loader, nullptr);
    EXPECT_FALSE(loader->BeginPreload().IsOK());
#else
    EXPECT_THROW(factory->CreateProvider(), OnnxRuntimeException);
#endif
  }
}

#if defined(_WIN32) && defined(ENABLE_WEBGPU_DIRECT_STORAGE)
TEST(WebGpuContextTest, DirectStorageCanBeEnabledAfterInitialOffSession) {
  auto off_factory = WebGpuProviderFactoryCreator::Create(
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Off));
  auto off_ep = off_factory->CreateProvider();
  ASSERT_NE(off_ep, nullptr);

  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto support_status =
      webgpu::CheckDirectStorageExternalWeightsSupport(context);
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  auto required_factory = WebGpuProviderFactoryCreator::Create(
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Required));
  auto required_ep = required_factory->CreateProvider();
  ASSERT_NE(required_ep, nullptr);
  auto loader = required_ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  EXPECT_STATUS_OK(loader->BeginPreload());
  loader->AbortLoad();
}

TEST(WebGpuContextTest, RequiredPipelinedRejectsNonPipelinedExistingContext) {
  auto off_options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Off);
  auto off_ep =
      WebGpuProviderFactoryCreator::Create(off_options)->CreateProvider();
  ASSERT_NE(off_ep, nullptr);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  context.WaitForInitializeComplete();
  const auto support_status =
      webgpu::CheckDirectStorageExternalWeightsSupport(context);
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  auto required_options = WeightLoadAccelerationOptions(
      kWeightLoadAcceleration_RequiredPipelined);
  auto required_ep =
      WebGpuProviderFactoryCreator::Create(required_options)->CreateProvider();
  ASSERT_NE(required_ep, nullptr);
  auto loader = required_ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto status = loader->BeginPreload();
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("required-pipelined"),
            std::string::npos);
}

TEST(WebGpuContextTest, DirectStorageLoadsExternalTensorAndEmptyTensor) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_direct_storage_external_tensor_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  constexpr size_t kDataOffset = 32;
  const std::array<uint32_t, 16> expected{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    const std::array<char, kDataOffset> prefix{};
    stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
    stream.write(reinterpret_cast<const char*>(expected.data()),
                 static_cast<std::streamsize>(sizeof(expected)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Required);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto& allocator = allocators.front();

  const auto support_status = webgpu::CheckDirectStorageExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginPreload());
  ASSERT_STATUS_OK(loader->PreloadTensor(
      Env::Default(), data_path, "weights", kDataOffset, sizeof(expected)));
  ASSERT_STATUS_OK(loader->PreloadTensor(
      Env::Default(), data_path, "empty", kDataOffset + sizeof(expected), 0));
  ASSERT_STATUS_OK(loader->FinalizePreload([]() { return false; }));

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->PrepareTensor(
      Env::Default(), data_path, "weights", kDataOffset, sizeof(expected)));
  ASSERT_STATUS_OK(loader->PrepareTensor(
      Env::Default(), data_path, "empty", kDataOffset + sizeof(expected), 0));
  ASSERT_STATUS_OK(loader->FinalizeLoad([]() { return false; }));

  Tensor weights{DataTypeImpl::GetType<uint32_t>(), TensorShape({16}),
                 nullptr, allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), data_path, "weights", kDataOffset, sizeof(expected),
      allocator, weights));
  EXPECT_EQ(ReadBufferWithExternalCommandEncoder(
                webgpu::WebGpuContextFactory::GetContext(0),
                reinterpret_cast<WGPUBuffer>(weights.MutableDataRaw())),
            expected);

  Tensor empty{DataTypeImpl::GetType<uint32_t>(), TensorShape({0}), nullptr,
               allocator};
  ASSERT_STATUS_OK(loader->LoadTensor(
      Env::Default(), data_path, "empty", kDataOffset + sizeof(expected), 0,
      allocator, empty));
  EXPECT_EQ(empty.SizeInBytes(), 0u);
  EXPECT_EQ(empty.MutableDataRaw(), nullptr);
}

TEST(WebGpuContextTest, PreferredDirectStoragePreservesCancellation) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_direct_storage_cancellation_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  const auto support_status = webgpu::CheckDirectStorageExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->PrepareTensor(
      Env::Default(), data_path, "weights", 0, sizeof(data)));
  const auto status = loader->FinalizeLoad([]() { return true; });
  EXPECT_EQ(status.Code(), common::MODEL_LOAD_CANCELED);
  loader->AbortLoad();
}

TEST(WebGpuContextTest, PreferredDirectStorageFallsBackAfterOperationalFailure) {
  TemporaryDirectory temp_dir{
      ORT_TSTR("webgpu_direct_storage_fallback_test")};
  const auto data_path =
      std::filesystem::path{temp_dir.Path()} / ORT_TSTR("weights.bin");
  const std::array<uint32_t, 16> data{};
  {
    std::ofstream stream{data_path,
                         std::ios::binary | std::ios::trunc};
    ASSERT_TRUE(stream.good());
    stream.write(reinterpret_cast<const char*>(data.data()),
                 static_cast<std::streamsize>(sizeof(data)));
    ASSERT_TRUE(stream.good());
  }

  auto options =
      WeightLoadAccelerationOptions(kWeightLoadAcceleration_Preferred);
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  auto loader = ep->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto& allocator = allocators.front();
  const auto support_status = webgpu::CheckDirectStorageExternalWeightsSupport(
      webgpu::WebGpuContextFactory::GetContext(0));
  if (!support_status.IsOK()) {
    GTEST_SKIP() << support_status.ErrorMessage();
  }

  ASSERT_STATUS_OK(loader->BeginLoad());
  ASSERT_STATUS_OK(loader->PrepareTensor(
      Env::Default(), data_path, "weights", 0, sizeof(data)));
  ASSERT_TRUE(std::filesystem::remove(data_path));
  EXPECT_STATUS_OK(loader->FinalizeLoad([]() { return false; }));
  EXPECT_FALSE(loader->CanLoad(allocator->Info()));
}

TEST(WebGpuContextTest, WeightLoadAccelerationModeResolution) {
  const auto unsupported =
      ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DirectStorage unavailable");
  bool enabled = true;

  EXPECT_STATUS_OK(webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::Off, unsupported, enabled));
  EXPECT_FALSE(enabled);

  enabled = true;
  EXPECT_STATUS_OK(webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::Preferred, unsupported, enabled));
  EXPECT_FALSE(enabled);

  enabled = false;
  EXPECT_STATUS_OK(webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::PreferredPipelined,
      common::Status::OK(), enabled));
  EXPECT_TRUE(enabled);

  enabled = true;
  const auto required_status = webgpu::ResolveWeightLoadAccelerationMode(
      webgpu::WeightLoadAccelerationMode::RequiredPipelined, unsupported,
      enabled);
  EXPECT_FALSE(required_status.IsOK());
  EXPECT_FALSE(enabled);
  EXPECT_EQ(required_status.ErrorMessage(), unsupported.ErrorMessage());
}
#endif

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
