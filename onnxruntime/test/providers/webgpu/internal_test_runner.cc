// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "ep/adapters.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/providers/webgpu/ep/ep.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"
#if !defined(DISABLE_CONTRIB_OPS)
#include "contrib_ops/webgpu/moe/block_fp8_expert_matmul.h"
#endif
#include "test/providers/webgpu/internal_test_api.h"

namespace onnxruntime::webgpu {
void CleanupKernelRegistries();
void CleanupWebGpuContexts();
}  // namespace onnxruntime::webgpu

namespace onnxruntime::test {
const WebGpuInternalTestHostApi* webgpu_test_host_api = nullptr;
const OrtLogger* webgpu_test_logger = nullptr;
struct WebGpuTestAccess {
  static webgpu::WebGpuContext& Context(const WebGpuExecutionProvider& ep) { return ep.context_; }
  static void ForceAlgorithm(WebGpuExecutionProvider& ep, int algorithm) {
    ep.forced_matmul_algorithm_ = algorithm < 0 ? std::nullopt : std::optional{static_cast<webgpu::MatMulAlgorithm>(algorithm)};
  }
};
}  // namespace onnxruntime::test

#ifdef _WIN32
#define WEBGPU_INTERNAL_TEST_EXPORT __declspec(dllexport)
#else
#define WEBGPU_INTERNAL_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" WEBGPU_INTERNAL_TEST_EXPORT OrtStatus* ConfigureWebGpuTestEp(OrtEp* ep, int algorithm) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  auto& impl = *static_cast<onnxruntime::WebGpuExecutionProvider*>(
      static_cast<onnxruntime::ep::adapter::Ep*>(ep)->EpImpl());
  onnxruntime::test::WebGpuTestAccess::ForceAlgorithm(impl, algorithm);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

extern "C" WEBGPU_INTERNAL_TEST_EXPORT OrtStatus* InspectWebGpuTestEp(
    const OrtEp* ep, onnxruntime::test::WebGpuTestCapabilities* result) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  using namespace onnxruntime;
  const auto& impl = *static_cast<const WebGpuExecutionProvider*>(static_cast<const ep::adapter::Ep*>(ep)->EpImpl());
  const auto& context = test::WebGpuTestAccess::Context(impl);
  const auto& adapter = context.AdapterInfo();
  *result = {};
  result->max_workgroup_size_x = context.DeviceLimits().maxComputeWorkgroupSizeX;
  result->max_workgroup_invocations = context.DeviceLimits().maxComputeInvocationsPerWorkgroup;
  result->supports_dp4a = context.DeviceHasFeature(wgpu::FeatureName::Subgroups) && adapter.vendor != std::string_view{"apple"};
  if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups)) {
    result->matmul_unavailable[static_cast<int>(webgpu::MatMulAlgorithm::Subgroup)] = "subgroup requires the WebGPU Subgroups feature.";
  } else if (!webgpu::intel::SelectMatMulSubgroupSize(adapter.subgroupMinSize, adapter.subgroupMaxSize,
                                                      context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl))
                  .has_value()) {
    result->matmul_unavailable[static_cast<int>(webgpu::MatMulAlgorithm::Subgroup)] = "subgroup requires a supported subgroup size (8, 16, or 32).";
  }
  if (context.GetSplitKConfig().GetSplitDimInner() == 0) {
    result->matmul_unavailable[static_cast<int>(webgpu::MatMulAlgorithm::PackedSplitK)] = "packed_split_k is not configured for the selected adapter.";
  }
  const auto& configs = context.SubgroupMatrixConfigs();
  const bool has_matrix = context.DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix);
  const bool size_control = context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl);
  constexpr auto f16 = wgpu::SubgroupMatrixComponentType::F16;
  constexpr auto f32 = wgpu::SubgroupMatrixComponentType::F32;
  if (!has_matrix || !webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
                         {configs.configs, configs.configCount}, adapter.subgroupMinSize, adapter.subgroupMaxSize, size_control,
                         {{f16, f16, 8, 16, 16, 32, false}})) {
    result->matmul_unavailable[static_cast<int>(webgpu::MatMulAlgorithm::SubgroupMatrix)] = "subgroup_matrix requires an 8x16x16 F16 configuration with subgroup size 32.";
  }
  result->supports_fp8_matrix = has_matrix && webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
                                                  {configs.configs, configs.configCount}, adapter.subgroupMinSize, adapter.subgroupMaxSize, size_control,
                                                  {{f16, f32, 16, 16, 16, 32, false}})
                                                  .has_value();
#if !defined(DISABLE_CONTRIB_OPS)
  result->fp8_matrix_dispatches = contrib::webgpu::BlockFp8MatrixDispatchCount();
#endif
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

extern "C" WEBGPU_INTERNAL_TEST_EXPORT OrtStatus* BeginWebGpuTestContext(
    const OrtSessionOptions* session_options, uint64_t max_storage_buffer_binding_size,
    void** context, bool* supports_dp4a) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  onnxruntime::ConfigOptions options;
  OrtKeyValuePairs* entries = nullptr;
  ORT_API_RETURN_IF_ERROR(onnxruntime::ep::Api().ort.GetSessionOptionsConfigEntries(session_options, &entries));
  Ort::KeyValuePairs holder(entries);
  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  size_t count = 0;
  onnxruntime::ep::Api().ort.GetKeyValuePairs(entries, &keys, &values, &count);
  for (size_t i = 0; i < count; ++i) {
    ORT_THROW_IF_ERROR(options.AddConfigEntry(keys[i], values[i]));
  }
  auto provider = onnxruntime::WebGpuProviderFactoryCreator::CreateForTesting(
                      options, {.max_storage_buffer_binding_size = max_storage_buffer_binding_size})
                      ->CreateProvider();
  const auto& webgpu_context = onnxruntime::webgpu::WebGpuContextFactory::GetContext(0);
  *supports_dp4a = webgpu_context.DeviceHasFeature(wgpu::FeatureName::Subgroups) &&
                   webgpu_context.AdapterInfo().vendor != std::string_view{"apple"};
  ORT_ENFORCE(max_storage_buffer_binding_size == 0 ||
              webgpu_context.DeviceLimits().maxStorageBufferBindingSize == max_storage_buffer_binding_size);
  *context = new decltype(provider)(std::move(provider));
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

extern "C" WEBGPU_INTERNAL_TEST_EXPORT void EndWebGpuTestContext(void* context) noexcept {
  delete static_cast<std::unique_ptr<onnxruntime::IExecutionProvider>*>(context);
}

extern "C" WEBGPU_INTERNAL_TEST_EXPORT int RunWebGpuInternalTests(
    const OrtApiBase* api_base, const OrtLogger* logger,
    const onnxruntime::test::WebGpuInternalTestHostApi* host_api, int argc, char** argv) {
  int result = 1;
  const std::string log_id = "WebGpuInternalTests";
  std::unique_ptr<onnxruntime::logging::LoggingManager> native_logging;
  ORT_TRY {
    onnxruntime::ep::ApiInit(api_base);
    onnxruntime::test::webgpu_test_host_api = host_api;
    onnxruntime::test::webgpu_test_logger = logger;
    onnxruntime::ep::adapter::LoggingManager::CreateDefaultLogger(logger);
    // Core utility code in this module also uses the native logging interface.
    if (!onnxruntime::logging::LoggingManager::HasDefaultLogger()) {
      native_logging = std::make_unique<onnxruntime::logging::LoggingManager>(
          std::make_unique<onnxruntime::logging::CLogSink>(), onnxruntime::logging::Severity::kWARNING,
          false, onnxruntime::logging::LoggingManager::InstanceType::Default, &log_id);
    }

    testing::InitGoogleTest(&argc, argv);
    result = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() { std::cerr << ex.what() << '\n'; });
  }

  onnxruntime::webgpu::CleanupKernelRegistries();
  onnxruntime::webgpu::CleanupWebGpuContexts();
  onnxruntime::ep::adapter::LoggingManager::DestroyDefaultLogger();
  google::protobuf::ShutdownProtobufLibrary();
  return result;
}

static_assert(std::is_same_v<decltype(&RunWebGpuInternalTests), onnxruntime::test::RunWebGpuInternalTestsFn>);
