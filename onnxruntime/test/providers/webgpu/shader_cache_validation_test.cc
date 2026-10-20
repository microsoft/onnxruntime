// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <thread>

#include "gtest/gtest.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN) && !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/tensor/cast.h"
#include "test/providers/webgpu/webgpu_context_test_helper.h"
#include "test/test_environment.h"
#endif

namespace onnxruntime::test {

TEST(WebGpuShaderCacheValidationTest, ReportsBothDescriptionsAndKeepsFirstObservation) {
  webgpu::detail::ShaderCacheValidator validator;
  ASSERT_STATUS_OK(validator.Validate("key", "first", "layout\nWGSL A\n"));
  ASSERT_STATUS_OK(validator.Validate("key", "another label", "layout\nWGSL A\n"));
  const auto status = validator.Validate("key", "second", "layout\nWGSL B\n");
  ASSERT_FALSE(status.IsOK());
  for (const char* text : {"shader cache collision", "key", "first", "second", "line: 2", "WGSL A", "WGSL B"}) {
    EXPECT_NE(status.ErrorMessage().find(text), std::string::npos) << text;
  }
  ASSERT_STATUS_OK(validator.Validate("key", "first", "layout\nWGSL A\n"));
  ASSERT_STATUS_OK(validator.Validate("different key", "second", "layout\nWGSL B\n"));
}

#if !defined(__wasm__)
TEST(WebGpuShaderCacheValidationTest, ConcurrentConflictingObservationsCannotBothSucceed) {
  webgpu::detail::ShaderCacheValidator validator;
  std::atomic<int> accepted{0};
  auto check = [&](const char* description) {
    if (validator.Validate("shared", "program", description).IsOK()) {
      ++accepted;
    }
  };
  std::thread first{check, "A"};
  std::thread second{check, "B"};
  first.join();
  second.join();
  EXPECT_EQ(accepted, 1);
}
#endif

namespace {
class DiagnosticProgram final : public webgpu::Program<DiagnosticProgram> {
 public:
  DiagnosticProgram(uint32_t workgroup_size, uint32_t payload, uint32_t factor, size_t& generations)
      : Program{"CacheDiagnosticTest"}, generations_{generations} {
    SetWorkgroupSize(workgroup_size);
    AddUniformVariables({payload});
    SetOverridableConstants({factor});
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DiagnosticProgram);
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"payload", webgpu::ProgramUniformVariableDataType::Uint32});
  WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS({"factor", uint32_t{1}});
  Status GenerateShaderCode(webgpu::ShaderHelper&) const override {
    ++generations_;
    return Status::OK();
  }

 private:
  size_t& generations_;
};
}  // namespace

TEST(WebGpuShaderCacheValidationTest, ChecksPendingAndCompletedDescriptionsWithoutCompilingAgain) {
  auto provider = DefaultWebGpuExecutionProvider();
  if (!provider) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  ScopedEnvironmentVariables enabled{{{"ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE", "1"}}};
  webgpu::ProgramManager manager{context};
  ASSERT_TRUE(manager.IsShaderCacheValidationEnabled());
  size_t generations = 0;
  DiagnosticProgram first{1, 10, 2, generations};
  DiagnosticProgram payload_only{1, 20, 2, generations};
  DiagnosticProgram source_change{2, 10, 2, generations};
  DiagnosticProgram override_change{1, 10, 3, generations};
  webgpu::PendingPipelineBuild build;
  build.callback_context = std::make_shared<webgpu::PipelineCallbackContext>();
  ASSERT_STATUS_OK(manager.Build(first, first.Metadata(), {}, {}, "same-key", 1, 1, 1,
                                 build.bind_group_layout, build.shape_uniform_ranks, build.future, build.callback_context));
  // Build records the description before its asynchronous pipeline has been published.
  const auto pending_status = manager.ValidateShaderCache(source_change, {}, {}, "same-key", 1, 1, 1);
  // Always drain the callback, even if the diagnostic assertion fails.
  ASSERT_STATUS_OK(context.Wait(build.future));
  ASSERT_STATUS_OK(build.callback_context->status);
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(pending_status, "WebGPU shader cache collision");
  const auto* artifact = manager.Set("same-key", webgpu::ProgramArtifact{
                                                     first.Name(), std::move(build.callback_context->pipeline), std::move(build.bind_group_layout),
                                                     std::move(build.shape_uniform_ranks)});
  ASSERT_NE(artifact->compute_pipeline, nullptr);
  ASSERT_STATUS_OK(manager.ValidateShaderCache(payload_only, {}, {}, "same-key", 1, 1, 1));
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(manager.ValidateShaderCache(override_change, {}, {}, "same-key", 1, 1, 1),
                                      "pipeline overrides");
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(manager.ValidateShaderCache(source_change, {}, {}, "same-key", 1, 1, 1),
                                      "WebGPU shader cache collision");
  EXPECT_EQ(manager.Get("same-key"), artifact);
  EXPECT_EQ(generations, 5u);

  ScopedEnvironmentVariables disabled{{{"ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE", "0"}}};
  webgpu::ProgramManager quiet{context};
  EXPECT_FALSE(quiet.IsShaderCacheValidationEnabled());
  ASSERT_STATUS_OK(quiet.ValidateShaderCache(source_change, {}, {}, "same-key", 1, 1, 1));
  EXPECT_EQ(generations, 5u);
}

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN) && !defined(ORT_USE_EP_API_ADAPTERS)
namespace {
class DiagnosticKernel final : public OpKernel {
 public:
  explicit DiagnosticKernel(const OpKernelInfo& info) : OpKernel(info) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DiagnosticKernel);
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};
}  // namespace

TEST(WebGpuShaderCacheValidationTest, RunRejectsCollisionBeforePendingAndReadyReuse) {
  ScopedEnvironmentVariables enabled{{{"ORT_WEBGPU_EP_VALIDATE_SHADER_CACHE", "1"}}};
  RunWithExternalCompletionContext([](auto& context, auto& ep, const auto&, auto&) {
    ep.SetLogger(&DefaultLoggingManager().DefaultLogger());
    ConfigOptions options;
    Model model("cache_diagnostic", false, DefaultLoggingManager().DefaultLogger());
    auto& node = model.MainGraph().AddNode("unused", "Identity", "", {}, {});
    auto kernel_def = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int, OrtValue> initializers;
    const OrtValueNameIdxMap values;
    const DataTransferManager transfers;
    const AllocatorMap allocators;
    OpKernelInfo info(node, *kernel_def, ep, initializers, values, transfers, allocators, options);
    DiagnosticKernel kernel{info};
    webgpu::ComputeContextBase compute_context{context, ep, kernel};
    wgpu::BufferDescriptor desc{};
    desc.size = 16;
    desc.usage = wgpu::BufferUsage::Storage;
    auto input_buffer = context.Device().CreateBuffer(&desc);
    auto output_buffer = context.Device().CreateBuffer(&desc);
    const OrtMemoryInfo memory_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(1), OrtMemTypeDefault);
    Tensor input(DataTypeImpl::GetType<float>(), {4}, input_buffer.Get(), memory_info);
    Tensor output_float(DataTypeImpl::GetType<float>(), {4}, output_buffer.Get(), memory_info);
    Tensor output_int(DataTypeImpl::GetType<int32_t>(), {4}, output_buffer.Get(), memory_info);
    webgpu::CastProgram first{ONNX_NAMESPACE::TensorProto_DataType_FLOAT, false, true, false, false};
    webgpu::CastProgram second{ONNX_NAMESPACE::TensorProto_DataType_INT32, false, true, false, false};
    auto configure = [&](auto& program, Tensor& output) {
      // Deliberately omit Cast's target-type hint, recreating a cache authoring error.
      program.CacheHint("diagnostic-cast")
          .AddInput({&input, webgpu::ProgramTensorMetadataDependency::Type, {1}, 4})
          .AddOutput({&output, webgpu::ProgramTensorMetadataDependency::None, {1}, 4})
          .SetDispatchGroupSize(1)
          .AddUniformVariables({uint32_t{1}, uint32_t{4}});
    };
    configure(first, output_float);
    configure(second, output_int);
    ASSERT_STATUS_OK(compute_context.RunProgram(first));
    EXPECT_EQ(ep.Recording().deferred_dispatches.size(), 1u);
    const auto pending_status = compute_context.RunProgram(second);
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(pending_status, "WebGPU shader cache collision");
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(compute_context.RunProgram(second), "WebGPU shader cache collision");
    ASSERT_STATUS_OK(compute_context.RunProgram(first));
    ASSERT_STATUS_OK(context.FlushAndWaitChecked(ep.BufferManager(), ep.Recording()));
  });
}
#endif

}  // namespace onnxruntime::test
