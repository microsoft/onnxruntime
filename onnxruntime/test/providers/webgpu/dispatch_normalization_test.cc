// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/gemm_packed.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/nn/conv2d_mm.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/vendor/intel/math/gemm.h"
#include "core/providers/webgpu/vendor/intel/math/matmul.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

using namespace webgpu;

TEST(WebGpuDispatchNormalizationTest, CeilDivHandlesIntegerLimits) {
  constexpr uint32_t max_u32 = std::numeric_limits<uint32_t>::max();
  constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
  constexpr int64_t max_i64 = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(CeilDiv(uint32_t{0}, uint32_t{2}), 0U);
  EXPECT_EQ(CeilDiv(uint32_t{8}, uint32_t{2}), 4U);
  EXPECT_EQ(CeilDiv(uint32_t{9}, uint32_t{2}), 5U);
  EXPECT_EQ(CeilDiv(max_u32, uint32_t{1}), max_u32);
  EXPECT_EQ(CeilDiv(max_u32, uint32_t{2}), max_u32 / 2 + 1);
  EXPECT_EQ(CeilDiv(max_u64, uint64_t{2}), max_u64 / 2 + 1);
  EXPECT_EQ(CeilDiv(max_i64, int64_t{1}), max_i64);
  EXPECT_EQ(CeilDiv(max_i64, int64_t{2}), max_i64 / 2 + 1);
}

TEST(WebGpuDispatchNormalizationTest, CeilDivRejectsInvalidArguments) {
  EXPECT_THROW(CeilDiv(uint32_t{1}, uint32_t{0}), OnnxRuntimeException);
  EXPECT_THROW(CeilDiv(int64_t{1}, int64_t{-1}), OnnxRuntimeException);
  EXPECT_THROW(CeilDiv(int64_t{-1}, int64_t{2}), OnnxRuntimeException);
}

TEST(WebGpuDispatchNormalizationTest, Conv2dMMRejectsSpatialSizeOverflow) {
  std::array<float, 1> data{};
  const OrtMemoryInfo memory_info;
  Tensor x(DataTypeImpl::GetType<float>(), {1, 1, 1, 1}, data.data(), memory_info);
  Tensor w(DataTypeImpl::GetType<float>(), {1, 1, 1, 1}, data.data(), memory_info);
  Tensor output(DataTypeImpl::GetType<float>(), {1, 1, 1, 1}, data.data(), memory_info);
  const Activation activation;
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  for (bool channels_last : {false, true}) {
    SCOPED_TRACE(channels_last);
    const TensorShape output_shape = channels_last ? TensorShape{1, max, 2, 1} : TensorShape{1, 1, max, 2};
    EXPECT_THROW(CreateConv2dMMProgram(
                     activation, {&x, &w}, {0, 0, 0, 0}, {1, 1}, {1, 1}, &output,
                     1, 1, 1, channels_last, {x.Shape(), w.Shape(), output_shape}),
                 OnnxRuntimeException);
  }
}

TEST(WebGpuDispatchNormalizationTest, PreservesValidGridsAndNormalizesOversizedAxes) {
  struct Case {
    std::array<uint32_t, 3> logical;
    uint32_t limit;
    std::array<uint32_t, 3> expected;
  };
  const Case cases[] = {
      {{1, 1, 65535}, 65535, {1, 1, 65535}},
      {{1, 1, 65536}, 65535, {256, 256, 1}},
      {{1, 1, 70000}, 65535, {265, 265, 1}},
      {{1, 1, 131072}, 65535, {363, 363, 1}},
      {{7, 1, 1}, 5, {3, 3, 1}},
      {{1, 7, 1}, 5, {3, 3, 1}},
      {{1, 1, 7}, 5, {3, 3, 1}},
      {{7, 3, 2}, 5, {4, 4, 4}},
      {{65535, 65535, 1}, 65535, {65535, 65535, 1}},
  };
  for (const auto& test_case : cases) {
    auto actual = test_case.logical;
    SCOPED_TRACE(::testing::PrintToString(actual));
    ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(
        actual[0], actual[1], actual[2], test_case.limit));
    EXPECT_EQ(actual, test_case.expected);
  }
}

TEST(WebGpuDispatchNormalizationTest, RejectsInvalidAndUnrepresentableCounts) {
  constexpr uint32_t max = std::numeric_limits<uint32_t>::max();
  const std::array<uint32_t, 3> invalid_grids[] = {
      {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {65536, 65536, 1}, {max, max, max}, {1, 1, max}};
  for (auto actual : invalid_grids) {
    SCOPED_TRACE(::testing::PrintToString(actual));
    const auto original = actual;
    const auto status = webgpu::detail::NormalizeDispatchGroupSize(actual[0], actual[1], actual[2], 65535);
    EXPECT_FALSE(status.IsOK());
    EXPECT_EQ(actual, original);
  }
  uint32_t overflow_x = 1, overflow_y = 1, overflow_z = max;
  const auto normalized_overflow = webgpu::detail::NormalizeDispatchGroupSize(
      overflow_x, overflow_y, overflow_z, 65535);
  EXPECT_NE(normalized_overflow.ErrorMessage().find("normalized dispatch group count"), std::string::npos);
  uint32_t x = 1, y = 1, z = 1;
  EXPECT_FALSE(webgpu::detail::NormalizeDispatchGroupSize(x, y, z, 0).IsOK());
  x = 126;
  EXPECT_FALSE(webgpu::detail::NormalizeDispatchGroupSize(x, y, z, 5).IsOK());

  x = max;
  ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(x, y, z, max));
  EXPECT_EQ(x, max);
}

class WebGpuDispatchShaderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    WebGpuContextConfig config;
    config.compile_only = true;
    context_ = &WebGpuContextFactory::CreateContext(config);
  }

  void TearDown() override {
    if (context_ != nullptr) {
      WebGpuContextFactory::ReleaseContext(0);
    }
  }

  void CheckGuard(const ProgramBase& program, bool split_k = false) {
    std::vector<uint32_t> inputs_segments(program.Inputs().size(), 1);
    std::vector<uint32_t> outputs_segments(program.Outputs().size(), 1);
    ShaderHelper shader{program, program.Metadata(), *context_,
                        inputs_segments, outputs_segments, 3, 3, 1};
    ASSERT_STATUS_OK(program.GenerateShaderCode(shader));
    const std::string body = std::move(shader.MainFunctionBody()).str();
    const auto guard = body.find("if (logical_workgroup_id_z >= uniforms.logical_dispatch_z) { return; }");
    ASSERT_NE(guard, std::string::npos) << body;
    EXPECT_EQ(body.find("uniforms.logical_dispatch_x * uniforms.logical_dispatch_y"), std::string::npos);
    EXPECT_EQ(body.substr(0, guard).find("local_id"), std::string::npos)
        << "The return must be workgroup-uniform.";
    for (const char* operation : {"mm_readA(", "mm_readB(", "mm_write(", "workgroupBarrier()"}) {
      const auto position = body.find(operation);
      if (position != std::string::npos) {
        EXPECT_LT(guard, position) << operation;
      }
    }
    if (split_k) {
      const auto split = body.find("let split_index");
      ASSERT_NE(split, std::string::npos);
      EXPECT_LT(guard, split);
    }
  }

 private:
  WebGpuContext* context_{nullptr};
};

TEST_F(WebGpuDispatchShaderTest, PackedMatMulScalarVec4AndSplitK) {
  std::vector<float> data(2 * 8 * 64);
  const OrtMemoryInfo memory_info;
  for (bool vec4 : {false, true}) {
    for (uint32_t split_dim_inner : {1U, 32U}) {
      if (!vec4 && split_dim_inner > 1) continue;
      SCOPED_TRACE(::testing::PrintToString(std::make_pair(vec4, split_dim_inner)));
      const int components = vec4 ? 4 : 1;
      Tensor a(DataTypeImpl::GetType<float>(), {2, 8, 64}, data.data(), memory_info);
      Tensor b(DataTypeImpl::GetType<float>(), {2, 64, 8}, data.data(), memory_info);
      Tensor output(DataTypeImpl::GetType<float>(), {2, 8, 8}, data.data(), memory_info);
      InlinedVector<int64_t> elements_per_thread{4, 1, 1};
      MatMulProgram program{Activation{}, false, vec4, elements_per_thread, true, split_dim_inner};
      ProgramOutput program_output{&output, ProgramTensorMetadataDependency::Rank, components};
      program_output.is_atomic = split_dim_inner > 1;
      program.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank, components},
                         {&b, ProgramTensorMetadataDependency::TypeAndRank, components}})
          .AddOutput(std::move(program_output))
          .AddIndices(TensorShape{2})
          .SetWorkgroupSize(8, 8, 1);
      ASSERT_NO_FATAL_FAILURE(CheckGuard(program, split_dim_inner > 1));
    }
  }
}

TEST_F(WebGpuDispatchShaderTest, PackedGemmAndIntelSubgroupPrograms) {
  std::vector<float> data(8 * 8);
  const OrtMemoryInfo memory_info;
  Tensor a(DataTypeImpl::GetType<float>(), {8, 8}, data.data(), memory_info);
  Tensor b(DataTypeImpl::GetType<float>(), {8, 8}, data.data(), memory_info);
  Tensor output(DataTypeImpl::GetType<float>(), {8, 8}, data.data(), memory_info);
  for (bool vec4 : {false, true}) {
    const int components = vec4 ? 4 : 1;
    GemmProgram gemm{false, false, 1.0f, false, true, false, components, vec4};
    gemm.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank, components},
                    {&b, ProgramTensorMetadataDependency::TypeAndRank, components}})
        .AddOutput({&output, ProgramTensorMetadataDependency::Rank, components})
        .SetWorkgroupSize(8, 8, 1);
    ASSERT_NO_FATAL_FAILURE(CheckGuard(gemm));

    InlinedVector<int64_t> elements_per_thread{4, 1, 1};
    intel::GemmSubgroupProgram intel_gemm{false, false, 1.0f, false, true,
                                          false, vec4, false, false, elements_per_thread};
    intel_gemm.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank},
                          {&b, ProgramTensorMetadataDependency::TypeAndRank, components}})
        .AddOutput({&output, ProgramTensorMetadataDependency::Rank, components})
        .SetWorkgroupSize(256, 1, 1);
    ASSERT_NO_FATAL_FAILURE(CheckGuard(intel_gemm));

    intel::MatMulSubgroupProgram intel_matmul{Activation{}, false, vec4, false,
                                              false, true, elements_per_thread};
    intel_matmul.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank},
                            {&b, ProgramTensorMetadataDependency::TypeAndRank, components}})
        .AddOutput({&output, ProgramTensorMetadataDependency::Rank, components})
        .AddIndices(TensorShape{})
        .SetWorkgroupSize(256, 1, 1);
    ASSERT_NO_FATAL_FAILURE(CheckGuard(intel_matmul));
  }
}

TEST_F(WebGpuDispatchShaderTest, Conv2dMMBothLayouts) {
  std::vector<float> data(8 * 8 * 2 * 2);
  const OrtMemoryInfo memory_info;
  Tensor x(DataTypeImpl::GetType<float>(), {2, 8, 3, 3}, data.data(), memory_info);
  Tensor w(DataTypeImpl::GetType<float>(), {8, 8, 2, 2}, data.data(), memory_info);
  Tensor output(DataTypeImpl::GetType<float>(), {2, 8, 2, 2}, data.data(), memory_info);
  const Activation activation;
  for (bool channels_last : {false, true}) {
    const std::vector<TensorShape> shapes = channels_last
                                                ? std::vector<TensorShape>{{2, 3, 3, 8}, {2, 2, 8, 8}, {2, 2, 2, 8}}
                                                : std::vector<TensorShape>{x.Shape(), {2, 2, 8, 8}, output.Shape()};
    auto program = CreateConv2dMMProgram(
        activation, {&x, &w}, {0, 0, 0, 0}, {1, 1}, {1, 1}, &output,
        channels_last ? 4 : 8, channels_last ? 8 : 4, 32, channels_last, shapes);
    ASSERT_NO_FATAL_FAILURE(CheckGuard(program));
  }
}

class DispatchTestKernel final : public OpKernel {
 public:
  explicit DispatchTestKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};

void RunAndReadProgram(WebGpuExecutionProvider& ep, const ProgramBase& program,
                       WGPUBuffer output_buffer, std::vector<float>& output_data) {
  auto& context = WebGpuContextFactory::GetContext(0);
  ConfigOptions options;
  Model model("dispatch_padding", false, DefaultLoggingManager().DefaultLogger());
  auto& node = model.MainGraph().AddNode("unused", "Identity", "", {}, {});
  auto kernel_def = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
  const std::unordered_map<int, OrtValue> initializers;
  const OrtValueNameIdxMap values;
  const DataTransferManager transfers;
  const AllocatorMap allocators;
  OpKernelInfo info(node, *kernel_def, ep, initializers, values, transfers, allocators, options);
  DispatchTestKernel kernel(info);
  ComputeContextBase compute_context(context, ep, kernel);
  ASSERT_STATUS_OK(compute_context.RunProgram(program));
  ep.BufferManager().Download(ep.Recording(), output_buffer, output_data.data(), output_data.size() * sizeof(float));
}

wgpu::Buffer CreateStorageBuffer(WebGpuExecutionProvider& ep, gsl::span<float> data) {
  auto& context = WebGpuContextFactory::GetContext(0);
  wgpu::BufferDescriptor desc{};
  desc.size = data.size() * sizeof(float);
  desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = context.Device().CreateBuffer(&desc);
  ep.BufferManager().Upload(ep.Recording(), data.data(), buffer.Get(), desc.size);
  return buffer;
}

// Bind larger physical buffers while keeping logical tensor shapes unchanged.
// Even a broken shader stays in-bounds physically; padding writes fail the test.
void CheckMatMulPadding(uint32_t batches, uint32_t m, uint32_t n, uint32_t k,
                        uint32_t dispatch_limit, bool broadcast_a = false, uint32_t splits = 1) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  ep->SetLogger(&DefaultLoggingManager().DefaultLogger());
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = WebGpuContextFactory::GetContext(0);
  ASSERT_TRUE(context.HasDevice());

  const bool vec4 = n % 4 == 0 && k % 4 == 0;
  const int components = vec4 ? 4 : 1;
  InlinedVector<int64_t> elements_per_thread{4, 1, 1};
  const std::array<uint32_t, 3> logical{(n + 31) / 32, (m + 7) / 8, batches * splits};
  auto physical = logical;
  ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(
      physical[0], physical[1], physical[2], dispatch_limit));
  const uint32_t physical_count = physical[0] * physical[1] * physical[2];
  const uint32_t padded_batches = (physical_count + logical[0] * logical[1] * splits - 1) /
                                  (logical[0] * logical[1] * splits);
  const size_t output_size = static_cast<size_t>(batches) * m * n;
  constexpr float canary = -12345.0f;
  std::vector<float> a_data(static_cast<size_t>(padded_batches) * m * k, 1.0f);
  std::vector<float> b_data(static_cast<size_t>(padded_batches) * k * n, 1.0f);
  std::vector<float> output_data(static_cast<size_t>(padded_batches) * m * n, canary);
  if (splits > 1) {
    std::fill_n(output_data.begin(), output_size, 0.0f);
  }

  auto a_buffer = CreateStorageBuffer(webgpu_ep, a_data);
  auto b_buffer = CreateStorageBuffer(webgpu_ep, b_data);
  auto output_buffer = CreateStorageBuffer(webgpu_ep, output_data);
  const OrtMemoryInfo memory_info(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice, OrtMemTypeDefault);
  Tensor a(DataTypeImpl::GetType<float>(), {broadcast_a ? 1 : batches, m, k}, a_buffer.Get(), memory_info);
  Tensor b(DataTypeImpl::GetType<float>(), {batches, k, n}, b_buffer.Get(), memory_info);
  Tensor output(DataTypeImpl::GetType<float>(), {batches, m, n}, output_buffer.Get(), memory_info);
  MatMulProgram program{Activation{}, false, vec4, elements_per_thread, true, splits > 1 ? k / splits : 1};
  ProgramOutput program_output{&output, ProgramTensorMetadataDependency::Rank, components};
  program_output.is_atomic = splits > 1;
  program.CacheHint(vec4, splits)
      .AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank, components},
                  {&b, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .AddOutput(std::move(program_output))
      .AddIndices(TensorShape{batches})
      .AddUniformVariables({{m}, {n}, {k}, {logical[0]}, {logical[1]}, {logical[2]}, {splits}})
      .SetWorkgroupSize(8, 8, 1)
      .SetDispatchGroupSize(physical[0], physical[1], physical[2]);
  AppendActivationUniformsData(Activation{}, program);
  ASSERT_NO_FATAL_FAILURE(RunAndReadProgram(webgpu_ep, program, output_buffer.Get(), output_data));
  EXPECT_TRUE(std::all_of(output_data.begin(), output_data.begin() + output_size,
                          [k](float value) { return value == static_cast<float>(k); }));
  EXPECT_TRUE(std::all_of(output_data.begin() + output_size, output_data.end(),
                          [](float value) { return value == canary; }))
      << "Normalization-added workgroups modified output padding.";
}

TEST(WebGpuDispatchExecutionTest, PackedMatMulPreservesPadding) {
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(7, 8, 8, 8, 5));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(7, 8, 8, 7, 5));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(7, 8, 8, 8, 5, true));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(2, 24, 224, 8, 5));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(2, 56, 96, 7, 5));
}

TEST(WebGpuDispatchExecutionTest, SplitKPreservesPaddingAndValidSplits) {
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(7, 8, 8, 96, 5, false, 3));
}

TEST(WebGpuDispatchExecutionTest, LargeBatchAndExactDispatchControls) {
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(65535, 8, 8, 8, 65535));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(65536, 8, 8, 8, 65535));
  ASSERT_NO_FATAL_FAILURE(CheckMatMulPadding(70000, 8, 8, 8, 65535));
}

TEST(WebGpuDispatchExecutionTest, Conv2dMMPreservesPaddingInBothLayouts) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(ep, nullptr);
  ep->SetLogger(&DefaultLoggingManager().DefaultLogger());
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  const OrtMemoryInfo memory_info(WEBGPU_BUFFER, OrtDeviceAllocator, WebGpuDevice, OrtMemTypeDefault);
  constexpr uint32_t batches = 7;
  constexpr float canary = -12345.0f;
  std::array<uint32_t, 3> physical{1, 1, batches};
  ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(physical[0], physical[1], physical[2], 5));
  const uint32_t padded_batches = physical[0] * physical[1] * physical[2];
  std::vector<float> x_data(padded_batches * 8 * 3 * 3, 1.0f);
  std::vector<float> w_data(8 * 8 * 2 * 2, 1.0f);
  auto x_buffer = CreateStorageBuffer(webgpu_ep, x_data);
  auto w_buffer = CreateStorageBuffer(webgpu_ep, w_data);
  const Activation activation;
  for (bool channels_last : {false, true}) {
    SCOPED_TRACE(channels_last);
    std::vector<float> output_data(padded_batches * 8 * 2 * 2, canary);
    auto output_buffer = CreateStorageBuffer(webgpu_ep, output_data);
    const TensorShape x_shape = channels_last ? TensorShape{batches, 3, 3, 8} : TensorShape{batches, 8, 3, 3};
    const TensorShape output_shape = channels_last ? TensorShape{batches, 2, 2, 8} : TensorShape{batches, 8, 2, 2};
    Tensor x(DataTypeImpl::GetType<float>(), x_shape, x_buffer.Get(), memory_info);
    Tensor w(DataTypeImpl::GetType<float>(), {2, 2, 8, 8}, w_buffer.Get(), memory_info);
    Tensor output(DataTypeImpl::GetType<float>(), output_shape, output_buffer.Get(), memory_info);
    auto program = CreateConv2dMMProgram(
        activation, {&x, &w}, {0, 0, 0, 0}, {1, 1}, {1, 1}, &output,
        channels_last ? 4 : 8, channels_last ? 8 : 4, 32, channels_last,
        {x_shape, w.Shape(), output_shape});
    program.SetDispatchGroupSize(physical[0], physical[1], physical[2]);
    ASSERT_NO_FATAL_FAILURE(RunAndReadProgram(webgpu_ep, program, output_buffer.Get(), output_data));
    const auto padding = output_data.begin() + batches * 8 * 2 * 2;
    EXPECT_TRUE(std::all_of(output_data.begin(), padding, [](float value) { return value == 32.0f; }));
    EXPECT_TRUE(std::all_of(padding, output_data.end(), [](float value) { return value == canary; }));
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
