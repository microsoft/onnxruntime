// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/gemm_utils.h"
#include "core/providers/webgpu/math/matmul_packed.h"
#include "core/providers/webgpu/math/matmul_utils.h"
#include "core/providers/webgpu/nn/conv2d_mm.h"
#include "core/providers/webgpu/program_manager.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

using namespace webgpu;

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

// Supplies node metadata to ComputeContextBase; Compute() is not called.
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
  const size_t size = data.size() * sizeof(float);
  auto buffer = wgpu::Buffer::Acquire(ep.BufferManager().Create(
      ep.Recording(), size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst));
  ep.BufferManager().Upload(ep.Recording(), data.data(), buffer.Get(), size);
  return buffer;
}

class WebGpuDispatchExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConfigOptions options;
    ep_ = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
    ASSERT_NE(ep_, nullptr);
    ep_->SetLogger(&DefaultLoggingManager().DefaultLogger());
    ASSERT_TRUE(WebGpuContextFactory::GetContext(0).HasDevice());
  }

  WebGpuExecutionProvider& Ep() { return static_cast<WebGpuExecutionProvider&>(*ep_); }

 private:
  std::unique_ptr<IExecutionProvider> ep_;
};

TEST_F(WebGpuDispatchExecutionTest, PackedMatMulSupportsNonDefaultValidTuning) {
  constexpr uint32_t M = 4;
  constexpr uint32_t N = 64;
  constexpr uint32_t K = 32;
  std::vector<float> a_data(M * K);
  std::vector<float> b_data(K * N);
  std::vector<float> output_data(M * N, 0.0f);
  for (size_t i = 0; i < a_data.size(); ++i) {
    a_data[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.125f;
  }
  for (size_t i = 0; i < b_data.size(); ++i) {
    b_data[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.0625f;
  }

  auto a_buffer = CreateStorageBuffer(Ep(), a_data);
  auto b_buffer = CreateStorageBuffer(Ep(), b_data);
  auto output_buffer = CreateStorageBuffer(Ep(), output_data);
  const OrtMemoryInfo memory_info(
      WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  Tensor a(DataTypeImpl::GetType<float>(), TensorShape{M, K}, a_buffer.Get(), memory_info);
  Tensor b(DataTypeImpl::GetType<float>(), TensorShape{K, N}, b_buffer.Get(), memory_info);
  Tensor output(DataTypeImpl::GetType<float>(), TensorShape{M, N}, output_buffer.Get(), memory_info);

  const TensorShape outer_dims{};
  const TensorShape a_program_shape = CreateMatMulIntermediateShape(outer_dims, M, K, 4);
  const TensorShape b_program_shape = CreateMatMulIntermediateShape(outer_dims, K, N, 4);
  const TensorShape output_program_shape{1, M, N / 4};
  InlinedVector<int64_t> elements_per_thread{4, 2, 1};
  const Activation activation;
  MatMulProgram program{activation, /*bias=*/false, /*is_vec4=*/true,
                        elements_per_thread, /*is_channels_last=*/true,
                        /*split_dim_inner=*/1, /*tile_inner=*/64};
  program.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank, a_program_shape, 4},
                     {&b, ProgramTensorMetadataDependency::TypeAndRank, b_program_shape, 4}})
      .AddUniformVariables({{M}, {N}, {K}, {1}, {1}, {1}, {1}})
      .AddIndices(outer_dims)
      .SetDispatchGroupSize(1, 1, 1)
      .SetWorkgroupSize(16, 4, 1)
      .AddOutput(ProgramOutput(&output, ProgramTensorMetadataDependency::Rank,
                               output_program_shape, 4));
  AppendActivationUniformsData(activation, program);

  RunAndReadProgram(Ep(), program, output_buffer.Get(), output_data);
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float expected = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        expected += a_data[m * K + k] * b_data[k * N + n];
      }
      EXPECT_NEAR(output_data[m * N + n], expected, 1e-5f)
          << "at output (" << m << ", " << n << ")";
    }
  }
}

constexpr float kOutputPaddingCanary = -12345.0f;

void CheckResultsAndPadding(gsl::span<const float> output_data, size_t output_size, float expected) {
  ASSERT_LE(output_size, output_data.size());
  EXPECT_TRUE(std::all_of(output_data.begin(), output_data.begin() + output_size,
                          [expected](float value) { return value == expected; }));
  EXPECT_TRUE(std::all_of(output_data.begin() + output_size, output_data.end(),
                          [](float value) { return value == kOutputPaddingCanary; }))
      << "Normalization-added workgroups modified output padding.";
}

class LogicalDispatchProbeProgram final : public Program<LogicalDispatchProbeProgram> {
 public:
  LogicalDispatchProbeProgram() : Program{"LogicalDispatchProbe"} {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LogicalDispatchProbeProgram);

  static constexpr std::array<uint32_t, 3> kWorkgroupSize{4, 2, 2};
  static constexpr uint32_t kValuesPerWorkgroup = 6;

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"logical_dispatch_x", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_y", ProgramUniformVariableDataType::Uint32},
      {"logical_dispatch_z", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    InitializeLogicalWorkgroupIDAndGlobalID(shader);
    // Padded groups must return uniformly before this barrier.
    shader.MainFunctionBody() << R"(
  workgroupBarrier();
  if (all(local_id == vec3u(workgroup_size_x - 1u, workgroup_size_y - 1u, workgroup_size_z - 1u))) {
    let offset = workgroup_idx * 6u;
)"
                              << output.SetByOffset("offset", "f32(logical_workgroup_id.x)") << "\n"
                              << output.SetByOffset("offset + 1u", "f32(logical_workgroup_id.y)") << "\n"
                              << output.SetByOffset("offset + 2u", "f32(logical_workgroup_id.z)") << "\n"
                              << output.SetByOffset("offset + 3u", "f32(logical_global_id.x)") << "\n"
                              << output.SetByOffset("offset + 4u", "f32(logical_global_id.y)") << "\n"
                              << output.SetByOffset("offset + 5u", "f32(logical_global_id.z)") << "\n"
                              << "  }\n";
    return Status::OK();
  }
};

void CheckLogicalDispatch(WebGpuExecutionProvider& ep, const std::array<uint32_t, 3>& logical,
                          uint32_t dispatch_limit, bool expect_logical_oob_writes = false) {
  SCOPED_TRACE(::testing::PrintToString(logical));
  auto physical = logical;
  ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(
      physical[0], physical[1], physical[2], dispatch_limit));
  const uint32_t logical_count = logical[0] * logical[1] * logical[2];
  const uint32_t physical_count = physical[0] * physical[1] * physical[2];
  ASSERT_GE(physical_count, logical_count);
  if (expect_logical_oob_writes) {
    ASSERT_GT(physical_count, logical_count);
  }
  const uint32_t allowed_dispatch_z = expect_logical_oob_writes ? physical_count : logical[2];
  constexpr auto workgroup_size = LogicalDispatchProbeProgram::kWorkgroupSize;
  constexpr uint32_t values_per_workgroup = LogicalDispatchProbeProgram::kValuesPerWorkgroup;
  const size_t output_size = static_cast<size_t>(logical_count) * values_per_workgroup;
  // Even without the guard, every physical workgroup has a valid output slot.
  std::vector<float> output_data(static_cast<size_t>(physical_count) * values_per_workgroup, kOutputPaddingCanary);
  auto output_buffer = CreateStorageBuffer(ep, output_data);
  const OrtMemoryInfo memory_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  Tensor output(DataTypeImpl::GetType<float>(), {static_cast<int64_t>(output_size)}, output_buffer.Get(), memory_info);
  LogicalDispatchProbeProgram program;
  program.AddOutput({&output, ProgramTensorMetadataDependency::TypeAndRank})
      .AddUniformVariables({{logical[0]}, {logical[1]}, {allowed_dispatch_z}})
      .SetWorkgroupSize(workgroup_size[0], workgroup_size[1], workgroup_size[2])
      .SetDispatchGroupSize(physical[0], physical[1], physical[2]);
  ASSERT_NO_FATAL_FAILURE(RunAndReadProgram(ep, program, output_buffer.Get(), output_data));

  size_t offset = 0;
  for (uint32_t z = 0; z < logical[2]; ++z) {
    for (uint32_t y = 0; y < logical[1]; ++y) {
      for (uint32_t x = 0; x < logical[0]; ++x) {
        const std::array<uint32_t, 3> coordinates{x, y, z};
        for (size_t axis = 0; axis < coordinates.size(); ++axis) {
          ASSERT_EQ(output_data[offset + axis], static_cast<float>(coordinates[axis]));
          ASSERT_EQ(output_data[offset + 3 + axis],
                    static_cast<float>(coordinates[axis] * workgroup_size[axis] + workgroup_size[axis] - 1));
        }
        offset += values_per_workgroup;
      }
    }
  }
  if (expect_logical_oob_writes) {
    EXPECT_TRUE(std::any_of(output_data.begin() + output_size, output_data.end(),
                            [](float value) { return value != kOutputPaddingCanary; }))
        << "The canary must detect writes by excess workgroups.";
  } else {
    EXPECT_TRUE(std::all_of(output_data.begin() + output_size, output_data.end(),
                            [](float value) { return value == kOutputPaddingCanary; }))
        << "Normalization-added workgroups modified output padding.";
  }
}

TEST_F(WebGpuDispatchExecutionTest, LogicalCoordinatesAndPadding) {
  struct Case {
    std::array<uint32_t, 3> logical;
    uint32_t limit;
  };
  const Case cases[] = {
      {{1, 1, 1}, 5},
      {{3, 2, 2}, 5},
      {{7, 1, 1}, 5},
      {{1, 7, 1}, 5},
      {{1, 1, 7}, 5},
      {{7, 3, 2}, 5},
      {{1, 1, 65535}, 65535},
      {{1, 1, 65536}, 65535},
      {{1, 1, 70000}, 65535},
  };
  for (const auto& test_case : cases) {
    ASSERT_NO_FATAL_FAILURE(CheckLogicalDispatch(Ep(), test_case.logical, test_case.limit));
  }
}

TEST_F(WebGpuDispatchExecutionTest, CanaryDetectsExcessWorkgroups) {
  // Trigger logical OOB writes into physically valid padding to verify canary detection.
  ASSERT_NO_FATAL_FAILURE(CheckLogicalDispatch(Ep(), {1, 1, 7}, 5, true));
}

TEST_F(WebGpuDispatchExecutionTest, Conv2dMMPreservesPaddingInBothLayouts) {
  const OrtMemoryInfo memory_info(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault);
  constexpr uint32_t batches = 7;
  std::array<uint32_t, 3> physical{1, 1, batches};
  ASSERT_STATUS_OK(webgpu::detail::NormalizeDispatchGroupSize(physical[0], physical[1], physical[2], 5));
  const uint32_t padded_batches = physical[0] * physical[1] * physical[2];
  std::vector<float> x_data(padded_batches * 8 * 3 * 3, 1.0f);
  std::vector<float> w_data(8 * 8 * 2 * 2, 1.0f);
  auto x_buffer = CreateStorageBuffer(Ep(), x_data);
  auto w_buffer = CreateStorageBuffer(Ep(), w_data);
  const Activation activation;
  for (bool channels_last : {false, true}) {
    SCOPED_TRACE(channels_last);
    std::vector<float> output_data(padded_batches * 8 * 2 * 2, kOutputPaddingCanary);
    auto output_buffer = CreateStorageBuffer(Ep(), output_data);
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
    ASSERT_NO_FATAL_FAILURE(RunAndReadProgram(Ep(), program, output_buffer.Get(), output_data));
    CheckResultsAndPadding(output_data, batches * 8 * 2 * 2, 32.0f);
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
