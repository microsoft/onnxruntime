// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>
#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "core/framework/customregistry.h"
#include "core/providers/webgpu/math/softmax.h"
#include "core/providers/webgpu/math/top_k.h"
#include "core/providers/webgpu/nn/lp_norm.h"
#include "core/providers/webgpu/rnn/gru.h"
#include "core/providers/webgpu/rnn/lstm.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/tensor/where.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#endif

namespace onnxruntime::test {
namespace {

using namespace webgpu;
using ComputeFn = std::function<Status(webgpu::ComputeContext&)>;

class BufferAccessKernel final : public WebGpuKernel {
 public:
  BufferAccessKernel(const OpKernelInfo& info, ComputeFn compute) : WebGpuKernel(info), compute_(std::move(compute)) {}
  Status ComputeInternal(webgpu::ComputeContext& context) const override { return compute_(context); }

 private:
  ComputeFn compute_;
};

// A session-local kernel dispatches production programs with explicit buffer views.
// Keeping the surrounding padding in the result detects misplaced writes.
void RunViews(const std::vector<float>& input, const std::vector<float>& expected, ComputeFn compute,
              bool segmented = false, std::string_view expected_error = {}) {
  bool executed = false;
  auto registry = std::make_shared<CustomRegistry>();
  constexpr const char* domain = "test.webgpu";
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("BufferAccess").SetDomain(domain).SinceVersion(1).Input(0, "X", "Backing input storage", "T").Output(0, "Y", "Backing output storage", "T").TypeConstraint("T", {"tensor(float)"}, "Float storage");
  std::vector<ONNX_NAMESPACE::OpSchema> schemas{schema};
  ASSERT_STATUS_OK(registry->RegisterOpSet(schemas, domain, 1, 2));
  KernelDefBuilder def;
  def.SetName("BufferAccess").SetDomain(domain).SinceVersion(1).Provider(kWebGpuExecutionProvider).TypeConstraint("T", DataTypeImpl::GetTensorType<float>());
  ASSERT_STATUS_OK(registry->RegisterCustomKernel(
      def, [&](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& kernel) {
        kernel = std::make_unique<BufferAccessKernel>(info, [&](webgpu::ComputeContext& context) {
          executed = true;
          return compute(context);
        });
        return Status::OK();
      }));
  OpTester test("BufferAccess", 1, domain);
  test.AddCustomOpRegistry(registry);
  test.AddInput<float>("X", {static_cast<int64_t>(input.size())}, input);
  test.AddOutput<float>("Y", {static_cast<int64_t>(expected.size())}, expected);
  ConfigOptions config;
  auto bootstrap = DefaultWebGpuExecutionProvider();
  ASSERT_NE(bootstrap, nullptr);
  const auto& device_context = WebGpuContextFactory::GetContext(0);
  wgpu::Device validation_device;
  if (!expected_error.empty()) {
    // The shared Release device may skip Dawn validation. Use a separate device
    // with default validation to check diagnostics for deliberately invalid WGSL.
    wgpu::Adapter adapter;
    ASSERT_EQ(device_context.Instance().WaitAny(device_context.Instance().RequestAdapter(
                                                    nullptr, wgpu::CallbackMode::WaitAnyOnly,
                                                    [](wgpu::RequestAdapterStatus status, wgpu::Adapter result, wgpu::StringView message,
                                                       wgpu::Adapter* adapter) noexcept {
                                                      EXPECT_EQ(status, wgpu::RequestAdapterStatus::Success) << std::string_view(message);
                                                      *adapter = std::move(result);
                                                    },
                                                    &adapter),
                                                UINT64_MAX),
              wgpu::WaitStatus::Success);
    ASSERT_NE(adapter, nullptr);
    wgpu::DeviceDescriptor descriptor{};
    ASSERT_EQ(device_context.Instance().WaitAny(adapter.RequestDevice(
                                                    &descriptor, wgpu::CallbackMode::WaitAnyOnly,
                                                    [](wgpu::RequestDeviceStatus status, wgpu::Device result, wgpu::StringView message,
                                                       wgpu::Device* device) noexcept {
                                                      EXPECT_EQ(status, wgpu::RequestDeviceStatus::Success) << std::string_view(message);
                                                      *device = std::move(result);
                                                    },
                                                    &validation_device),
                                                UINT64_MAX),
              wgpu::WaitStatus::Success);
    ASSERT_NE(validation_device, nullptr);
  }
  ASSERT_STATUS_OK(config.AddConfigEntry(options::kDeviceId,
                                         validation_device ? (segmented ? "329283" : "329282") : (segmented ? "329281" : "329280")));
  ASSERT_STATUS_OK(config.AddConfigEntry(options::kValidationMode, options::kValidationMode_full));
  ASSERT_STATUS_OK(config.AddConfigEntry(options::kWebGpuInstance,
                                         std::to_string(reinterpret_cast<uintptr_t>(device_context.Instance().Get())).c_str()));
  ASSERT_STATUS_OK(config.AddConfigEntry(options::kWebGpuDevice,
                                         std::to_string(reinterpret_cast<uintptr_t>(validation_device ? validation_device.Get() : device_context.Device().Get())).c_str()));
  ASSERT_STATUS_OK(config.AddConfigEntry(options::kPreserveDevice, "0"));
  auto factory = segmented ? WebGpuProviderFactoryCreator::CreateForTesting(config, 256)
                           : WebGpuProviderFactoryCreator::Create(config);
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  InferenceSession session(options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterCustomRegistry(registry));
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(factory->CreateProvider()));
  auto& model = test.BuildModel();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());
  const auto serialized = model.ToProto().SerializeAsString();
  ASSERT_STATUS_OK(session.Load(serialized.data(), static_cast<int>(serialized.size())));
  ASSERT_STATUS_OK(session.Initialize());
  OrtValue input_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(input.size())},
                       std::make_shared<CPUAllocator>(), input_value);
  std::copy(input.begin(), input.end(), input_value.GetMutable<Tensor>()->MutableData<float>());
  std::vector<OrtValue> outputs;
  const auto status = session.Run(RunOptions{}, std::vector<std::string>{"X"}, std::vector<OrtValue>{input_value},
                                  std::vector<std::string>{"Y"}, &outputs);
  EXPECT_TRUE(executed);
  if (!expected_error.empty()) {
    ASSERT_FALSE(status.IsOK());
    EXPECT_NE(status.ErrorMessage().find(expected_error), std::string::npos) << status.ErrorMessage();
    return;
  }
  ASSERT_STATUS_OK(status);
  ASSERT_EQ(outputs.size(), 1u);
  const auto& actual = outputs[0].Get<Tensor>();
  ASSERT_EQ(actual.Shape().Size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual.Data<float>()[i], expected[i], 1e-5f) << "element " << i;
  }
}

template <typename TProgram>
Status RunUnaryView(webgpu::ComputeContext& context, TProgram& program, uint32_t input_offset,
                    uint32_t output_offset, uint32_t count, int components = 1) {
  auto* output = context.Output(0, context.Input(0)->Shape());
  context.FillZero(*output);
  const TensorShape shape{count / components};
  program.AddInput(ProgramInput::BufferView(context.Input(0), ProgramTensorMetadataDependency::TypeAndRank,
                                            shape, input_offset, components))
      .AddOutput(ProgramOutput::BufferView(output, ProgramTensorMetadataDependency::TypeAndRank,
                                           shape, output_offset, components));
  return context.RunProgram(program);
}

class WebGpuBufferAccessTest : public testing::TestWithParam<bool> {};

enum class StorageAccess { Helper,
                           RawRead,
                           RawWrite,
                           RawPointer };

class StorageAccessProgram final : public Program<StorageAccessProgram> {
 public:
  explicit StorageAccessProgram(StorageAccess access) : Program{"StorageAccess"}, access_{access} {
    CacheHint(static_cast<int>(access));
  }

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::None);
    const auto& output = shader.AddOutput("output", ShaderUsage::None);
    switch (access_) {
      case StorageAccess::RawRead:
        shader.MainFunctionBody() << output.SetByOffset("0u", "input[0u]");
        break;
      case StorageAccess::RawWrite:
        shader.MainFunctionBody() << "output[0u] = " << input.GetByOffset("0u") << ";\n";
        break;
      case StorageAccess::RawPointer:
        shader.MainFunctionBody() << "let pointer = &input;\n"
                                  << output.SetByOffset("0u", "(*pointer)[0u]");
        break;
      case StorageAccess::Helper:
        shader.AdditionalImplementation() << "var<workgroup> tile: array<f32, 1>;\n";
        shader.MainFunctionBody() << "var values: array<f32, 1>;\n"
                                  << "values[0] = " << input.GetByOffset("0u") << ";\n"
                                  << "tile[0] = values[0];\n"
                                  << output.SetByOffset("0u", "tile[0]");
        break;
    }
    return Status::OK();
  }

 private:
  StorageAccess access_;
};

TEST_P(WebGpuBufferAccessTest, RawStorageAccessFailsShaderCompilation) {
  for (const auto access : {StorageAccess::RawRead, StorageAccess::RawWrite, StorageAccess::RawPointer}) {
    SCOPED_TRACE(static_cast<int>(access));
    RunViews({2.f}, {2.f}, [=](webgpu::ComputeContext& context) {
      StorageAccessProgram program(access);
      program.SetWorkgroupSize(1).SetDispatchGroupSize(1);
      return RunUnaryView(context, program, 0, 0, 1); }, GetParam(), access == StorageAccess::RawWrite ? "unresolved value 'output'" : "unresolved value 'input'");
  }
}

TEST_P(WebGpuBufferAccessTest, HelpersAllowLocalAndWorkgroupIndexing) {
  RunViews({2.f}, {2.f}, [](webgpu::ComputeContext& context) {
    StorageAccessProgram program(StorageAccess::Helper);
    program.SetWorkgroupSize(1).SetDispatchGroupSize(1);
    return RunUnaryView(context, program, 0, 0, 1); }, GetParam());
}

TEST_P(WebGpuBufferAccessTest, RecurrentStateCopiesUseViewOffsets) {
  for (bool lstm : {false, true}) {
    SCOPED_TRACE(lstm);
    std::vector<float> input(128, -100.f), expected(128, 0.f);
    std::iota(input.begin() + 60, input.begin() + 68, 1.f);
    std::copy_n(input.begin() + 60, 8, expected.begin() + 63);
    RunViews(input, expected, [=](webgpu::ComputeContext& context) {
      if (lstm) {
        LstmStateCopyProgram program(true, 0);
        program.SetDispatchGroupSize(1).AddUniformVariables({1u, 8u, 0u, 1u});
        return RunUnaryView(context, program, 60, 63, 8);
      }
      GruStateCopyProgram program(true, 0);
      program.SetDispatchGroupSize(1).AddUniformVariables({1u, 8u, 0u, 1u});
      return RunUnaryView(context, program, 60, 63, 8); }, GetParam());
  }
}

TEST_P(WebGpuBufferAccessTest, SoftmaxUsesScalarAndVectorViews) {
  for (int components : {1, 4}) {
    SCOPED_TRACE(components);
    std::vector<float> input(128, -100.f), expected(128, 0.f);
    const uint32_t input_offset = components == 1 ? 60 : 15;
    const uint32_t output_offset = components == 1 ? 63 : 14;
    std::fill_n(input.begin() + input_offset * components, 8, 2.f);
    std::fill_n(expected.begin() + output_offset * components, 8, 0.125f);
    RunViews(input, expected, [=](webgpu::ComputeContext& context) {
      SoftmaxProgram program(64, true);
      program.SetWorkgroupSize(64).SetDispatchGroupSize(1).AddUniformVariables({int32_t{8 / components}});
      return RunUnaryView(context, program, input_offset, output_offset, 8, components); }, GetParam());
  }
}

TEST_P(WebGpuBufferAccessTest, GruGatesReadSharedInputsAndWriteSharedOutputs) {
  std::vector<float> input(128, 0.f), expected(128, 0.f);
  input[60] = 2.f;
  input[61] = 4.f;
  expected[63] = expected[64] = 0.5f;
  expected[70] = 1.f;
  expected[71] = 2.f;
  RunViews(input, expected, [](webgpu::ComputeContext& context) {
    auto* output = context.Output(0, context.Input(0)->Shape());
    context.FillZero(*output);
    GruGateProgram program(true, false, false, 0, "sigmoid_f");
    for (const auto& [offset, count] : {std::pair{2u, 2u}, {10u, 12u}, {28u, 12u}, {60u, 2u}, {70u, 12u}}) {
      program.AddInput(ProgramInput::BufferView(context.Input(0), ProgramTensorMetadataDependency::TypeAndRank,
                                                TensorShape{count}, offset));
    }
    for (uint32_t offset : {63u, 70u}) {
      program.AddOutput(ProgramOutput::BufferView(output, ProgramTensorMetadataDependency::TypeAndRank,
                                                  TensorShape{2}, offset));
    }
    program.SetDispatchGroupSize(1).AddUniformVariables({1u, 2u, 2u, 0u, 1u, 0u, 1u, 0.f});
    return context.RunProgram(program); }, GetParam());
}

TEST_P(WebGpuBufferAccessTest, LstmCellReadsSharedOptionalInputs) {
  std::vector<float> input(128, 0.f), expected(128, 0.f);
  input[60] = input[61] = 2.f;
  expected[63] = expected[64] = 0.5f * std::tanh(1.f);
  expected[70] = expected[71] = 1.f;
  expected[80] = expected[81] = expected[63];
  RunViews(input, expected, [](webgpu::ComputeContext& context) {
    auto* output = context.Output(0, context.Input(0)->Shape());
    context.FillZero(*output);
    LstmCellProgram program(true, true, true, false, false, false, 0, "sigmoid_f", "tanh_f", "tanh_f");
    for (const auto& [offset, count] : {std::pair{2u, 2u}, {10u, 16u}, {28u, 16u}, {50u, 2u}, {60u, 2u}, {70u, 16u}, {90u, 6u}}) {
      program.AddInput(ProgramInput::BufferView(context.Input(0), ProgramTensorMetadataDependency::TypeAndRank,
                                                TensorShape{count}, offset));
    }
    for (uint32_t offset : {63u, 70u, 80u}) {
      program.AddOutput(ProgramOutput::BufferView(output, ProgramTensorMetadataDependency::TypeAndRank,
                                                  TensorShape{2}, offset));
    }
    program.SetDispatchGroupSize(1).AddUniformVariables({1u, 2u, 2u, 0u, 1u, 0u, 1u, 0.f});
    return context.RunProgram(program); }, GetParam());
}

TEST_P(WebGpuBufferAccessTest, LpNormalizationUsesViewOffsets) {
  std::vector<float> input(128, -100.f), expected(128, 0.f);
  std::fill_n(input.begin() + 60, 8, 2.f);
  std::fill_n(expected.begin() + 63, 8, 0.125f);
  RunViews(input, expected, [](webgpu::ComputeContext& context) {
    LpNormProgram program(1);
    program.SetWorkgroupSize(64).SetDispatchGroupSize(1).AddUniformVariables({1u, 8u, 1u});
    return RunUnaryView(context, program, 60, 63, 8); }, GetParam());
}

TEST_P(WebGpuBufferAccessTest, TopKReadsAndWritesOutputViews) {
  std::vector<float> input(128, -100.f), expected(128, -100.f);
  input[63] = 1.f;
  input[64] = 3.f;
  expected[63] = 3.f;
  expected[64] = 1.f;
  RunViews(input, expected, [](webgpu::ComputeContext& context) {
    auto* output = context.Output(0, context.Input(0)->Shape());
    ORT_RETURN_IF_ERROR(context.CopyTensor(*context.Input(0), *output));
    auto indices = context.CreateGPUTensor(DataTypeImpl::GetType<int32_t>(), TensorShape{128});
    context.FillZero(indices);
    TopKSortStepProgram program(true);
    program.AddOutput(ProgramOutput::BufferView(output, ProgramTensorMetadataDependency::TypeAndRank,
                                                TensorShape{2}, 63))
        .AddOutput(ProgramOutput::BufferView(&indices, ProgramTensorMetadataDependency::TypeAndRank,
                                              TensorShape{2}, 63))
        .SetDispatchGroupSize(1).AddUniformVariables({2u, 1u, 2u, 1u});
    return context.RunProgram(program); }, GetParam());
}

TEST_P(WebGpuBufferAccessTest, WhereBroadcastsPackedBooleansAndSharedInputViews) {
  std::vector<float> input(128, -100.f), expected(128, 0.f);
  std::iota(input.begin() + 60, input.begin() + 68, 1.f);
  for (size_t i = 0; i < 8; ++i) expected[60 + i] = i % 2 == 0 ? static_cast<float>(i + 1) : -100.f;
  RunViews(input, expected, [](webgpu::ComputeContext& context) {
    auto cpu_allocator = std::make_shared<CPUAllocator>();
    Tensor condition_cpu(DataTypeImpl::GetType<bool>(), TensorShape{512}, cpu_allocator);
    std::fill_n(condition_cpu.MutableData<bool>(), 512, false);
    for (size_t i = 0; i < 8; i += 2) condition_cpu.MutableData<bool>()[252 + i] = true;
    auto condition = context.CreateGPUTensor(DataTypeImpl::GetType<bool>(), TensorShape{512});
    ORT_RETURN_IF_ERROR(context.CopyTensor(condition_cpu, condition));
    auto* output = context.Output(0, context.Input(0)->Shape());
    context.FillZero(*output);
    WhereProgram program(true);
    program.AddInput(ProgramInput::BufferView(&condition, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{2}, 63, 4))
        .AddInput(ProgramInput::BufferView(context.Input(0), ProgramTensorMetadataDependency::TypeAndRank, TensorShape{2}, 15, 4))
        .AddInput(ProgramInput::BufferView(context.Input(0), ProgramTensorMetadataDependency::TypeAndRank, TensorShape{1}, 1, 4))
        .AddOutput(ProgramOutput::BufferView(output, ProgramTensorMetadataDependency::TypeAndRank, TensorShape{2}, 15, 4))
        .AddIndices(TensorShape{8}).AddIndices(TensorShape{8}).AddIndices(TensorShape{1}).AddIndices(TensorShape{8})
        .SetDispatchGroupSize(1).AddUniformVariables({2u});
    return context.RunProgram(program); }, GetParam());
}

INSTANTIATE_TEST_SUITE_P(Storage, WebGpuBufferAccessTest, testing::Bool());

#ifndef DISABLE_CONTRIB_OPS
TEST(WebGpuBufferBindingTest, MatrixVariantsReceiveSharedZeroPointHelper) {
  auto bootstrap = DefaultWebGpuExecutionProvider();
  ASSERT_NE(bootstrap, nullptr);
  const auto& context = WebGpuContextFactory::GetContext(0);
  auto allocator = std::make_shared<CPUAllocator>();
  Tensor a(DataTypeImpl::GetType<MLFloat16>(), TensorShape{1, 128, 128}, allocator);
  Tensor b(DataTypeImpl::GetType<uint8_t>(), TensorShape{128, 4, 16}, allocator);
  Tensor scales(DataTypeImpl::GetType<MLFloat16>(), TensorShape{128, 4}, allocator);
  Tensor zero_points(DataTypeImpl::GetType<uint8_t>(), TensorShape{128, 2}, allocator);
  Tensor output(DataTypeImpl::GetType<MLFloat16>(), TensorShape{1, 128, 128}, allocator);
  constexpr auto f16 = wgpu::SubgroupMatrixComponentType::F16;
  const std::array configs{
      SubgroupMatrixConfig{f16, f16, 8, 8, 8, 32, false},
      SubgroupMatrixConfig{f16, f16, 8, 16, 16, 32, false},
      SubgroupMatrixConfig{f16, f16, 16, 16, 16, 32, true}};
  for (const auto& config : configs) {
    SCOPED_TRACE(MakeString(config.M, "x", config.N, "x", config.K));
    contrib::webgpu::SubgroupMatrixMatMulNBitsProgram program(4, config, true, false, false, false, false);
    program.AddInputs({{&a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                       {&b, ProgramTensorMetadataDependency::TypeAndRank, 4},
                       {&scales, ProgramTensorMetadataDependency::TypeAndRank, 1},
                       {&zero_points, ProgramTensorMetadataDependency::None, TensorShape{64}, 4}})
        .AddOutput({&output, ProgramTensorMetadataDependency::TypeAndRank, 1})
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(1)
        .AddUniformVariables({128u, 128u, 128u, 1u, 0u, 1u});
    std::array<uint32_t, 4> input_segments{1, 1, 1, 1};
    std::array<uint32_t, 1> output_segments{1};
    ShaderHelper shader(program, program.Metadata(), context, input_segments, output_segments, 1, 1, 1);
    ASSERT_STATUS_OK(shader.Init());
    ASSERT_STATUS_OK(program.GenerateShaderCode(shader));
    const auto source = std::move(shader.AdditionalImplementation()).str() + std::move(shader.MainFunctionBody()).str();
    EXPECT_NE(source.find("storage_zero_points["), std::string::npos);
  }
}
#endif

}  // namespace
}  // namespace onnxruntime::test
