// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_block_quantized_fp8_weight.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/common/narrow.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

namespace {

class Fp8ActivationProgram final : public Program<Fp8ActivationProgram> {
 public:
  Fp8ActivationProgram() : Program{"Fp8MatMulActivationQdq"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input_a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
    const auto& input_scale = shader.AddInput("input_scale", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_block_quantized_fp8_activation.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(input_a, input_a),
                               WGSL_TEMPLATE_VARIABLE(input_scale, input_scale),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};

class Fp8EmptyReductionProgram final : public Program<Fp8EmptyReductionProgram> {
 public:
  explicit Fp8EmptyReductionProgram(bool has_bias)
      : Program{"Fp8MatMulEmptyReduction"}, has_bias_(has_bias) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    if (has_bias_) {
      const auto& bias = shader.AddInput("bias");
      const auto& output = shader.AddOutput("output");
      shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                                << output.SetByOffset("global_idx", bias.GetByOffset("global_idx % uniforms.N"));
    } else {
      const auto& output = shader.AddOutput("output");
      shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.size")
                                << output.SetByOffset("global_idx", "f16(0)");
    }
    return Status::OK();
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
};

class Fp8MatMulProgram final : public Program<Fp8MatMulProgram> {
 public:
  Fp8MatMulProgram(bool has_bias, bool use_matrix)
      : Program{"MatMulBlockQuantizedFp8Weight"}, has_bias_(has_bias), use_matrix_(use_matrix) {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input_a = shader.AddInput("input_a", ShaderUsage::UseValueTypeAlias);
    const auto& weights = shader.AddInput("weights", ShaderUsage::UseValueTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseValueTypeAlias);
    const ShaderVariableHelper* bias = nullptr;
    if (has_bias_) {
      bias = &shader.AddInput("bias", ShaderUsage::UseValueTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "quantization/matmul_block_quantized_fp8_weight.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                               WGSL_TEMPLATE_PARAMETER(use_matrix, use_matrix_),
                               WGSL_TEMPLATE_OPTIONAL_VARIABLE(bias, bias),
                               WGSL_TEMPLATE_VARIABLE(input_a, input_a),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(scales, scales),
                               WGSL_TEMPLATE_VARIABLE(weights, weights));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"K_blocks", ProgramUniformVariableDataType::Uint32},
                                          {"block_size", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool use_matrix_;
};

}  // namespace

MatMulBlockQuantizedFp8Weight::MatMulBlockQuantizedFp8Weight(const OpKernelInfo& info)
    : WebGpuKernel(info),
      block_size_(narrow<uint32_t>(info.GetAttrOrDefault<int64_t>("block_size", 128))) {
  ORT_ENFORCE(block_size_ > 0, "block_size must be positive.");
}

Status MatMulBlockQuantizedFp8Weight::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* a = context.Input(0);
  const Tensor* b = context.Input(1);
  const Tensor* scales = context.Input(2);
  const Tensor* a_scale = context.Input(3);
  const Tensor* bias = context.Input(4);

  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1 && b_shape.NumDimensions() == 2,
                    "A must have rank >= 1 and B must have shape [N, K].");
  const int64_t n_dim = b_shape[0];
  const int64_t k_dim = b_shape[1];
  ORT_RETURN_IF_NOT(a_shape[a_shape.NumDimensions() - 1] == k_dim,
                    "A and B contraction dimensions must match.");
  const int64_t k_blocks = (k_dim + block_size_ - 1) / block_size_;
  ORT_RETURN_IF_NOT(scales->Shape() == TensorShape({n_dim, k_blocks}),
                    "b_scale must have shape [N, ceil(K/block_size)].");
  ORT_RETURN_IF_NOT(a_scale == nullptr || a_scale->Shape().Size() == 1,
                    "a_scale must contain one value.");
  ORT_RETURN_IF_NOT(bias == nullptr || bias->Shape() == TensorShape({n_dim}),
                    "bias must have shape [N].");

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b_shape, false, true));
  Tensor* y = context.Output(0, helper.OutputShape());
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }

  const uint32_t K = narrow<uint32_t>(k_dim);
  const uint32_t N = narrow<uint32_t>(n_dim);
  const uint32_t M = narrow<uint32_t>(y->Shape().Size() / n_dim);
  const uint32_t K_blocks = narrow<uint32_t>(k_blocks);
  if (K == 0) {
    Fp8EmptyReductionProgram empty_program{bias != nullptr};
    empty_program.SetWorkgroupSize(64);
    empty_program.SetDispatchGroupSize((narrow<uint32_t>(y->Shape().Size()) + 63u) / 64u);
    if (bias) {
      empty_program.AddInput({bias, ProgramTensorMetadataDependency::Type});
    }
    empty_program.AddOutput({y, ProgramTensorMetadataDependency::Type})
        .AddUniformVariables({{narrow<uint32_t>(y->Shape().Size())}, {N}});
    return context.RunProgram(empty_program);
  }
  const Tensor* activation = a;
  Tensor qdq;
  if (a_scale && K != 0) {
    qdq = context.CreateGPUTensor(a->DataType(), a_shape);
    Fp8ActivationProgram qdq_program;
    qdq_program.SetWorkgroupSize(64);
    qdq_program.SetDispatchGroupSize((narrow<uint32_t>(a_shape.Size()) + 63u) / 64u);
    qdq_program.AddInputs({{a, ProgramTensorMetadataDependency::Type},
                           {a_scale, ProgramTensorMetadataDependency::None}})
        .AddOutput({&qdq, ProgramTensorMetadataDependency::None})
        .AddUniformVariables({{narrow<uint32_t>(a_shape.Size())}});
    ORT_RETURN_IF_ERROR(context.RunProgram(qdq_program));
    activation = &qdq;
  }

  const bool use_matrix = M >= 8 && K != 0 &&
                          SelectSubgroupMatrixConfig(context, {{wgpu::SubgroupMatrixComponentType::F16,
                                                                wgpu::SubgroupMatrixComponentType::F32,
                                                                16, 16, 16, 32, false}})
                              .has_value();

  auto memory_info = OrtMemoryInfo{
      WEBGPU_BUFFER,
      OrtDeviceAllocator,
      OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};
  Tensor weight_bytes(DataTypeImpl::GetType<uint8_t>(), b_shape, const_cast<void*>(b->DataRaw()), memory_info);
  Fp8MatMulProgram program{bias != nullptr, use_matrix};
  program.SetWorkgroupSize(use_matrix ? 128 : 32);
  if (use_matrix && context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
    program.SetSubgroupSize(32);
  }
  program.SetDispatchGroupSize(use_matrix ? (N + 15u) / 16u : N,
                               use_matrix ? (M + 63u) / 64u : M);
  program.AddInputs({{activation, ProgramTensorMetadataDependency::Type},
                     {&weight_bytes, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4},
                     {scales, ProgramTensorMetadataDependency::Type}})
      .AddOutput({y, ProgramTensorMetadataDependency::Type})
      .AddUniformVariables({{M}, {N}, {K}, {K_blocks}, {block_size_}});
  if (bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type});
  }
  return context.RunProgram(program);
}

#if !defined(DISABLE_FLOAT8_TYPES)
ONNX_OPERATOR_KERNEL_EX(
    MatMulBlockQuantizedFp8Weight,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<Float8E4M3FN>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    MatMulBlockQuantizedFp8Weight);
#endif

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
