// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/linear_attention_gates.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    LinearAttentionGate,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TF", DataTypeImpl::GetTensorType<float>()),
    LinearAttentionGate);

ONNX_OPERATOR_KERNEL_EX(
    GatedRMSNorm,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    GatedRMSNorm);

Status LinearAttentionGateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& a = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& dt_bias = shader.AddInput("dt_bias", ShaderUsage::UseUniform);
  const auto& decay_scale = shader.AddInput("decay_scale", ShaderUsage::UseUniform);
  const ShaderVariableHelper* b = nullptr;
  if (has_b_) {
    b = &shader.AddInput("b", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& decay = shader.AddOutput("decay", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* beta = nullptr;
  if (has_beta_) {
    beta = &shader.AddOutput("beta", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }

  shader.AdditionalImplementation()
      << "fn la_softplus(x: f32) -> f32 {\n"
      << "  if (x > 0.0) {\n"
      << "    return x + log(exp(-x) + 1.0);\n"
      << "  }\n"
      << "  return log(exp(x) + 1.0);\n"
      << "}\n"
      << "fn la_sigmoid(x: f32) -> f32 {\n"
      << "  if (x > 0.0) {\n"
      << "    return 1.0 / (1.0 + exp(-x));\n"
      << "  }\n"
      << "  let e = exp(x);\n"
      << "  return e / (1.0 + e);\n"
      << "}\n";

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let h = global_idx % uniforms.num_heads;\n"
                            << "  let biased = f32(" << a.GetByOffset("global_idx") << ") + f32("
                            << dt_bias.GetByOffset("h") << ");\n"
                            << "  " << decay.SetByOffset("global_idx", "decay_element_t(f32(" + decay_scale.GetByOffset("h") + ") * la_softplus(biased))")
                            << "\n";
  if (has_beta_) {
    shader.MainFunctionBody() << "  "
                              << beta->SetByOffset("global_idx",
                                                   "beta_element_t(la_sigmoid(f32(" + b->GetByOffset("global_idx") + ")))")
                              << "\n";
  }

  return Status::OK();
}

Status LinearAttentionGate::ComputeInternal(ComputeContext& context) const {
  const auto* a = context.Input(0);
  const auto* dt_bias = context.Input(1);
  const auto* decay_scale = context.Input(2);
  const auto* b = context.Input(3);  // optional

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "a must have rank >= 1");
  const int64_t num_heads = a_shape[a_shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(num_heads > 0, "a last dimension must be positive");

  ORT_RETURN_IF_NOT(dt_bias->Shape().Size() == num_heads,
                    "dt_bias must have ", num_heads, " elements, got ", dt_bias->Shape().Size());
  ORT_RETURN_IF_NOT(decay_scale->Shape().Size() == num_heads,
                    "decay_scale must have ", num_heads, " elements, got ", decay_scale->Shape().Size());

  auto* decay = context.Output(0, a_shape);
  auto* beta = context.Output(1, a_shape);

  if (beta != nullptr) {
    ORT_RETURN_IF_NOT(b != nullptr, "The b input is required when the beta output is requested");
    ORT_RETURN_IF_NOT(b->Shape() == a_shape, "b must have the same shape as a");
  }

  const int64_t output_size = a_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

  LinearAttentionGateProgram program{b != nullptr, beta != nullptr};
  program.CacheHint(b != nullptr, beta != nullptr)
      .AddInputs({{a, ProgramTensorMetadataDependency::Type},
                  {dt_bias, ProgramTensorMetadataDependency::None},
                  {decay_scale, ProgramTensorMetadataDependency::None}});
  if (b != nullptr) {
    program.AddInput({b, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({decay, ProgramTensorMetadataDependency::None});
  if (beta != nullptr) {
    program.AddOutput({beta, ProgramTensorMetadataDependency::None});
  }
  program.SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(output_size) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(output_size)},
                            {onnxruntime::narrow<uint32_t>(num_heads)}});

  return context.RunProgram(program);
}

Status GatedRMSNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& scale = shader.AddInput("scale", ShaderUsage::UseUniform);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "fn stable_sigmoid(x: f32) -> f32 {\n"
      << "  if (x > 0.0) {\n"
      << "    return 1.0 / (1.0 + exp(-x));\n"
      << "  }\n"
      << "  let e = exp(x);\n"
      << "  return e / (1.0 + e);\n"
      << "}\n"
      << "var<workgroup> row_sum_sq : array<f32, workgroup_size_x>;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  let base = row * uniforms.norm_size;\n"
      << "  var sum_sq = 0.0;\n"
      << "  for (var i = local_idx; i < uniforms.norm_size; i += workgroup_size_x) {\n"
      << "    let v = f32(" << input.GetByOffset("base + i") << ");\n"
      << "    sum_sq += v * v;\n"
      << "  }\n"
      << "  row_sum_sq[local_idx] = sum_sq;\n"
      << "  workgroupBarrier();\n"
      << "  var reduce_size = workgroup_size_x;\n"
      << "  for (var curr_size = reduce_size >> 1u; curr_size > 0u; curr_size = reduce_size >> 1u) {\n"
      << "    reduce_size = curr_size + (reduce_size & 1u);\n"
      << "    if (local_idx < curr_size) {\n"
      << "      row_sum_sq[local_idx] += row_sum_sq[local_idx + reduce_size];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  let inv_rms = inverseSqrt(row_sum_sq[0] / f32(uniforms.norm_size) + uniforms.epsilon);\n"
      << "  for (var i = local_idx; i < uniforms.norm_size; i += workgroup_size_x) {\n"
      << "    let z = f32(" << gate.GetByOffset("base + i") << ");\n"
      << "    let normalized = f32(" << input.GetByOffset("base + i") << ") * inv_rms * f32("
      << scale.GetByOffset("i") << ");\n"
      << "    " << output.SetByOffset("base + i", "output_element_t(normalized * (z * stable_sigmoid(z)))") << "\n"
      << "  }\n";

  return Status::OK();
}

GatedRMSNorm::GatedRMSNorm(const OpKernelInfo& info) : WebGpuKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-5f);
}

Status GatedRMSNorm::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* scale = context.Input(1);
  const auto* gate = context.Input(2);

  const auto& shape = input->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() >= 1, "X must have rank >= 1");
  ORT_RETURN_IF_NOT(gate->Shape() == shape, "gate must have the same shape as X");

  const int64_t norm_size = scale->Shape().Size();
  ORT_RETURN_IF_NOT(norm_size > 0, "scale must not be empty");
  const int64_t last_dim = shape[shape.NumDimensions() - 1];
  ORT_RETURN_IF_NOT(last_dim % norm_size == 0,
                    "X last dimension (", last_dim, ") must be a multiple of the scale length (",
                    norm_size, ")");

  auto* output = context.Output(0, shape);
  const int64_t total_size = shape.Size();
  if (total_size == 0) {
    return Status::OK();
  }
  const int64_t num_rows = total_size / norm_size;

  const uint32_t workgroup_size = norm_size <= 64    ? 64
                                  : norm_size <= 128 ? 128
                                                     : 256;

  GatedRMSNormProgram program{};
  program.AddInputs({{input, ProgramTensorMetadataDependency::Type},
                     {scale, ProgramTensorMetadataDependency::Type},
                     {gate, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(num_rows))
      .SetWorkgroupSize(workgroup_size)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(norm_size)},
                            {epsilon_}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
