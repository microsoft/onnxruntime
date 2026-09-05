// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/engram_gate.h"

#include "contrib_ops/webgpu/bert/engram_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    EngramGate,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    EngramGate);

namespace {
constexpr uint32_t kGateWorkgroupSize = 64;
}  // namespace

Status EngramGateScalarProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& key_norm_scale = shader.AddInput("key_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& query_norm_scale = shader.AddInput("query_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& gate = shader.AddOutput("gate", ShaderUsage::UseUniform);

  const int components = key.NumComponents();

  shader.AdditionalImplementation()
      << engram_helper::kStableSigmoidWgsl << engram_helper::kEngramGateArgWgsl
      << "alias gate_f32_t = " << MakeScalarOrVectorType(components, "f32") << ";\n"
      << "var<workgroup> key_partials: array<f32, " << kGateWorkgroupSize << ">;\n"
      << "var<workgroup> query_partials: array<f32, " << kGateWorkgroupSize << ">;\n"
      << "var<workgroup> dot_partials: array<f32, " << kGateWorkgroupSize << ">;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let g = row % uniforms.hc_mult;\n"
      << "  let row_base = row * uniforms.hidden_vec_size;\n"
      << "  let scale_base = g * uniforms.hidden_vec_size;\n"
      << "  var key_sum_sq = 0.0;\n"
      << "  var query_sum_sq = 0.0;\n"
      << "  var dot_numerator = 0.0;\n"
      << "  for (var d = local_idx; d < uniforms.hidden_vec_size; d += " << kGateWorkgroupSize << "u) {\n"
      << "    let key_value = gate_f32_t(" << key.GetByOffset("row_base + d") << ");\n"
      << "    let query_value = gate_f32_t(" << query.GetByOffset("row_base + d") << ");\n"
      << "    let key_squared = key_value * key_value;\n"
      << "    let query_squared = query_value * query_value;\n"
      << "    let dot_terms = key_value * gate_f32_t(" << key_norm_scale.GetByOffset("scale_base + d")
      << ") * query_value * gate_f32_t(" << query_norm_scale.GetByOffset("scale_base + d") << ");\n"
      << "    key_sum_sq += " << SumVector("key_squared", components) << ";\n"
      << "    query_sum_sq += " << SumVector("query_squared", components) << ";\n"
      << "    dot_numerator += " << SumVector("dot_terms", components) << ";\n"
      << "  }\n"
      << "  key_partials[local_idx] = key_sum_sq;\n"
      << "  query_partials[local_idx] = query_sum_sq;\n"
      << "  dot_partials[local_idx] = dot_numerator;\n"
      << "  workgroupBarrier();\n"
      << "  for (var stride = " << (kGateWorkgroupSize / 2) << "u; stride > 0u; stride >>= 1u) {\n"
      << "    if (local_idx < stride) {\n"
      << "      key_partials[local_idx] += key_partials[local_idx + stride];\n"
      << "      query_partials[local_idx] += query_partials[local_idx + stride];\n"
      << "      dot_partials[local_idx] += dot_partials[local_idx + stride];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (local_idx == 0u) {\n"
      << "    let key_inv_rms = inverseSqrt(key_partials[0] / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "    let query_inv_rms = inverseSqrt(query_partials[0] / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "    let dot_value = dot_partials[0] * key_inv_rms * query_inv_rms / sqrt(f32(uniforms.hidden_size));\n"
      << "    " << gate.SetByOffset("row", "stable_sigmoid(engram_gate_arg(dot_value))") << "\n"
      << "  }\n";
  return Status::OK();
}

Status EngramGateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& value = shader.AddInput("value", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  const int components = value.NumComponents();

  shader.AdditionalImplementation() << "alias gate_f32_t = " << MakeScalarOrVectorType(components, "f32") << ";\n";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let c = global_idx % uniforms.hidden_vec_size;\n"
      << "  let row = global_idx / uniforms.hidden_vec_size;\n"
      << "  let token = row / uniforms.hc_mult;\n"
      << "  let value_element = gate_f32_t(" << value.GetByOffset("token * uniforms.hidden_vec_size + c") << ");\n"
      << "  " << output.SetByOffset("global_idx", "output_value_t(" + gate.GetByOffset("row") + " * value_element)")
      << "\n";
  return Status::OK();
}

Status EngramGateNormProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& gated_value = shader.AddInput("gated_value", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& conv_norm_scale = shader.AddInput("conv_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& gated_value_normed = shader.AddOutput("gated_value_normed", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "var<workgroup> sum_sq_partials: array<f32, " << kGateWorkgroupSize << ">;\n"
      << "var<workgroup> inv_rms: f32;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let g = row % uniforms.hc_mult;\n"
      << "  let row_base = row * uniforms.hidden_size;\n"
      << "  let scale_base = g * uniforms.hidden_size;\n"
      << "  var sum_sq = 0.0;\n"
      << "  for (var d = local_idx; d < uniforms.hidden_size; d += " << kGateWorkgroupSize << "u) {\n"
      << "    let value = f32(" << gated_value.GetByOffset("row_base + d") << ");\n"
      << "    sum_sq += value * value;\n"
      << "  }\n"
      << "  sum_sq_partials[local_idx] = sum_sq;\n"
      << "  workgroupBarrier();\n"
      << "  for (var stride = " << (kGateWorkgroupSize / 2) << "u; stride > 0u; stride >>= 1u) {\n"
      << "    if (local_idx < stride) {\n"
      << "      sum_sq_partials[local_idx] += sum_sq_partials[local_idx + stride];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "  if (local_idx == 0u) {\n"
      << "    inv_rms = inverseSqrt(sum_sq_partials[0] / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << "  for (var d = local_idx; d < uniforms.hidden_size; d += " << kGateWorkgroupSize << "u) {\n"
      << "    let value = f32(" << gated_value.GetByOffset("row_base + d") << ");\n"
      << "    " << gated_value_normed.SetByOffset("row_base + d", "gated_value_normed_element_t(value * inv_rms * f32(" + conv_norm_scale.GetByOffset("scale_base + d") + "))")
      << "\n"
      << "  }\n";
  return Status::OK();
}

EngramGate::EngramGate(const OpKernelInfo& info) : WebGpuKernel(info) {
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

Status EngramGate::ComputeInternal(ComputeContext& context) const {
  const auto* key = context.Input(0);
  const auto* query = context.Input(1);
  const auto* value = context.Input(2);
  const auto* key_norm_scale = context.Input(3);
  const auto* query_norm_scale = context.Input(4);
  const auto* conv_norm_scale = context.Input(5);

  const auto& key_shape = key->Shape();
  ORT_RETURN_IF_NOT(key_shape.NumDimensions() == 4,
                    "key must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = key_shape[0];
  const int64_t sequence_length = key_shape[1];
  const int64_t hc_mult = key_shape[2];
  const int64_t hidden_size = key_shape[3];

  ORT_RETURN_IF_NOT(query->Shape() == key_shape, "query must have the same shape as key");
  ORT_RETURN_IF_NOT(value->Shape() == TensorShape({batch_size, sequence_length, hidden_size}),
                    "value must have shape (batch_size, sequence_length, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");
  if (conv_norm_scale != nullptr) {
    ORT_RETURN_IF_NOT(conv_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                      "conv_norm_scale must have shape (hc_mult, hidden_size)");
  }

  auto* output = context.Output(0, key_shape);
  auto* output_normed = context.OutputCount() > 1 ? context.Output(1, key_shape) : nullptr;
  ORT_RETURN_IF_NOT(output_normed == nullptr || conv_norm_scale != nullptr,
                    "conv_norm_scale is required to produce the gated_value_normed output");
  const int64_t total = key_shape.Size();
  if (total == 0) {
    return Status::OK();
  }

  const int64_t rows = batch_size * sequence_length * hc_mult;
  const int components = onnxruntime::webgpu::GetMaxComponents(hidden_size);
  const int64_t hidden_vec_size = hidden_size / components;
  Tensor gate = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape({rows}));
  EngramGateScalarProgram gate_program;
  gate_program
      .CacheHint(components)
      .AddInputs({{key, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, components},
                  {query, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, components},
                  {key_norm_scale, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, components},
                  {query_norm_scale, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, components}})
      .AddOutput({&gate, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(kGateWorkgroupSize)
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(rows))
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(rows)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(hidden_vec_size)},
                            {epsilon_}});
  ORT_RETURN_IF_ERROR(context.RunProgram(gate_program));

  const int64_t total_vec = total / components;
  EngramGateProgram program;
  program
      .CacheHint(components)
      .AddInputs({{value, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, components},
                  {&gate, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output, ProgramTensorMetadataDependency::None, ProgramOutput::Flatten, components})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total_vec) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total_vec)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_vec_size)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  if (output_normed == nullptr) {
    return Status::OK();
  }

  EngramGateNormProgram norm_program{};
  norm_program.AddInputs({{output, ProgramTensorMetadataDependency::Type},
                          {conv_norm_scale, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output_normed, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(kGateWorkgroupSize)
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(rows))
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(rows)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {epsilon_}});
  return context.RunProgram(norm_program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
