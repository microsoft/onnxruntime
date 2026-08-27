// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/engram_gate.h"

#include "contrib_ops/webgpu/bert/kernel_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

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
  const auto& embeddings = shader.AddInput("embeddings", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& hidden_states = shader.AddInput("hidden_states", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& key_weight = shader.AddInput("key_weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* key_bias = nullptr;
  if (has_key_bias_) {
    key_bias = &shader.AddInput("key_bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& key_norm_scale = shader.AddInput("key_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& query_norm_scale = shader.AddInput("query_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& gate = shader.AddOutput("gate", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << kernel_helper::kStableSigmoidWgsl << kernel_helper::kEngramGateArgWgsl
      << "var<workgroup> key_partials: array<f32, " << kGateWorkgroupSize << ">;\n"
      << "var<workgroup> query_partials: array<f32, " << kGateWorkgroupSize << ">;\n"
      << "var<workgroup> dot_partials: array<f32, " << kGateWorkgroupSize << ">;\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let g = row % uniforms.hc_mult;\n"
      << "  let token = row / uniforms.hc_mult;\n"
      << "  let embedding_base = token * uniforms.embedding_size;\n"
      << "  let hidden_base = row * uniforms.hidden_size;\n"
      << "  let scale_base = g * uniforms.hidden_size;\n"
      << "  var key_sum_sq = 0.0;\n"
      << "  var query_sum_sq = 0.0;\n"
      << "  var dot_numerator = 0.0;\n"
      << "  for (var d = local_idx; d < uniforms.hidden_size; d += " << kGateWorkgroupSize << "u) {\n";
  if (has_key_bias_) {
    shader.MainFunctionBody() << "    var key = f32(" << key_bias->GetByOffset("scale_base + d") << ");\n";
  } else {
    shader.MainFunctionBody() << "    var key = 0.0;\n";
  }
  shader.MainFunctionBody()
      << "    for (var e = 0u; e < uniforms.embedding_size; e++) {\n"
      << "      key += f32(" << embeddings.GetByOffset("embedding_base + e") << ") * f32("
      << key_weight.GetByOffset("(g * uniforms.embedding_size + e) * uniforms.hidden_size + d") << ");\n"
      << "    }\n"
      << "    let query = f32(" << hidden_states.GetByOffset("hidden_base + d") << ");\n"
      << "    key_sum_sq += key * key;\n"
      << "    query_sum_sq += query * query;\n"
      << "    dot_numerator += key * f32(" << key_norm_scale.GetByOffset("scale_base + d")
      << ") * query * f32(" << query_norm_scale.GetByOffset("scale_base + d") << ");\n"
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
  const auto& embeddings = shader.AddInput("embeddings", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& value_weight = shader.AddInput("value_weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* value_bias = nullptr;
  if (has_value_bias_) {
    value_bias = &shader.AddInput("value_bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let row = global_idx / uniforms.hidden_size;\n"
      << "  let token = row / uniforms.hc_mult;\n"
      << "  let embedding_base = token * uniforms.embedding_size;\n";
  if (has_value_bias_) {
    shader.MainFunctionBody() << "  var value = f32(" << value_bias->GetByOffset("c") << ");\n";
  } else {
    shader.MainFunctionBody() << "  var value = 0.0;\n";
  }
  shader.MainFunctionBody()
      << "  for (var e = 0u; e < uniforms.embedding_size; e++) {\n"
      << "    value += f32(" << embeddings.GetByOffset("embedding_base + e") << ") * f32("
      << value_weight.GetByOffset("e * uniforms.hidden_size + c") << ");\n"
      << "  }\n"
      << "  " << output.SetByOffset("global_idx", "output_element_t(" + gate.GetByOffset("row") + " * value)") << "\n";
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
  const auto* embeddings = context.Input(0);
  const auto* hidden_states = context.Input(1);
  const auto* key_weight = context.Input(2);
  const auto* key_bias = context.Input(3);
  const auto* value_weight = context.Input(4);
  const auto* value_bias = context.Input(5);
  const auto* key_norm_scale = context.Input(6);
  const auto* query_norm_scale = context.Input(7);
  const auto* conv_norm_scale = context.Input(8);
  const auto& embeddings_shape = embeddings->Shape();
  const auto& hidden_shape = hidden_states->Shape();
  ORT_RETURN_IF_NOT(embeddings_shape.NumDimensions() == 3,
                    "embeddings must have shape (batch_size, sequence_length, embedding_size)");
  ORT_RETURN_IF_NOT(hidden_shape.NumDimensions() == 4,
                    "hidden_states must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  const int64_t batch_size = hidden_shape[0];
  const int64_t sequence_length = hidden_shape[1];
  const int64_t hc_mult = hidden_shape[2];
  const int64_t hidden_size = hidden_shape[3];
  const int64_t embedding_size = embeddings_shape[2];
  ORT_RETURN_IF_NOT(embeddings_shape[0] == batch_size && embeddings_shape[1] == sequence_length,
                    "embeddings and hidden_states batch/sequence dimensions must match");
  ORT_RETURN_IF_NOT(key_weight->Shape() == TensorShape({hc_mult, embedding_size, hidden_size}),
                    "key_weight must have shape (hc_mult, embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(value_weight->Shape() == TensorShape({embedding_size, hidden_size}),
                    "value_weight must have shape (embedding_size, hidden_size)");
  ORT_RETURN_IF_NOT(key_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "key_norm_scale must have shape (hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(query_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "query_norm_scale must have shape (hc_mult, hidden_size)");
  if (key_bias != nullptr) {
    ORT_RETURN_IF_NOT(key_bias->Shape() == TensorShape({hc_mult, hidden_size}),
                      "key_bias must have shape (hc_mult, hidden_size)");
  }
  if (value_bias != nullptr) {
    ORT_RETURN_IF_NOT(value_bias->Shape() == TensorShape({hidden_size}),
                      "value_bias must have shape (hidden_size)");
  }
  if (conv_norm_scale != nullptr) {
    ORT_RETURN_IF_NOT(conv_norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                      "conv_norm_scale must have shape (hc_mult, hidden_size)");
  }

  auto* output = context.Output(0, hidden_shape);
  auto* output_normed = context.OutputCount() > 1 ? context.Output(1, hidden_shape) : nullptr;
  ORT_RETURN_IF_NOT(output_normed == nullptr || conv_norm_scale != nullptr,
                    "conv_norm_scale is required to produce the gated_value_normed output");
  const int64_t total = hidden_shape.Size();
  if (total == 0) {
    return Status::OK();
  }
  // First pass: one scalar gate per (token, g) row.
  const int64_t rows = batch_size * sequence_length * hc_mult;
  Tensor gate = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape({rows}));
  EngramGateScalarProgram gate_program{key_bias != nullptr};
  gate_program.CacheHint(key_bias != nullptr)
      .AddInputs({{embeddings, ProgramTensorMetadataDependency::Type},
                  {hidden_states, ProgramTensorMetadataDependency::Type},
                  {key_weight, ProgramTensorMetadataDependency::Type}});
  if (key_bias != nullptr) {
    gate_program.AddInput({key_bias, ProgramTensorMetadataDependency::Type});
  }
  gate_program.AddInputs({{key_norm_scale, ProgramTensorMetadataDependency::Type},
                          {query_norm_scale, ProgramTensorMetadataDependency::Type}})
      .AddOutput({&gate, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(kGateWorkgroupSize)
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(rows))
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(rows)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(embedding_size)},
                            {epsilon_}});
  ORT_RETURN_IF_ERROR(context.RunProgram(gate_program));

  // Second pass: apply the shared gate to the value projection for every output channel.
  EngramGateProgram program{value_bias != nullptr};
  program.CacheHint(value_bias != nullptr)
      .AddInputs({{embeddings, ProgramTensorMetadataDependency::Type},
                  {value_weight, ProgramTensorMetadataDependency::Type}});
  if (value_bias != nullptr) {
    program.AddInput({value_bias, ProgramTensorMetadataDependency::Type});
  }
  program.AddInput({&gate, ProgramTensorMetadataDependency::Type})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(embedding_size)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(program));

  if (output_normed == nullptr) {
    return Status::OK();
  }

  // Third pass: branchwise RMSNorm of gated_value into gated_value_normed, one workgroup per row.
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
