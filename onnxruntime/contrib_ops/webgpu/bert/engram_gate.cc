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

Status EngramGateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& embeddings = shader.AddInput("embeddings", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& hidden_states = shader.AddInput("hidden_states", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& key_weight = shader.AddInput("key_weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* key_bias = nullptr;
  if (has_key_bias_) {
    key_bias = &shader.AddInput("key_bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& value_weight = shader.AddInput("value_weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* value_bias = nullptr;
  if (has_value_bias_) {
    value_bias = &shader.AddInput("value_bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& key_norm_scale = shader.AddInput("key_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& query_norm_scale = shader.AddInput("query_norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation() << kernel_helper::kStableSigmoidWgsl;

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let token = global_idx / (uniforms.hc_mult * uniforms.hidden_size);\n"
      << "  let embedding_base = token * uniforms.embedding_size;\n"
      << "  let hidden_base = (token * uniforms.hc_mult + g) * uniforms.hidden_size;\n";
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
      << "  var key_sum_sq = 0.0;\n"
      << "  var query_sum_sq = 0.0;\n"
      << "  var dot_numerator = 0.0;\n"
      << "  for (var d = 0u; d < uniforms.hidden_size; d++) {\n";
  if (has_key_bias_) {
    shader.MainFunctionBody() << "    var key = f32(" << key_bias->GetByOffset("g * uniforms.hidden_size + d") << ");\n";
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
      << "    dot_numerator += key * f32(" << key_norm_scale.GetByOffset("g * uniforms.hidden_size + d")
      << ") * query * f32(" << query_norm_scale.GetByOffset("g * uniforms.hidden_size + d") << ");\n"
      << "  }\n"
      << "  let key_inv_rms = inverseSqrt(key_sum_sq / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "  let query_inv_rms = inverseSqrt(query_sum_sq / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "  let dot = dot_numerator * key_inv_rms * query_inv_rms / sqrt(f32(uniforms.hidden_size));\n"
      << "  let gate_arg = sign(dot) * sqrt(max(abs(dot), 0.000001));\n"
      << "  " << output.SetByOffset("global_idx", "output_element_t(stable_sigmoid(gate_arg) * value)") << "\n";
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

  auto* output = context.Output(0, hidden_shape);
  const int64_t total = hidden_shape.Size();
  if (total == 0) {
    return Status::OK();
  }
  EngramGateProgram program{key_bias != nullptr, value_bias != nullptr};
  program.CacheHint(key_bias != nullptr, value_bias != nullptr)
      .AddInputs({{embeddings, ProgramTensorMetadataDependency::Type},
                  {hidden_states, ProgramTensorMetadataDependency::Type},
                  {key_weight, ProgramTensorMetadataDependency::Type}});
  if (key_bias != nullptr) {
    program.AddInput({key_bias, ProgramTensorMetadataDependency::Type});
  }
  program.AddInput({value_weight, ProgramTensorMetadataDependency::Type});
  if (value_bias != nullptr) {
    program.AddInput({value_bias, ProgramTensorMetadataDependency::Type});
  }
  program.AddInputs({{key_norm_scale, ProgramTensorMetadataDependency::Type},
                     {query_norm_scale, ProgramTensorMetadataDependency::Type}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(embedding_size)},
                            {epsilon_}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
