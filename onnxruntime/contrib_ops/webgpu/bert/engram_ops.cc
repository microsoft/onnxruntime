// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/engram_ops.h"

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    ShortConv,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    ShortConv);

ONNX_OPERATOR_KERNEL_EX(
    NgramHashMapping,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    NgramHashMapping);

ONNX_OPERATOR_KERNEL_EX(
    EngramGate,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes()),
    EngramGate);

Status ShortConvProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("input", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& norm_scale = shader.AddInput("norm_scale", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* bias = nullptr;
  if (has_bias_) {
    bias = &shader.AddInput("bias", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "fn stable_sigmoid(x: f32) -> f32 {\n"
      << "  if (x > 0.0) { return 1.0 / (1.0 + exp(-x)); }\n"
      << "  let e = exp(x);\n"
      << "  return e / (1.0 + e);\n"
      << "}\n";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let channels = uniforms.hc_mult * uniforms.hidden_size;\n"
      << "  let c = global_idx % uniforms.hidden_size;\n"
      << "  let g = (global_idx / uniforms.hidden_size) % uniforms.hc_mult;\n"
      << "  let t = (global_idx / channels) % uniforms.sequence_length;\n"
      << "  let b = global_idx / (uniforms.sequence_length * channels);\n"
      << "  let flat_channel = g * uniforms.hidden_size + c;\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "  var sum = f32(" << bias->GetByOffset("flat_channel") << ");\n";
  } else {
    shader.MainFunctionBody() << "  var sum = 0.0;\n";
  }
  shader.MainFunctionBody()
      << "  for (var k = 0u; k < uniforms.kernel_size; k++) {\n"
      << "    let offset = (uniforms.kernel_size - 1u - k) * uniforms.dilation;\n"
      << "    if (t >= offset) {\n"
      << "      let source_t = t - offset;\n"
      << "      let row_base = ((b * uniforms.sequence_length + source_t) * uniforms.hc_mult + g) * uniforms.hidden_size;\n"
      << "      var sum_sq = 0.0;\n"
      << "      for (var i = 0u; i < uniforms.hidden_size; i++) {\n"
      << "        let v = f32(" << input.GetByOffset("row_base + i") << ");\n"
      << "        sum_sq += v * v;\n"
      << "      }\n"
      << "      let inv_rms = inverseSqrt(sum_sq / f32(uniforms.hidden_size) + uniforms.epsilon);\n"
      << "      let normed = f32(" << input.GetByOffset("row_base + c") << ") * inv_rms * f32("
      << norm_scale.GetByOffset("g * uniforms.hidden_size + c") << ");\n"
      << "      sum += normed * f32(" << weight.GetByOffset("flat_channel * uniforms.kernel_size + k") << ");\n"
      << "    }\n"
      << "  }\n";
  if (apply_silu_) {
    shader.MainFunctionBody() << "  sum = sum * stable_sigmoid(sum);\n";
  }
  shader.MainFunctionBody() << "  " << output.SetByOffset("global_idx", "output_element_t(sum)") << "\n";
  return Status::OK();
}

ShortConv::ShortConv(const OpKernelInfo& info) : WebGpuKernel(info) {
  activation_ = info.GetAttrOrDefault<std::string>("activation", "silu");
  ORT_ENFORCE(activation_ == "none" || activation_ == "silu" || activation_ == "swish",
              "activation must be one of: none, silu, swish");
  dilation_ = info.GetAttrOrDefault<int64_t>("dilation", 1);
  ORT_ENFORCE(dilation_ >= 1, "dilation must be >= 1");
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-5f);
}

Status ShortConv::ComputeInternal(ComputeContext& context) const {
  const auto* input = context.Input(0);
  const auto* weight = context.Input(1);
  const auto* norm_scale = context.Input(2);
  const auto* bias = context.Input(3);
  const auto& input_shape = input->Shape();
  const auto& weight_shape = weight->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 4,
                    "input must have shape (batch_size, sequence_length, hc_mult, hidden_size)");
  ORT_RETURN_IF_NOT(weight_shape.NumDimensions() == 3,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t hc_mult = input_shape[2];
  const int64_t hidden_size = input_shape[3];
  const int64_t channels = hc_mult * hidden_size;
  ORT_RETURN_IF_NOT(norm_scale->Shape() == TensorShape({hc_mult, hidden_size}),
                    "norm_scale shape must match input hc_mult and hidden_size");
  ORT_RETURN_IF_NOT(weight_shape[0] == channels && weight_shape[1] == 1,
                    "weight must have shape (hc_mult * hidden_size, 1, kernel_size)");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape() == TensorShape({channels}), "bias must have shape (hc_mult * hidden_size)");
  }
  auto* output = context.Output(0, input_shape);
  const int64_t total = input_shape.Size();
  if (total == 0) {
    return Status::OK();
  }

  ShortConvProgram program{bias != nullptr, activation_ == "silu" || activation_ == "swish"};
  program.CacheHint(bias != nullptr, activation_)
      .AddInputs({{input, ProgramTensorMetadataDependency::Type},
                  {weight, ProgramTensorMetadataDependency::Type},
                  {norm_scale, ProgramTensorMetadataDependency::Type}});
  if (bias != nullptr) {
    program.AddInput({bias, ProgramTensorMetadataDependency::Type});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(hc_mult)},
                            {onnxruntime::narrow<uint32_t>(hidden_size)},
                            {onnxruntime::narrow<uint32_t>(weight_shape[2])},
                            {onnxruntime::narrow<uint32_t>(dilation_)},
                            {epsilon_}});
  return context.RunProgram(program);
}

Status NgramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let num_heads = (uniforms.max_ngram_size - 1u) * uniforms.n_head_per_ngram;\n"
      << "  let t = global_idx % uniforms.sequence_length;\n"
      << "  let b = global_idx / uniforms.sequence_length;\n"
      << "  let input_base = b * uniforms.sequence_length;\n"
      << "  let output_base = global_idx * num_heads;\n"
      << "  for (var n = 2u; n <= uniforms.max_ngram_size; n++) {\n"
      << "    var mix = 0i;\n"
      << "    for (var k = 0u; k < n; k++) {\n"
      << "      var token = uniforms.pad_id;\n"
      << "      if (t >= k) {\n"
      << "        token = " << input_ids.GetByOffset("input_base + t - k") << ";\n"
      << "      }\n"
      << "      let product = token * " << multipliers.GetByOffset("k") << ";\n"
      << "      if (k == 0u) { mix = product; } else { mix = mix ^ product; }\n"
      << "    }\n"
      << "    let ngram_offset = (n - 2u) * uniforms.n_head_per_ngram;\n"
      << "    for (var h = 0u; h < uniforms.n_head_per_ngram; h++) {\n"
      << "      let out_h = ngram_offset + h;\n"
      << "      let mod_value = " << vocab_sizes.GetByOffset("out_h") << ";\n"
      << "      var result = 0i;\n"
      << "      if (mod_value > 0i) {\n"
      << "        result = mix % mod_value;\n"
      << "        if (result < 0i) { result += mod_value; }\n"
      << "      }\n"
      << "      " << output.SetByOffset("output_base + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

NgramHashMapping::NgramHashMapping(const OpKernelInfo& info) : WebGpuKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id_).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id_ >= std::numeric_limits<int32_t>::min() && pad_id_ <= std::numeric_limits<int32_t>::max(),
              "WebGPU NgramHashMapping only supports int32 ids");
}

Status NgramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  auto* output = context.Output(0, TensorShape({input_shape[0], input_shape[1], num_heads}));
  const int64_t total = input_shape.Size();
  if (total == 0) {
    return Status::OK();
  }

  NgramHashMappingProgram program;
  program.AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                     {multipliers, ProgramTensorMetadataDependency::None},
                     {vocab_sizes, ProgramTensorMetadataDependency::None}})
      .AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(input_shape[1])},
                            {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                            {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                            {onnxruntime::narrow<int32_t>(pad_id_)}});
  return context.RunProgram(program);
}

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

  shader.AdditionalImplementation()
      << "fn stable_sigmoid(x: f32) -> f32 {\n"
      << "  if (x > 0.0) { return 1.0 / (1.0 + exp(-x)); }\n"
      << "  let e = exp(x);\n"
      << "  return e / (1.0 + e);\n"
      << "}\n";

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
