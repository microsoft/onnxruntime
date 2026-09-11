// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/ngram_hash_mapping.h"

#include "contrib_ops/webgpu/bert/kernel_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    NGramHashMapping,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    NGramHashMapping);

Status NGramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_tokens = nullptr;
  if (has_past_tokens_) {
    past_tokens = &shader.AddInput("past_tokens", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* head_offsets = nullptr;
  if (has_head_offsets_) {
    head_offsets = &shader.AddInput("head_offsets", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << kernel_helper::kPositiveModWgsl;
  if (qwen_mode_) {
    shader.AdditionalImplementation()
        << "fn token_at(b: u32, history_t: u32) -> i32 {\n"
        << "  if (history_t < uniforms.context_length) {\n";
    if (has_past_tokens_) {
      shader.AdditionalImplementation() << "    return " << past_tokens->GetByOffset("b * uniforms.context_length + history_t") << ";\n";
    } else {
      shader.AdditionalImplementation() << "    return uniforms.eos_token_id;\n";
    }
    shader.AdditionalImplementation()
        << "  }\n"
        << "  return " << input_ids.GetByOffset("b * uniforms.sequence_length + history_t - uniforms.context_length") << ";\n"
        << "}\n";
  }

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
      << "      var token = uniforms.pad_id;\n";
  if (qwen_mode_) {
    shader.MainFunctionBody()
        << "      let history_t = uniforms.context_length + t;\n"
        << "      let source_t = history_t - k;\n"
        << "      var valid = true;\n"
        << "      if (uniforms.reset_on_eos != 0u && k > 0u) {\n"
        << "        for (var p = source_t; p < history_t; p++) {\n"
        << "          if (token_at(b, p) == uniforms.eos_token_id) { valid = false; break; }\n"
        << "        }\n"
        << "      }\n"
        << "      token = select(uniforms.eos_token_id, token_at(b, source_t), valid);\n";
  } else {
    shader.MainFunctionBody()
        << "      if (t >= k) {\n"
        << "        token = " << input_ids.GetByOffset("input_base + t - k") << ";\n"
        << "      }\n";
  }
  shader.MainFunctionBody()
      << "      let product = token * " << multipliers.GetByOffset("k") << ";\n"
      << "      if (k == 0u) { mix = product; } else { mix = mix ^ product; }\n"
      << "    }\n"
      << "    let ngram_offset = (n - 2u) * uniforms.n_head_per_ngram;\n"
      << "    for (var h = 0u; h < uniforms.n_head_per_ngram; h++) {\n"
      << "      let out_h = ngram_offset + h;\n"
      << "      let mod_value = " << vocab_sizes.GetByOffset("out_h") << ";\n"
      << "      var result = 0i;\n"
      << "      if (mod_value > 0i) {\n"
      << "        result = positive_mod(mix, mod_value);\n";
  if (has_head_offsets_) {
    shader.MainFunctionBody() << "        result += " << head_offsets->GetByOffset("out_h") << ";\n";
  }
  shader.MainFunctionBody()
      << "      }\n"
      << "      " << output.SetByOffset("output_base + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status NGramPresentTokensProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_tokens = nullptr;
  if (has_past_tokens_) {
    past_tokens = &shader.AddInput("past_tokens", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("present_tokens", ShaderUsage::UseUniform);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let i = global_idx % uniforms.context_length;\n"
      << "  let b = global_idx / uniforms.context_length;\n"
      << "  let history_t = uniforms.sequence_length + i;\n"
      << "  var token = uniforms.eos_token_id;\n"
      << "  if (history_t < uniforms.context_length) {\n";
  if (has_past_tokens_) {
    shader.MainFunctionBody() << "    token = " << past_tokens->GetByOffset("b * uniforms.context_length + history_t") << ";\n";
  }
  shader.MainFunctionBody()
      << "  } else {\n"
      << "    token = " << input_ids.GetByOffset("b * uniforms.sequence_length + history_t - uniforms.context_length") << ";\n"
      << "  }\n"
      << "  " << output.SetByOffset("global_idx", "token") << "\n";
  return Status::OK();
}

NGramHashMapping::NGramHashMapping(const OpKernelInfo& info) : WebGpuKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  pad_id_ = info.GetAttrOrDefault<int64_t>("pad_id", 0);
  has_eos_token_id_ = info.GetAttr<int64_t>("eos_token_id", &eos_token_id_).IsOK();
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 1) != 0;
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id_ >= std::numeric_limits<int32_t>::min() && pad_id_ <= std::numeric_limits<int32_t>::max(),
              "WebGPU NGramHashMapping only supports int32 ids");
}

Status NGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* past_tokens = context.Input(3);
  const auto* head_offsets = context.Input(4);
  const auto& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t context_length = max_ngram_size_ - 1;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  if (past_tokens != nullptr) {
    ORT_RETURN_IF_NOT(past_tokens->Shape() == TensorShape({batch_size, context_length}),
                      "past_tokens must have shape (batch_size, max_ngram_size - 1)");
  }
  if (head_offsets != nullptr) {
    ORT_RETURN_IF_NOT(head_offsets->Shape() == TensorShape({num_heads}),
                      "head_offsets must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  }
  auto* output = context.Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  auto* present_tokens = context.Output(1, TensorShape({batch_size, context_length}));
  const int64_t total = input_shape.Size();
  const bool qwen_mode = has_eos_token_id_ || past_tokens != nullptr || head_offsets != nullptr || present_tokens != nullptr;
  if (total > 0) {
    NGramHashMappingProgram program{past_tokens != nullptr, head_offsets != nullptr, qwen_mode};
    program.CacheHint(past_tokens != nullptr, head_offsets != nullptr, qwen_mode)
        .AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                    {multipliers, ProgramTensorMetadataDependency::None},
                    {vocab_sizes, ProgramTensorMetadataDependency::None}});
    if (past_tokens != nullptr) {
      program.AddInput({past_tokens, ProgramTensorMetadataDependency::None});
    }
    if (head_offsets != nullptr) {
      program.AddInput({head_offsets, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({output, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(context_length)},
                              {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                              {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                              {onnxruntime::narrow<int32_t>(pad_id_)},
                              {onnxruntime::narrow<int32_t>(eos_token_id_)},
                              {onnxruntime::narrow<uint32_t>(reset_on_eos_ ? 1 : 0)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(program));
  }

  if (present_tokens != nullptr) {
    const int64_t present_total = batch_size * context_length;
    NGramPresentTokensProgram present_program{past_tokens != nullptr};
    present_program.CacheHint(past_tokens != nullptr)
        .AddInput({input_ids, ProgramTensorMetadataDependency::None});
    if (past_tokens != nullptr) {
      present_program.AddInput({past_tokens, ProgramTensorMetadataDependency::None});
    }
    present_program.AddOutput({present_tokens, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(present_total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(present_total)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(context_length)},
                              {onnxruntime::narrow<int32_t>(eos_token_id_)}});
    return context.RunProgram(present_program);
  }
  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
