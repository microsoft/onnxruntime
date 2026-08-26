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
  const ShaderVariableHelper* eos_token_id = nullptr;
  if (has_eos_token_id_) {
    eos_token_id = &shader.AddInput("eos_token_id", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* segment_ids = nullptr;
  if (has_segment_ids_) {
    segment_ids = &shader.AddInput("segment_ids", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const ShaderVariableHelper* present_tokens = nullptr;
  if (has_present_tokens_) {
    present_tokens = &shader.AddOutput("present_tokens", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << kernel_helper::kPositiveModWgsl;

  // Reads the raw (never EOS-substituted) token id at combined-timeline position `idx` (in
  // [0, history_length + sequence_length)) for batch row `b`: idx < history_length comes from
  // past_tokens (or eos_value when past_tokens is absent), otherwise it comes from input_ids.
  shader.AdditionalImplementation()
      << "fn combined_value(b: i32, history_length: i32, eos_value: i32, idx: i32) -> i32 {\n"
      << "  if (idx < history_length) {\n";
  if (has_past_tokens_) {
    shader.AdditionalImplementation()
        << "    return " << past_tokens->GetByOffset("b * history_length + idx") << ";\n";
  } else {
    shader.AdditionalImplementation() << "    return eos_value;\n";
  }
  shader.AdditionalImplementation()
      << "  }\n"
      << "  return " << input_ids.GetByOffset("b * i32(uniforms.sequence_length) + idx - history_length") << ";\n"
      << "}\n";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let history_length = i32(uniforms.history_length);\n"
      << "  let sequence_length = i32(uniforms.sequence_length);\n";
  if (has_eos_token_id_) {
    shader.MainFunctionBody() << "  let eos_value = " << eos_token_id->GetByOffset("0") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let eos_value = uniforms.pad_id;\n";
  }
  const bool do_reset = has_eos_token_id_ && reset_on_eos_;

  shader.MainFunctionBody()
      << "  if (global_idx >= uniforms.main_total) {\n"
      << "    let p = i32(global_idx - uniforms.main_total);\n"
      << "    let b = p / history_length;\n"
      << "    let i = p % history_length;\n"
      << "    let input_base = b * sequence_length;\n"
      << "    let idx = sequence_length + i;\n"
      << "    " << (present_tokens != nullptr ? present_tokens->SetByOffset("p", "combined_value(b, history_length, eos_value, idx)") : "") << "\n"
      << "    return;\n"
      << "  }\n"
      << "  let num_heads = (uniforms.max_ngram_size - 1u) * uniforms.n_head_per_ngram;\n"
      << "  let t = i32(global_idx % uniforms.sequence_length);\n"
      << "  let b = i32(global_idx / uniforms.sequence_length);\n"
      << "  let input_base = b * sequence_length;\n"
      << "  let output_base = i32(global_idx) * i32(num_heads);\n"
      << "  let idx = history_length + t;\n"
      << "  var last_reset = -(history_length + 2);\n"
      << "  var j = idx - 1;\n"
      << "  loop {\n"
      << "    if (j < idx - history_length || j < 0) { break; }\n"
      << "    var boundary = false;\n";
  if (do_reset) {
    shader.MainFunctionBody() << "    boundary = combined_value(b, history_length, eos_value, j) == eos_value;\n";
  }
  if (has_segment_ids_) {
    shader.MainFunctionBody()
        << "    if (!boundary && j > history_length) {\n"
        << "      let tj = j - history_length;\n"
        << "      if (" << segment_ids->GetByOffset("input_base + tj") << " != " << segment_ids->GetByOffset("input_base + tj - 1") << ") {\n"
        << "        boundary = true;\n"
        << "      }\n"
        << "    }\n";
  }
  shader.MainFunctionBody()
      << "    if (boundary) { last_reset = j; break; }\n"
      << "    j -= 1;\n"
      << "  }\n"
      << "  for (var n = 2u; n <= uniforms.max_ngram_size; n++) {\n"
      << "    var mix = 0i;\n"
      << "    for (var k = 0u; k < n; k++) {\n"
      << "      let source = idx - i32(k);\n"
      << "      var token = eos_value;\n"
      << "      if (last_reset < source) {\n"
      << "        token = combined_value(b, history_length, eos_value, source);\n"
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
      << "        result = positive_mod(mix, mod_value);\n"
      << "      }\n";
  if (has_head_offsets_) {
    shader.MainFunctionBody() << "      result = result + " << head_offsets->GetByOffset("out_h") << ";\n";
  }
  shader.MainFunctionBody()
      << "      " << output.SetByOffset("output_base + i32(out_h)", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

NGramHashMapping::NGramHashMapping(const OpKernelInfo& info) : WebGpuKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id_).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id_ >= std::numeric_limits<int32_t>::min() && pad_id_ <= std::numeric_limits<int32_t>::max(),
              "WebGPU NGramHashMapping only supports int32 ids");
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 0);
}

Status NGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* past_tokens = context.Input(3);
  const auto* head_offsets = context.Input(4);
  const auto* eos_token_id = context.Input(5);
  const auto* segment_ids = context.Input(6);
  const auto& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  const int64_t history_length = max_ngram_size_ - 1;

  if (past_tokens != nullptr) {
    ORT_RETURN_IF_NOT(past_tokens->Shape() == TensorShape({batch_size, history_length}),
                      "past_tokens must have shape (batch_size, max_ngram_size - 1)");
  }
  if (head_offsets != nullptr) {
    ORT_RETURN_IF_NOT(head_offsets->Shape() == TensorShape({num_heads}),
                      "head_offsets must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  }
  if (eos_token_id != nullptr) {
    ORT_RETURN_IF_NOT(eos_token_id->Shape().Size() == 1, "eos_token_id must be a scalar");
  }
  if (segment_ids != nullptr) {
    ORT_RETURN_IF_NOT(segment_ids->Shape() == TensorShape({batch_size, sequence_length}),
                      "segment_ids must have shape (batch_size, sequence_length)");
  }

  auto* output = context.Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  auto* present_tokens = context.Output(1, TensorShape({batch_size, history_length}));

  const int64_t main_total = input_shape.Size();
  const int64_t present_total = present_tokens != nullptr ? batch_size * history_length : 0;
  const int64_t total = main_total + present_total;
  if (total == 0) {
    return Status::OK();
  }

  NGramHashMappingProgram program(past_tokens != nullptr, head_offsets != nullptr, eos_token_id != nullptr,
                                  segment_ids != nullptr, present_tokens != nullptr, reset_on_eos_ != 0);
  program.AddInput({input_ids, ProgramTensorMetadataDependency::None});
  program.AddInput({multipliers, ProgramTensorMetadataDependency::None});
  program.AddInput({vocab_sizes, ProgramTensorMetadataDependency::None});
  if (past_tokens != nullptr) {
    program.AddInput({past_tokens, ProgramTensorMetadataDependency::None});
  }
  if (head_offsets != nullptr) {
    program.AddInput({head_offsets, ProgramTensorMetadataDependency::None});
  }
  if (eos_token_id != nullptr) {
    program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
  }
  if (segment_ids != nullptr) {
    program.AddInput({segment_ids, ProgramTensorMetadataDependency::None});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None});
  if (present_tokens != nullptr) {
    program.AddOutput({present_tokens, ProgramTensorMetadataDependency::None});
  }
  program.SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                            {onnxruntime::narrow<uint32_t>(main_total)},
                            {onnxruntime::narrow<uint32_t>(sequence_length)},
                            {onnxruntime::narrow<uint32_t>(history_length)},
                            {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                            {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                            {onnxruntime::narrow<int32_t>(pad_id_)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
