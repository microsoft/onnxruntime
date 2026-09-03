// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/varlen_ngram_hash_mapping.h"

#include "contrib_ops/webgpu/bert/engram_helper.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

#include <limits>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

ONNX_OPERATOR_KERNEL_EX(
    VarlenNGramHashMapping,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()),
    VarlenNGramHashMapping);

Status VarlenNGramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << engram_helper::kPositiveModWgsl;

  // Malformed offsets make this workgroup return before any input, history, or output access.
  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size) { return; }\n"
      << "  let first = " << cu_seqlens.GetByOffset("0u") << ";\n"
      << "  let last = " << cu_seqlens.GetByOffset("uniforms.batch_size") << ";\n"
      << "  let start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  if (first != 0 || u32(last) != uniforms.total_tokens || start < 0 || start >= end ||\n"
      << "      u32(end) > uniforms.total_tokens) {\n"
      << "    return;\n"
      << "  }\n"
      << "  let local_length = u32(end - start);\n"
      << "  let state_length = uniforms.max_ngram_size - 1u;\n"
      << "  let num_heads = state_length * uniforms.n_head_per_ngram;\n"
      << "  let past_base = b * state_length;\n"
      << "  for (var out_h = local_idx; out_h < num_heads; out_h += workgroup_size_x) {\n"
      << "    let n = out_h / uniforms.n_head_per_ngram + 2u;\n"
      << "    let mod_value = " << vocab_sizes.GetByOffset("out_h") << ";\n"
      << "    for (var t = 0u; t < local_length; t++) {\n"
      << "      var mix = 0i;\n"
      << "      for (var k = 0u; k < n; k++) {\n"
      << "        var token = uniforms.pad_id;\n"
      << "        if (t >= k) {\n"
      << "          token = " << input_ids.GetByOffset("u32(start) + t - k") << ";\n"
      << "        }\n";
  if (has_past_ids_) {
    shader.MainFunctionBody()
        << "        if (t < k) {\n"
        << "          token = " << past_ids->GetByOffset("past_base + state_length + t - k") << ";\n"
        << "        }\n";
  }
  shader.MainFunctionBody()
      << "        let product = token * " << multipliers.GetByOffset("k") << ";\n"
      << "        if (k == 0u) { mix = product; } else { mix = mix ^ product; }\n"
      << "      }\n"
      << "      var result = 0i;\n"
      << "      if (mod_value > 0i) {\n"
      << "        result = positive_mod(mix, mod_value);\n"
      << "      }\n"
      << "      " << output.SetByOffset("(u32(start) + t) * num_heads + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status VarlenNGramPresentIdsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& cu_seqlens = shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const ShaderVariableHelper* input_ids = nullptr;
  if (has_input_ids_) {
    input_ids = &shader.AddInput("input_ids", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const auto& present_ids = shader.AddOutput("present_ids", ShaderUsage::UseUniform);

  // Malformed offsets make this thread's request fall back to an all-pad_id present_ids row
  // instead of indexing into input_ids/past_ids with an out-of-range start/end, mirroring the
  // containment applied by VarlenNGramHashMappingProgram.
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let slot = global_idx % uniforms.state_length;\n"
      << "  let b = global_idx / uniforms.state_length;\n"
      << "  let first = " << cu_seqlens.GetByOffset("0u") << ";\n"
      << "  let last = " << cu_seqlens.GetByOffset("uniforms.batch_size") << ";\n"
      << "  let start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  var token = uniforms.pad_id;\n"
      << "  if (first == 0 && u32(last) == uniforms.total_tokens && start >= 0 && start < end &&\n"
      << "      u32(end) <= uniforms.total_tokens) {\n"
      << "    let local_length = u32(end - start);\n";
  if (has_input_ids_) {
    shader.MainFunctionBody()
        << "    if (slot + local_length >= uniforms.state_length) {\n"
        << "      let source_t = slot + local_length - uniforms.state_length;\n"
        << "      token = " << input_ids->GetByOffset("u32(start) + source_t") << ";\n"
        << "    }\n";
  }
  if (has_past_ids_) {
    shader.MainFunctionBody()
        << "    if (slot + local_length < uniforms.state_length) {\n"
        << "      token = "
        << past_ids->GetByOffset("b * uniforms.state_length + slot + local_length") << ";\n"
        << "    }\n";
  }
  shader.MainFunctionBody()
      << "  }\n"
      << "  " << present_ids.SetByOffset("global_idx", "token") << "\n";
  return Status::OK();
}

VarlenNGramHashMapping::VarlenNGramHashMapping(const OpKernelInfo& info) : WebGpuKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("max_ngram_size", &max_ngram_size_).IsOK(),
              "max_ngram_size attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("n_head_per_ngram", &n_head_per_ngram_).IsOK(),
              "n_head_per_ngram attribute is required");
  ORT_ENFORCE(info.GetAttr<int64_t>("pad_id", &pad_id_).IsOK(), "pad_id attribute is required");
  ORT_ENFORCE(max_ngram_size_ >= 2, "max_ngram_size must be at least 2");
  ORT_ENFORCE(n_head_per_ngram_ >= 1, "n_head_per_ngram must be positive");
  ORT_ENFORCE(pad_id_ >= std::numeric_limits<int32_t>::min() && pad_id_ <= std::numeric_limits<int32_t>::max(),
              "WebGPU VarlenNGramHashMapping only supports int32 ids");
}

Status VarlenNGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* cu_seqlens = context.Input(3);
  const auto* past_ids = context.Input(4);

  ORT_RETURN_IF_NOT(input_ids->Shape().NumDimensions() == 1, "input_ids must have rank 1 (total_tokens)");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] == max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const auto& cu_seqlens_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1 && cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have rank 1 with at least 2 elements");
  // batch_size = cu_seqlens.Shape()[0] - 1. This is a shape-only computation; the offset values
  // themselves stay on the device and are validated inside the shader.
  const int64_t batch_size = cu_seqlens_shape[0] - 1;

  const int64_t total_tokens = input_ids->Shape()[0];
  ORT_RETURN_IF_NOT(total_tokens >= batch_size,
                    "total_tokens must be at least batch_size because every request must contain a token");

  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }
  const bool has_past_ids = past_ids != nullptr;

  auto* output = context.Output(0, TensorShape({total_tokens, num_heads}));
  auto* present_ids = context.Output(1, TensorShape({batch_size, state_length}));

  if (present_ids != nullptr && batch_size * state_length > 0) {
    const int64_t present_total = batch_size * state_length;
    const bool has_input_ids = total_tokens > 0;
    VarlenNGramPresentIdsProgram present_program{has_input_ids, has_past_ids};
    present_program.CacheHint(has_input_ids, has_past_ids);
    present_program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
    if (has_input_ids) {
      present_program.AddInput({input_ids, ProgramTensorMetadataDependency::None});
    }
    if (has_past_ids) {
      present_program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    present_program.AddOutput({present_ids, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(present_total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(present_total)},
                              {onnxruntime::narrow<uint32_t>(state_length)},
                              {onnxruntime::narrow<uint32_t>(batch_size)},
                              {onnxruntime::narrow<uint32_t>(total_tokens)},
                              {onnxruntime::narrow<int32_t>(pad_id_)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(present_program));
  }

  if (total_tokens == 0 || batch_size == 0) {
    return Status::OK();
  }

  VarlenNGramHashMappingProgram program{has_past_ids};
  program.CacheHint(has_past_ids)
      .AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                  {multipliers, ProgramTensorMetadataDependency::None},
                  {vocab_sizes, ProgramTensorMetadataDependency::None},
                  {cu_seqlens, ProgramTensorMetadataDependency::None}});
  if (has_past_ids) {
    program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(batch_size))
      .SetWorkgroupSize(WORKGROUP_SIZE)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(batch_size)},
                            {onnxruntime::narrow<uint32_t>(total_tokens)},
                            {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                            {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                            {onnxruntime::narrow<int32_t>(pad_id_)}});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
