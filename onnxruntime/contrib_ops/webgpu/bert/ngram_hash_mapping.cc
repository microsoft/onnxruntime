// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/ngram_hash_mapping.h"

#include "contrib_ops/webgpu/bert/engram_helper.h"
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
        .MayInplace(3, 1)
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    NGramHashMapping);

Status NGramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);

  shader.AdditionalImplementation() << engram_helper::kPositiveModWgsl;

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let num_heads = (uniforms.max_ngram_size - 1u) * uniforms.n_head_per_ngram;\n"
      << "  let t = global_idx % uniforms.sequence_length;\n"
      << "  let b = global_idx / uniforms.sequence_length;\n"
      << "  let input_base = b * uniforms.sequence_length;\n"
      << "  let output_base = global_idx * num_heads;\n"
      << "  let state_length = uniforms.max_ngram_size - 1u;\n"
      << "  let past_base = b * state_length;\n"
      << "  for (var n = 2u; n <= uniforms.max_ngram_size; n++) {\n"
      << "    var mix = 0i;\n"
      << "    for (var k = 0u; k < n; k++) {\n"
      << "      var token = uniforms.pad_id;\n"
      << "      if (t >= k) {\n"
      << "        token = " << input_ids.GetByOffset("input_base + t - k") << ";\n"
      << "      }\n";
  if (has_past_ids_) {
    // past_ids is right-aligned, so position -1 is its last slot. k <= max_ngram_size - 1 keeps the
    // slot inside the window, so no additional bounds check is needed here.
    shader.MainFunctionBody()
        << "      if (t < k) {\n"
        << "        token = " << past_ids->GetByOffset("past_base + state_length + t - k") << ";\n"
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
      << "        result = positive_mod(mix, mod_value);\n"
      << "      }\n"
      << "      " << output.SetByOffset("output_base + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status NGramPresentIdsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* input_ids = nullptr;
  if (has_input_ids_) {
    input_ids = &shader.AddInput("input_ids", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const auto& present_ids = shader.AddOutput("present_ids", ShaderUsage::UseUniform);

  // past_ids and present_ids may be the same buffer, which is what threading present_ids straight
  // back into past_ids produces. Slot `slot` writes index `slot` and reads index
  // `slot + sequence_length`, so the read and write ranges overlap and must be separated. One
  // workgroup owns one batch row and walks it in ascending workgroup-sized chunks: a barrier
  // separates the chunk's reads from its writes, and a chunk only writes indices strictly below the
  // read indices of every later chunk.
  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      // NormalizeDispatchGroupSize reshapes an oversized 1-D dispatch to a 2-D grid that rounds up,
      // so a large batch_size can produce workgroups past the last row.
      << "  if (b >= uniforms.batch_size) { return; }\n"
      << "  let row_base = b * uniforms.state_length;\n"
      << "  for (var chunk = 0u; chunk < uniforms.state_length; chunk += workgroup_size_x) {\n"
      << "    let slot = chunk + local_idx;\n"
      << "    var token = uniforms.pad_id;\n"
      << "    if (slot < uniforms.state_length) {\n";
  if (has_input_ids_) {
    shader.MainFunctionBody()
        << "      if (slot + uniforms.sequence_length >= uniforms.state_length) {\n"
        << "        let source_t = slot + uniforms.sequence_length - uniforms.state_length;\n"
        << "        token = " << input_ids->GetByOffset("b * uniforms.sequence_length + source_t") << ";\n"
        << "      }\n";
  }
  if (has_past_ids_) {
    shader.MainFunctionBody()
        << "      if (slot + uniforms.sequence_length < uniforms.state_length) {\n"
        << "        token = " << past_ids->GetByOffset("b * uniforms.state_length + slot + uniforms.sequence_length")
        << ";\n"
        << "      }\n";
  }
  shader.MainFunctionBody()
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "    if (slot < uniforms.state_length) {\n"
      << "      " << present_ids.SetByOffset("row_base + slot", "token") << "\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
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
}

Status NGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* past_ids = context.Input(3);
  const auto& input_shape = input_ids->Shape();
  ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "input_ids must have rank 2");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] == max_ngram_size_,
                    "multipliers must have shape (max_ngram_size)");
  const int64_t num_heads = (max_ngram_size_ - 1) * n_head_per_ngram_;
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");
  const int64_t batch_size = input_shape[0];
  const int64_t sequence_length = input_shape[1];
  // An n-gram window reaches this many positions before the current token.
  const int64_t state_length = max_ngram_size_ - 1;
  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
  }
  const bool has_past_ids = past_ids != nullptr;

  auto* output = context.Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  auto* present_ids = context.Output(1, TensorShape({batch_size, state_length}));

  // The hash program reads past_ids and the present program writes present_ids, so when the caller
  // aliases the two the hash program must be queued first.
  const int64_t total = input_shape.Size();
  if (total > 0) {
    NGramHashMappingProgram program{has_past_ids};
    program.CacheHint(has_past_ids)
        .AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                    {multipliers, ProgramTensorMetadataDependency::None},
                    {vocab_sizes, ProgramTensorMetadataDependency::None}});
    if (has_past_ids) {
      program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    program.AddOutput({output, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(total) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(total)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(max_ngram_size_)},
                              {onnxruntime::narrow<uint32_t>(n_head_per_ngram_)},
                              {onnxruntime::narrow<int32_t>(pad_id_)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(program));
  }

  if (present_ids != nullptr && batch_size * state_length > 0) {
    // WebGPU rejects zero-sized storage buffer bindings, so an empty input_ids tensor must not be
    // bound. When sequence_length == 0 every present slot comes from history (or pad_id), so the
    // input_ids branch of the shader is dead anyway.
    const bool has_input_ids = sequence_length > 0;
    NGramPresentIdsProgram present_program{has_input_ids, has_past_ids};
    present_program.CacheHint(has_input_ids, has_past_ids);
    if (has_input_ids) {
      present_program.AddInput({input_ids, ProgramTensorMetadataDependency::None});
    }
    if (has_past_ids) {
      present_program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    // One workgroup per batch row, so the shader can use a workgroup barrier to order its reads
    // against its writes.
    present_program.AddOutput({present_ids, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize(onnxruntime::narrow<uint32_t>(batch_size))
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(batch_size)},
                              {onnxruntime::narrow<uint32_t>(sequence_length)},
                              {onnxruntime::narrow<uint32_t>(state_length)},
                              {onnxruntime::narrow<int32_t>(pad_id_)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(present_program));
  }

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
