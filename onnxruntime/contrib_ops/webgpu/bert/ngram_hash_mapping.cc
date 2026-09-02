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

  shader.AdditionalImplementation() << engram_helper::kPositiveModWgsl;
  shader.AdditionalImplementation()
      << "fn combined_value(b: i32, history_length: i32, eos_value: i32, idx: i32) -> i32 {\n"
      << "  if (idx < history_length) {\n";
  if (has_past_ids_) {
    shader.AdditionalImplementation()
        << "    return " << past_ids->GetByOffset("b * history_length + idx") << ";\n";
  } else {
    shader.AdditionalImplementation() << "    return eos_value;\n";
  }
  shader.AdditionalImplementation()
      << "  }\n"
      << "  return " << input_ids.GetByOffset("b * i32(uniforms.sequence_length) + idx - history_length") << ";\n"
      << "}\n";

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let history_length = i32(uniforms.max_ngram_size - 1u);\n"
      << "  let sequence_length = i32(uniforms.sequence_length);\n";
  if (has_eos_token_id_) {
    shader.MainFunctionBody() << "  let eos_value = " << eos_token_id->GetByOffset("0") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let eos_value = uniforms.pad_id;\n";
  }
  const bool do_reset = has_eos_token_id_ && reset_on_eos_;

  shader.MainFunctionBody()
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
        << "    if (!boundary && j >= history_length) {\n"
        << "      let tj = j - history_length;\n"
        << "      if (" << segment_ids->GetByOffset("input_base + tj") << " != "
        << segment_ids->GetByOffset("input_base + tj + 1") << ") {\n"
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

Status NGramPresentIdsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* input_ids = nullptr;
  if (has_input_ids_) {
    input_ids = &shader.AddInput("input_ids", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_ && !past_aliases_present_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* eos_token_id = nullptr;
  if (has_eos_token_id_) {
    eos_token_id = &shader.AddInput("eos_token_id", ShaderUsage::UseUniform);
  }
  const auto& present_ids = shader.AddOutput("present_ids", ShaderUsage::UseUniform);
  const ShaderVariableHelper* history = past_aliases_present_ ? &present_ids : past_ids;

  if (has_eos_token_id_) {
    shader.MainFunctionBody() << "  let missing_history_value = " << eos_token_id->GetByOffset("0") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let missing_history_value = uniforms.pad_id;\n";
  }
  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size) { return; }\n"
      << "  let row_base = b * uniforms.state_length;\n"
      << "  for (var chunk = 0u; chunk < uniforms.state_length; chunk += workgroup_size_x) {\n"
      << "    let slot = chunk + local_idx;\n"
      << "    var token = missing_history_value;\n"
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
        << "        token = " << history->GetByOffset("b * uniforms.state_length + slot + uniforms.sequence_length")
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
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 0);
}

Status NGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* past_ids = context.Input(3);
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
  const int64_t state_length = max_ngram_size_ - 1;

  if (past_ids != nullptr) {
    ORT_RETURN_IF_NOT(past_ids->Shape() == TensorShape({batch_size, state_length}),
                      "past_ids must have shape (batch_size, max_ngram_size - 1)");
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
  const bool has_past_ids = past_ids != nullptr;
  const bool has_eos_token_id = eos_token_id != nullptr;

  auto* output = context.Output(0, TensorShape({batch_size, sequence_length, num_heads}));
  auto* present_ids = context.Output(1, TensorShape({batch_size, state_length}));

  const int64_t total = input_shape.Size();
  if (total > 0) {
    NGramHashMappingProgram program{has_past_ids, head_offsets != nullptr, has_eos_token_id,
                                    segment_ids != nullptr, reset_on_eos_ != 0};
    program.CacheHint(has_past_ids, head_offsets != nullptr, has_eos_token_id,
                      segment_ids != nullptr, reset_on_eos_ != 0)
        .AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                    {multipliers, ProgramTensorMetadataDependency::None},
                    {vocab_sizes, ProgramTensorMetadataDependency::None}});
    if (has_past_ids) {
      program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    if (head_offsets != nullptr) {
      program.AddInput({head_offsets, ProgramTensorMetadataDependency::None});
    }
    if (has_eos_token_id) {
      program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
    }
    if (segment_ids != nullptr) {
      program.AddInput({segment_ids, ProgramTensorMetadataDependency::None});
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
    const bool has_input_ids = sequence_length > 0;
    const bool past_aliases_present = has_past_ids && past_ids->DataRaw() == present_ids->DataRaw();
    NGramPresentIdsProgram present_program{has_input_ids, has_past_ids, has_eos_token_id, past_aliases_present};
    present_program.CacheHint(has_input_ids, has_past_ids, has_eos_token_id, past_aliases_present);
    if (has_input_ids) {
      present_program.AddInput({input_ids, ProgramTensorMetadataDependency::None});
    }
    if (has_past_ids && !past_aliases_present) {
      present_program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    if (has_eos_token_id) {
      present_program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
    }
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
