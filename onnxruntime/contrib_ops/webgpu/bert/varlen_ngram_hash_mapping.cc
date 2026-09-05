// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/varlen_ngram_hash_mapping.h"

#include "contrib_ops/cpu/bert/engram_helper.h"
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

Status VarlenNGramValidateCuSeqlensProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& cu_seqlens = shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const auto& is_valid = shader.AddOutput("is_valid", ShaderUsage::UseUniform);

  // Single-invocation scan: per-workgroup local checks elsewhere (start < end, end <= total_tokens)
  // only look at one request's own pair of offsets and cannot detect a globally non-monotonic
  // cu_seqlens array (see the class comment in the header for the concrete race).
  shader.MainFunctionBody()
      << "  if (global_idx != 0u) { return; }\n"
      << "  var valid = 1u;\n"
      << "  if (uniforms.batch_size == 0u) {\n"
      << "    valid = 0u;\n"
      << "  } else {\n"
      << "    if (" << cu_seqlens.GetByOffset("0u") << " != 0) { valid = 0u; }\n"
      << "    for (var i = 0u; i < uniforms.batch_size; i++) {\n"
      << "      let start = " << cu_seqlens.GetByOffset("i") << ";\n"
      << "      let end = " << cu_seqlens.GetByOffset("i + 1u") << ";\n"
      << "      if (start < 0 || start >= end || u32(end) > uniforms.total_tokens) { valid = 0u; }\n"
      << "    }\n"
      << "    if (u32(" << cu_seqlens.GetByOffset("uniforms.batch_size") << ") != uniforms.total_tokens) {\n"
      << "      valid = 0u;\n"
      << "    }\n"
      << "  }\n"
      << "  " << is_valid.SetByOffset("0u", "valid") << "\n";
  return Status::OK();
}

Status VarlenNGramFillDefaultProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& is_valid = shader.AddInput("is_valid", ShaderUsage::UseUniform);
  const ShaderVariableHelper* eos_token_id = nullptr;
  if (has_present_ids_ && has_eos_token_id_) {
    eos_token_id = &shader.AddInput("eos_token_id", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
  const ShaderVariableHelper* present_ids = nullptr;
  if (has_present_ids_) {
    present_ids = &shader.AddOutput("present_ids", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody()
      << "  if (" << is_valid.GetByOffset("0u") << " != 0u) { return; }\n"
      << "  if (global_idx < uniforms.output_count) {\n"
      << "    " << output.SetByOffset("global_idx", "0i") << "\n"
      << "  }\n";
  if (has_present_ids_) {
    const std::string missing_history_value =
        has_eos_token_id_ ? eos_token_id->GetByOffset("0u") : "uniforms.pad_id";
    shader.MainFunctionBody()
        << "  if (global_idx < uniforms.present_count) {\n"
        << "    " << present_ids->SetByOffset("global_idx", missing_history_value) << "\n"
        << "  }\n";
  }
  return Status::OK();
}

Status VarlenNGramHashMappingProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input_ids = shader.AddInput("input_ids", ShaderUsage::UseUniform);
  const auto& multipliers = shader.AddInput("multipliers", ShaderUsage::UseUniform);
  const auto& vocab_sizes = shader.AddInput("vocab_sizes", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const auto& is_valid = shader.AddInput("is_valid", ShaderUsage::UseUniform);
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

  // Every workgroup bails out immediately (leaving VarlenNGramFillDefaultProgram's defaults in
  // place) unless VarlenNGramValidateCuSeqlensProgram found cu_seqlens globally valid. The two
  // programs' writes are therefore mutually exclusive, so the output is always fully and
  // unambiguously written regardless of launch order (see the header comment for why per-workgroup
  // local checks alone are not enough).
  shader.MainFunctionBody()
      << "  if (" << is_valid.GetByOffset("0u") << " == 0u) { return; }\n"
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size) { return; }\n"
      << "  let start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      // Defense-in-depth: is_valid already guarantees this globally, but keep the local check too
      // in case a future change narrows what is_valid covers.
      << "  if (start < 0 || start >= end || u32(end) > uniforms.total_tokens) { return; }\n"
      << "  let local_length = u32(end - start);\n"
      << "  let state_length = uniforms.max_ngram_size - 1u;\n"
      << "  let num_heads = state_length * uniforms.n_head_per_ngram;\n"
      << "  let past_base = b * state_length;\n";
  if (has_eos_token_id_) {
    shader.MainFunctionBody()
        << "  let missing_history_value = " << eos_token_id->GetByOffset("0u") << ";\n";
  } else {
    shader.MainFunctionBody() << "  let missing_history_value = uniforms.pad_id;\n";
  }
  shader.MainFunctionBody()
      << "  for (var out_h = local_idx; out_h < num_heads; out_h += workgroup_size_x) {\n"
      << "    let n = out_h / uniforms.n_head_per_ngram + 2u;\n"
      << "    let mod_value = " << vocab_sizes.GetByOffset("out_h") << ";\n"
      << "    var has_reset = false;\n"
      << "    var last_reset = 0u;\n";
  if (reset_on_eos_ && has_eos_token_id_) {
    shader.MainFunctionBody()
        << "    for (var i = 0u; i < state_length; i++) {\n"
        << "      var history_token = missing_history_value;\n";
    if (has_past_ids_) {
      shader.MainFunctionBody()
          << "      history_token = " << past_ids->GetByOffset("past_base + i") << ";\n";
    }
    shader.MainFunctionBody()
        << "      if (history_token == missing_history_value) { has_reset = true; last_reset = i; }\n"
        << "    }\n";
  }
  shader.MainFunctionBody()
      << "    for (var t = 0u; t < local_length; t++) {\n"
      << "      let idx = state_length + t;\n";
  if ((reset_on_eos_ && has_eos_token_id_) || has_segment_ids_) {
    shader.MainFunctionBody()
        << "      if (t > 0u) {\n"
        << "        var boundary = false;\n";
    if (reset_on_eos_ && has_eos_token_id_) {
      shader.MainFunctionBody()
          << "        boundary = " << input_ids.GetByOffset("u32(start) + t - 1u")
          << " == missing_history_value;\n";
    }
    if (has_segment_ids_) {
      shader.MainFunctionBody()
          << "        boundary = boundary || "
          << segment_ids->GetByOffset("u32(start) + t") << " != "
          << segment_ids->GetByOffset("u32(start) + t - 1u") << ";\n";
    }
    shader.MainFunctionBody()
        << "        if (boundary) { has_reset = true; last_reset = idx - 1u; }\n"
        << "      }\n";
  }
  shader.MainFunctionBody()
      << "      var mix = 0i;\n"
      << "      for (var k = 0u; k < n; k++) {\n"
      << "        let source = idx - k;\n"
      << "        var token = missing_history_value;\n"
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
      << "        if (has_reset && source <= last_reset) { token = missing_history_value; }\n"
      << "        let product = token * " << multipliers.GetByOffset("k") << ";\n"
      << "        if (k == 0u) { mix = product; } else { mix = mix ^ product; }\n"
      << "      }\n"
      << "      var result = 0i;\n"
      << "      if (mod_value > 0i) {\n"
      << "        result = positive_mod(mix, mod_value);\n";
  if (has_head_offsets_) {
    shader.MainFunctionBody()
        << "        result = result + " << head_offsets->GetByOffset("out_h") << ";\n";
  }
  shader.MainFunctionBody()
      << "      }\n"
      << "      " << output.SetByOffset("(u32(start) + t) * num_heads + out_h", "result") << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status VarlenNGramPresentIdsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& cu_seqlens = shader.AddInput("cu_seqlens", ShaderUsage::UseUniform);
  const auto& is_valid = shader.AddInput("is_valid", ShaderUsage::UseUniform);
  const ShaderVariableHelper* eos_token_id = nullptr;
  if (has_eos_token_id_) {
    eos_token_id = &shader.AddInput("eos_token_id", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* input_ids = nullptr;
  if (has_input_ids_) {
    input_ids = &shader.AddInput("input_ids", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* past_ids = nullptr;
  if (has_past_ids_) {
    past_ids = &shader.AddInput("past_ids", ShaderUsage::UseUniform);
  }
  const auto& present_ids = shader.AddOutput("present_ids", ShaderUsage::UseUniform);

  // Left in place (rather than overwritten) by VarlenNGramFillDefaultProgram's pad_id defaults
  // unless VarlenNGramValidateCuSeqlensProgram found cu_seqlens globally valid; see
  // VarlenNGramHashMappingProgram for why a per-request local check alone is not enough.
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  if (" << is_valid.GetByOffset("0u") << " == 0u) { return; }\n"
      << "  let slot = global_idx % uniforms.state_length;\n"
      << "  let b = global_idx / uniforms.state_length;\n"
      << "  let start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  var token = ";
  if (has_eos_token_id_) {
    shader.MainFunctionBody() << eos_token_id->GetByOffset("0u");
  } else {
    shader.MainFunctionBody() << "uniforms.pad_id";
  }
  shader.MainFunctionBody()
      << ";\n"
      // Defense-in-depth: is_valid already guarantees this globally.
      << "  if (start >= 0 && start < end && u32(end) <= uniforms.total_tokens) {\n"
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
  reset_on_eos_ = info.GetAttrOrDefault<int64_t>("reset_on_eos", 0) != 0;
}

Status VarlenNGramHashMapping::ComputeInternal(ComputeContext& context) const {
  const auto* input_ids = context.Input(0);
  const auto* multipliers = context.Input(1);
  const auto* vocab_sizes = context.Input(2);
  const auto* cu_seqlens = context.Input(3);
  const auto* past_ids = context.Input(4);
  const auto* head_offsets = context.Input(5);
  const auto* eos_token_id = context.Input(6);
  const auto* segment_ids = context.Input(7);

  ORT_RETURN_IF_NOT(input_ids->Shape().NumDimensions() == 1, "input_ids must have rank 1 (total_tokens)");
  ORT_RETURN_IF_NOT(multipliers->Shape().NumDimensions() == 1 && multipliers->Shape()[0] >= max_ngram_size_,
                    "multipliers must have at least max_ngram_size elements");
  int64_t num_heads = 0;
  ORT_RETURN_IF_NOT(
      onnxruntime::contrib::engram_helper::TryMultiplyDims(max_ngram_size_ - 1, n_head_per_ngram_, num_heads),
      "VarlenNGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram overflows int64_t");
  // Even though max_ngram_size and n_head_per_ngram individually fit the uint32_t uniforms below
  // (WebGpuKernel construction already narrows each of them), their product is recomputed inside the
  // WGSL shader as u32 arithmetic and could itself overflow a uint32_t even when int64_t num_heads
  // does not. Reject that case here instead of letting the shader silently wrap.
  ORT_RETURN_IF_NOT(num_heads <= std::numeric_limits<uint32_t>::max(),
                    "VarlenNGramHashMapping: (max_ngram_size - 1) * n_head_per_ngram must fit in a uint32_t "
                    "for the WebGPU execution provider");
  ORT_RETURN_IF_NOT(vocab_sizes->Shape() == TensorShape({num_heads}),
                    "vocab_sizes must have shape ((max_ngram_size - 1) * n_head_per_ngram)");

  const auto& cu_seqlens_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_seqlens_shape.NumDimensions() == 1 && cu_seqlens_shape[0] >= 2,
                    "cumulative_sequence_length must have rank 1 with at least 2 elements");
  // batch_size = cu_seqlens.Shape()[0] - 1. This is a shape-only computation; the offset values
  // themselves stay on the device and are validated inside the shader.
  const int64_t batch_size = cu_seqlens_shape[0] - 1;

  const int64_t total_tokens = input_ids->Shape()[0];
  const int64_t state_length = max_ngram_size_ - 1;
  ORT_RETURN_IF_NOT(max_ngram_size_ <= std::numeric_limits<uint32_t>::max() &&
                        n_head_per_ngram_ <= std::numeric_limits<uint32_t>::max() &&
                        batch_size <= std::numeric_limits<uint32_t>::max() &&
                        total_tokens <= std::numeric_limits<uint32_t>::max() &&
                        state_length <= std::numeric_limits<uint32_t>::max(),
                    "VarlenNGramHashMapping: dimensions must fit in WebGPU uint32_t indices");
  ORT_RETURN_IF_NOT(total_tokens >= batch_size,
                    "total_tokens must be at least batch_size because every request must contain a token");

  int64_t output_count = 0;
  int64_t present_count = 0;
  ORT_RETURN_IF_NOT(onnxruntime::contrib::engram_helper::TryMultiplyDims(total_tokens, num_heads, output_count) &&
                        onnxruntime::contrib::engram_helper::TryMultiplyDims(batch_size, state_length, present_count),
                    "VarlenNGramHashMapping: output dimensions overflow int64_t");
  ORT_RETURN_IF_NOT(num_heads <= std::numeric_limits<uint32_t>::max() &&
                        output_count <= std::numeric_limits<uint32_t>::max() &&
                        present_count <= std::numeric_limits<uint32_t>::max(),
                    "VarlenNGramHashMapping: output dimensions must fit in WebGPU uint32_t indices");
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
    ORT_RETURN_IF_NOT(segment_ids->Shape() == TensorShape({total_tokens}),
                      "segment_ids must have shape (total_tokens)");
  }
  const bool has_past_ids = past_ids != nullptr;
  const bool has_head_offsets = head_offsets != nullptr;
  const bool has_eos_token_id = eos_token_id != nullptr;
  const bool has_segment_ids = segment_ids != nullptr;

  auto* output = context.Output(0, TensorShape({total_tokens, num_heads}));
  auto* present_ids = context.Output(1, TensorShape({batch_size, state_length}));
  const bool has_present_ids = present_ids != nullptr;
  if (!has_present_ids) {
    present_count = 0;
  }

  if (batch_size == 0) {
    return Status::OK();
  }

  // Establish global monotonicity of cu_seqlens exactly once, before any output-producing program
  // runs. VarlenNGramHashMappingProgram and VarlenNGramPresentIdsProgram only ever access
  // input_ids/past_ids/write outputs once this flag confirms the whole array is well-formed; when
  // it is not, VarlenNGramFillDefaultProgram (and the pad_id fallback embedded directly in
  // VarlenNGramPresentIdsProgram) supplies deterministic defaults instead. See
  // VarlenNGramValidateCuSeqlensProgram's declaration for why per-request local range checks alone
  // cannot detect this.
  Tensor is_valid = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape({1}));
  VarlenNGramValidateCuSeqlensProgram validate_program{};
  validate_program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None})
      .AddOutput({&is_valid, ProgramTensorMetadataDependency::None})
      .SetDispatchGroupSize(1)
      .AddUniformVariables({{onnxruntime::narrow<uint32_t>(batch_size)},
                            {onnxruntime::narrow<uint32_t>(total_tokens)}});
  ORT_RETURN_IF_ERROR(context.RunProgram(validate_program));

  if (output_count > 0 || present_count > 0) {
    const bool fill_uses_eos_token_id = has_present_ids && has_eos_token_id;
    VarlenNGramFillDefaultProgram fill_program{has_present_ids, fill_uses_eos_token_id};
    fill_program.CacheHint(has_present_ids, fill_uses_eos_token_id);
    fill_program.AddInput({&is_valid, ProgramTensorMetadataDependency::None});
    if (fill_uses_eos_token_id) {
      fill_program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
    }
    fill_program.AddOutput({output, ProgramTensorMetadataDependency::None});
    if (has_present_ids) {
      fill_program.AddOutput({present_ids, ProgramTensorMetadataDependency::None});
    }
    const int64_t max_elements = std::max(output_count, present_count);
    fill_program
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(max_elements) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(output_count)},
                              {onnxruntime::narrow<uint32_t>(present_count)},
                              {onnxruntime::narrow<int32_t>(pad_id_)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(fill_program));
  }

  if (present_count > 0) {
    const bool has_input_ids = total_tokens > 0;
    VarlenNGramPresentIdsProgram present_program{has_input_ids, has_past_ids, has_eos_token_id};
    present_program.CacheHint(has_input_ids, has_past_ids, has_eos_token_id);
    present_program.AddInput({cu_seqlens, ProgramTensorMetadataDependency::None});
    present_program.AddInput({&is_valid, ProgramTensorMetadataDependency::None});
    if (has_eos_token_id) {
      present_program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
    }
    if (has_input_ids) {
      present_program.AddInput({input_ids, ProgramTensorMetadataDependency::None});
    }
    if (has_past_ids) {
      present_program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
    }
    present_program.AddOutput({present_ids, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((onnxruntime::narrow<uint32_t>(present_count) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({{onnxruntime::narrow<uint32_t>(present_count)},
                              {onnxruntime::narrow<uint32_t>(state_length)},
                              {onnxruntime::narrow<uint32_t>(batch_size)},
                              {onnxruntime::narrow<uint32_t>(total_tokens)},
                              {onnxruntime::narrow<int32_t>(pad_id_)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(present_program));
  }

  if (total_tokens == 0) {
    return Status::OK();
  }

  VarlenNGramHashMappingProgram program{
      has_past_ids, has_head_offsets, has_eos_token_id, has_segment_ids, reset_on_eos_};
  program.CacheHint(has_past_ids, has_head_offsets, has_eos_token_id, has_segment_ids, reset_on_eos_)
      .AddInputs({{input_ids, ProgramTensorMetadataDependency::None},
                  {multipliers, ProgramTensorMetadataDependency::None},
                  {vocab_sizes, ProgramTensorMetadataDependency::None},
                  {cu_seqlens, ProgramTensorMetadataDependency::None},
                  {&is_valid, ProgramTensorMetadataDependency::None}});
  if (has_past_ids) {
    program.AddInput({past_ids, ProgramTensorMetadataDependency::None});
  }
  if (has_head_offsets) {
    program.AddInput({head_offsets, ProgramTensorMetadataDependency::None});
  }
  if (has_eos_token_id) {
    program.AddInput({eos_token_id, ProgramTensorMetadataDependency::None});
  }
  if (has_segment_ids) {
    program.AddInput({segment_ids, ProgramTensorMetadataDependency::None});
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
