// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/turbo_quant_hadamard.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/wgsl_templates/wgsl_gen.h"

#include <cmath>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status TurboQuantRotateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  // Input/output: BNSH tensor to rotate in-place.
  // The shader applies the Fast Walsh-Hadamard Transform (FWHT) to the last dimension.
  // Each workgroup handles one (batch, head, seq_pos) vector of head_size elements.
  // Workgroup size = head_size.

  // Use the tensor only as output (read-write) for in-place operation.
  // Reading from and writing to the same storage buffer works when bound as read_write.
  shader.AddOutput("data", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t log2_head_size = static_cast<uint32_t>(std::log2(head_size_));

  // Declare workgroup shared memory at module scope
  shader.AdditionalImplementation() << "var<workgroup> tq_shared_data : array<data_element_t, " << head_size_ << ">;\n";

  shader.MainFunctionBody() << "  let num_heads = " << num_heads_ << "u;\n"
                            << "  let head_size = " << head_size_ << "u;\n"
                            << "  let log2_head_size = " << log2_head_size << "u;\n"
                            << "\n"
                            << "  // Decode workgroup_idx -> (batch, head, token_in_range)\n"
                            << "  let token_in_range = workgroup_idx % uniforms.sequence_length;\n"
                            << "  let head_idx = (workgroup_idx / uniforms.sequence_length) % num_heads;\n"
                            << "  let batch_idx = workgroup_idx / (uniforms.sequence_length * num_heads);\n"
                            << "  let seq_pos = uniforms.start_token + token_in_range;\n"
                            << "\n"
                            << "  // Compute offset into BNSH tensor\n"
                            << "  // Shape: [batch, num_heads, present_sequence_length, head_size]\n"
                            << "  let kv_head_idx = head_idx / uniforms.n_reps;\n"
                            << "  let base_offset = (batch_idx * num_heads + kv_head_idx) * uniforms.present_sequence_length * head_size + seq_pos * head_size;\n"
                            << "\n"
                            << "  // Load vector element into shared memory\n"
                            << "  tq_shared_data[local_idx] = data_element_t(data[base_offset + local_idx]);\n"
                            << "  workgroupBarrier();\n"
                            << "\n"
                            << "  // Fast Walsh-Hadamard Transform (in-place, iterative)\n"
                            << "  // After log2(head_size) butterfly stages, tq_shared_data contains H * input / sqrt(head_size)\n"
                            << "  for (var stage = 0u; stage < log2_head_size; stage++) {\n"
                            << "    let half_block = 1u << stage;\n"
                            << "    let block_size = half_block << 1u;\n"
                            << "    let block_id = local_idx / block_size;\n"
                            << "    let idx_in_block = local_idx % block_size;\n"
                            << "    if (idx_in_block < half_block) {\n"
                            << "      let i = block_id * block_size + idx_in_block;\n"
                            << "      let j = i + half_block;\n"
                            << "      let a = tq_shared_data[i];\n"
                            << "      let b = tq_shared_data[j];\n"
                            << "      tq_shared_data[i] = a + b;\n"
                            << "      tq_shared_data[j] = a - b;\n"
                            << "    }\n"
                            << "    workgroupBarrier();\n"
                            << "  }\n"
                            << "\n"
                            << "  // Normalize: divide by sqrt(head_size)\n"
                            << "  let scale = data_element_t(1.0) / data_element_t(" << std::sqrt(static_cast<double>(head_size_)) << ");\n"
                            << "  data[base_offset + local_idx] = tq_shared_data[local_idx] * scale;\n";

  return Status::OK();
}

Status ApplyTurboQuantRotation(onnxruntime::webgpu::ComputeContext& context,
                               Tensor* tensor,
                               uint32_t head_size,
                               uint32_t num_heads,
                               uint32_t batch_size,
                               uint32_t num_tokens,
                               uint32_t present_sequence_length,
                               uint32_t start_token,
                               uint32_t n_reps) {
  uint32_t kv_num_heads = num_heads / n_reps;
  uint32_t num_workgroups = batch_size * kv_num_heads * num_tokens;
  if (num_workgroups == 0) {
    return Status::OK();
  }

  TurboQuantRotateProgram program{head_size, kv_num_heads};
  program.AddOutputs({{tensor, ProgramTensorMetadataDependency::TypeAndRank}});
  program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(head_size)
      .CacheHint(head_size, kv_num_heads)
      .AddUniformVariables({{num_tokens},
                            {present_sequence_length},
                            {start_token},
                            {1u}});  // n_reps=1 for KV heads (no GQA expansion)

  return context.RunProgram(program);
}

Status ApplyTurboQuantInverseRotation(onnxruntime::webgpu::ComputeContext& context,
                                      Tensor* output,
                                      uint32_t head_size,
                                      uint32_t num_heads,
                                      uint32_t batch_size,
                                      uint32_t sequence_length,
                                      uint32_t /*n_reps*/) {
  // Hadamard is self-inverse, so we apply the same FWHT transform.
  // Output is in BSNH format: [batch, sequence_length, num_heads, head_size]
  // Data is contiguous per-head, so treat as flat array of (batch * seq * heads) vectors.

  uint32_t num_workgroups = batch_size * num_heads * sequence_length;
  if (num_workgroups == 0) {
    return Status::OK();
  }

  // For BSNH output: treat as flat array of vectors.
  // output shape: [batch, seq, num_heads, head_size] — contiguous per-head vectors.
  // Each workgroup handles one vector.
  // We set num_heads=1 in the program so the shader treats the data as a flat array:
  //   base_offset = workgroup_idx * head_size

  TurboQuantRotateProgram flat_program{head_size, 1};  // num_heads=1 for flat access
  flat_program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank}});

  flat_program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(head_size)
      .CacheHint(head_size, 1u, true /* is_inverse */)
      .AddUniformVariables({{num_workgroups},    // sequence_length = total workgroups
                            {num_workgroups},    // present_sequence_length = same (flat)
                            {0u},                // start_token = 0
                            {1u}});              // n_reps = 1

  return context.RunProgram(flat_program);
}

// ---------------------------------------------------------------------------
// TurboQuantRotateQuantizeProgram: Fused FWHT rotation + quantization + int4 packing.
// Each workgroup processes one head vector (head_size elements).
// Steps:
//   1. Load from source (BSNH or BNSH) into f32 shared memory
//   2. FWHT butterfly stages (rotation)
//   3. f32 reduction for L2 norm
//   4. Normalize, searchsorted → centroid index (0–15)
//   5. Pack 8 indices per u32 → write as fp16 pairs to dest (BNSH compressed)
//   6. Write f32 norm as bitcast<vec2<f16>> to first 2 fp16 slots
// ---------------------------------------------------------------------------

Status TurboQuantRotateQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("source", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddOutput("dest", ShaderUsage::UseUniform);

  const uint32_t log2_head_size = static_cast<uint32_t>(std::log2(head_size_));

  return WGSL_TEMPLATE_APPLY(shader, "bert/turbo_quant_rotate_quantize.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(head_size, head_size_),
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_ ? 1 : 0),
                             WGSL_TEMPLATE_PARAMETER(log2_head_size, log2_head_size),
                             WGSL_TEMPLATE_PARAMETER(num_heads, num_heads_),
                             WGSL_TEMPLATE_PARAMETER(source_BSNH, source_BSNH_ ? 1 : 0));
}

Status ApplyTurboQuantRotateAndQuantize(onnxruntime::webgpu::ComputeContext& context,
                                        const Tensor* source,
                                        Tensor* dest,
                                        uint32_t head_size,
                                        uint32_t num_heads,
                                        uint32_t batch_size,
                                        uint32_t kv_sequence_length,
                                        uint32_t present_sequence_length,
                                        uint32_t start_token,
                                        uint32_t n_reps,
                                        uint32_t compressed_dim,
                                        bool is_fp16,
                                        bool source_BSNH) {
  uint32_t kv_num_heads = num_heads / n_reps;
  uint32_t num_workgroups = batch_size * kv_num_heads * kv_sequence_length;
  if (num_workgroups == 0) {
    return Status::OK();
  }

  TurboQuantRotateQuantizeProgram program{head_size, kv_num_heads, is_fp16, source_BSNH};
  program.AddInputs({{source, ProgramTensorMetadataDependency::TypeAndRank}});
  program.AddOutputs({{dest, ProgramTensorMetadataDependency::TypeAndRank}});
  program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(head_size)
      .CacheHint(head_size, kv_num_heads, is_fp16, source_BSNH)
      .AddUniformVariables({{kv_sequence_length},
                            {present_sequence_length},
                            {start_token},
                            {1u},
                            {compressed_dim}});

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
