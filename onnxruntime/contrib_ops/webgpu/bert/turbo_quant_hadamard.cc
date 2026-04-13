// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/turbo_quant_hadamard.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

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
// TurboQuantRotateQuantizeProgram: Fused FWHT rotation + pseudo-quantization.
// Each workgroup processes one head vector (head_size elements).
// Steps:
//   1. Load into shared memory
//   2. FWHT butterfly stages (rotation)
//   3. f32 reduction for L2 norm (via shared memory)
//   4. Normalize to unit sphere, searchsorted → centroid index (0–15)
//   5. Write: indices in [0..head_size-3], f32 norm in [head_size-2..head_size-1]
// ---------------------------------------------------------------------------

Status TurboQuantRotateQuantizeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddOutput("data", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  const uint32_t log2_head_size = static_cast<uint32_t>(std::log2(head_size_));
  const double fwht_scale = 1.0 / std::sqrt(static_cast<double>(head_size_));

  // Shared memory for FWHT data and f32 reduction
  shader.AdditionalImplementation()
      << "var<workgroup> tq_shared : array<f32, " << head_size_ << ">;\n"
      << "var<workgroup> tq_sq_shared : array<f32, " << head_size_ << ">;\n"
      << "\n"
      << "// TurboQuant 4-bit codebook: 16 MSE-optimal centroids for unit-sphere coordinates\n"
      << "const TQ_CENTROIDS = array<f32, 16>(\n"
      << "    -0.2377, -0.1809, -0.1419, -0.1104, -0.0829, -0.0578, -0.0342, -0.0113,\n"
      << "     0.0113,  0.0342,  0.0578,  0.0829,  0.1104,  0.1419,  0.1809,  0.2377);\n"
      << "\n"
      << "// 15 interior decision boundaries (we handle the -inf/+inf edges in code)\n"
      << "const TQ_BOUNDARIES = array<f32, 15>(\n"
      << "    -0.2093, -0.1614, -0.1261, -0.0966, -0.0704, -0.0460, -0.0227,\n"
      << "     0.0000,  0.0227,  0.0460,  0.0704,  0.0966,  0.1261,  0.1614,  0.2093);\n"
      << "\n"
      << "fn searchsorted_tq(val: f32) -> u32 {\n"
      << "  // Binary search over 15 boundaries → returns index 0..15\n"
      << "  var lo = 0u;\n"
      << "  var hi = 15u;\n"
      << "  for (var iter = 0u; iter < 4u; iter++) {\n"  // ceil(log2(15)) = 4
      << "    let mid = (lo + hi) >> 1u;\n"
      << "    if (val >= TQ_BOUNDARIES[mid]) {\n"
      << "      lo = mid + 1u;\n"
      << "    } else {\n"
      << "      hi = mid;\n"
      << "    }\n"
      << "  }\n"
      << "  return lo;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let num_heads = " << num_heads_ << "u;\n"
      << "  let head_size = " << head_size_ << "u;\n"
      << "  let log2_head_size = " << log2_head_size << "u;\n"
      << "\n"
      << "  // Decode workgroup_idx -> (batch, head, token_in_range)\n"
      << "  let token_in_range = workgroup_idx % uniforms.sequence_length;\n"
      << "  let head_idx = (workgroup_idx / uniforms.sequence_length) % num_heads;\n"
      << "  let batch_idx = workgroup_idx / (uniforms.sequence_length * num_heads);\n"
      << "  let seq_pos = uniforms.start_token + token_in_range;\n"
      << "\n"
      << "  // BNSH tensor offset\n"
      << "  let kv_head_idx = head_idx / uniforms.n_reps;\n"
      << "  let base_offset = (batch_idx * num_heads + kv_head_idx) * uniforms.present_sequence_length * head_size + seq_pos * head_size;\n"
      << "\n"
      << "  // Step 1: Load into shared memory as f32 for FWHT + accumulation\n"
      << "  tq_shared[local_idx] = f32(data[base_offset + local_idx]);\n"
      << "  workgroupBarrier();\n"
      << "\n"
      << "  // Step 2: FWHT butterfly stages (in f32 for precision)\n"
      << "  for (var stage = 0u; stage < log2_head_size; stage++) {\n"
      << "    let half_block = 1u << stage;\n"
      << "    let block_size = half_block << 1u;\n"
      << "    let block_id = local_idx / block_size;\n"
      << "    let idx_in_block = local_idx % block_size;\n"
      << "    if (idx_in_block < half_block) {\n"
      << "      let i = block_id * block_size + idx_in_block;\n"
      << "      let j = i + half_block;\n"
      << "      let a = tq_shared[i];\n"
      << "      let b = tq_shared[j];\n"
      << "      tq_shared[i] = a + b;\n"
      << "      tq_shared[j] = a - b;\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "\n"
      << "  // Apply FWHT normalization: divide by sqrt(head_size)\n"
      << "  let fwht_scale = f32(" << fwht_scale << ");\n"
      << "  let rotated_val = tq_shared[local_idx] * fwht_scale;\n"
      << "\n"
      << "  // Step 3: Compute L2 norm in f32 via shared memory reduction\n"
      << "  tq_sq_shared[local_idx] = rotated_val * rotated_val;\n"
      << "  workgroupBarrier();\n"
      << "\n"
      << "  // Tree reduction for sum of squares\n"
      << "  for (var stride = head_size >> 1u; stride > 0u; stride >>= 1u) {\n"
      << "    if (local_idx < stride) {\n"
      << "      tq_sq_shared[local_idx] += tq_sq_shared[local_idx + stride];\n"
      << "    }\n"
      << "    workgroupBarrier();\n"
      << "  }\n"
      << "\n"
      << "  let norm = sqrt(max(tq_sq_shared[0], 1e-12));\n"
      << "  let inv_norm = 1.0 / norm;\n"
      << "\n"
      << "  // Step 4: Normalize and quantize\n"
      << "  if (local_idx < head_size - 2u) {\n"
      << "    let unit_val = rotated_val * inv_norm;\n"
      << "    let idx = searchsorted_tq(unit_val);\n"
      << "    data[base_offset + local_idx] = data_element_t(idx);\n"
      << "  }\n"
      << "\n"
      << "  // Step 5: Store f32 norm in the last 2 element slots via bitcast\n"
      << "  if (local_idx == 0u) {\n";

  if (is_fp16_) {
    // fp16: bitcast f32 norm to vec2<f16>, store each component
    shader.MainFunctionBody()
        << "    let norm_f16x2 = bitcast<vec2<f16>>(f32(norm));\n"
        << "    data[base_offset + head_size - 2u] = norm_f16x2.x;\n"
        << "    data[base_offset + head_size - 1u] = norm_f16x2.y;\n";
  } else {
    // fp32: bitcast f32 norm bits into 2 f32 slots (lo16 and hi16 as reinterpreted u32)
    shader.MainFunctionBody()
        << "    let norm_bits = bitcast<u32>(f32(norm));\n"
        << "    let lo16 = norm_bits & 0xFFFFu;\n"
        << "    let hi16 = (norm_bits >> 16u) & 0xFFFFu;\n"
        << "    data[base_offset + head_size - 2u] = bitcast<data_element_t>(lo16);\n"
        << "    data[base_offset + head_size - 1u] = bitcast<data_element_t>(hi16);\n";
  }

  shader.MainFunctionBody()
      << "  }\n";

  return Status::OK();
}

Status ApplyTurboQuantRotateAndQuantize(onnxruntime::webgpu::ComputeContext& context,
                                        Tensor* tensor,
                                        uint32_t head_size,
                                        uint32_t num_heads,
                                        uint32_t batch_size,
                                        uint32_t num_tokens,
                                        uint32_t present_sequence_length,
                                        uint32_t start_token,
                                        uint32_t n_reps,
                                        bool is_fp16) {
  uint32_t kv_num_heads = num_heads / n_reps;
  uint32_t num_workgroups = batch_size * kv_num_heads * num_tokens;
  if (num_workgroups == 0) {
    return Status::OK();
  }

  TurboQuantRotateQuantizeProgram program{head_size, kv_num_heads, is_fp16};
  program.AddOutputs({{tensor, ProgramTensorMetadataDependency::TypeAndRank}});
  program.SetDispatchGroupSize(num_workgroups)
      .SetWorkgroupSize(head_size)
      .CacheHint(head_size, kv_num_heads, is_fp16)
      .AddUniformVariables({{num_tokens},
                            {present_sequence_length},
                            {start_token},
                            {1u}});  // n_reps=1 for KV heads

  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
