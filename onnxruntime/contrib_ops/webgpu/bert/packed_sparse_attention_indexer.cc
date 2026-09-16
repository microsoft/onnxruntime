// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// WebGPU implementation of com.microsoft.PackedSparseAttentionIndexer. Like its dense
// SparseAttentionIndexer counterpart, every program below is single-invocation-per-row
// (`if (... || local_idx != 0u) { return; }`) and recomputes reductions on demand rather than
// staging per-row intermediates in workgroup memory; this keeps every kernel correct without
// requiring WGSL arrays sized by a runtime (uniform) head_size, which WGSL does not support for
// function-local variables. See docs/contrib_ops/webgpu/packed_sparse_attention_indexer.md and
// packed_sparse_attention_indexer_impl.cu (the CUDA implementation) for the full operator
// contract and the device-side metadata-safety argument, which apply unchanged here: every
// per-request quantity is read directly from device buffers inside the shader (never on the
// host), and values are clamped into the fixed-capacity range before use so malformed metadata
// can never cause an out-of-bounds access.

#include "contrib_ops/webgpu/bert/packed_sparse_attention_indexer.h"

#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace psai = onnxruntime::contrib::packed_sparse_attention_indexer;

ONNX_OPERATOR_KERNEL_EX(
    PackedSparseAttentionIndexer,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    PackedSparseAttentionIndexer);

namespace {

constexpr uint32_t kWorkgroupSize = 64;

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "PackedSparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "PackedSparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

uint32_t ToUint32(int64_t value) { return onnxruntime::narrow<uint32_t>(value); }

struct RotaryCacheShape {
  bool batched;
  int64_t max_rotary_length;
  int64_t rotary_width;
};

Status CheckRotaryCache(const Tensor* cos_cache, const Tensor* sin_cache, int64_t batch_size,
                        RotaryCacheShape& out) {
  ORT_RETURN_IF(cos_cache == nullptr, "PackedSparseAttentionIndexer: cos_cache is required");
  const auto& cos_shape = cos_cache->Shape();
  out.batched = cos_shape.NumDimensions() == 3;
  ORT_RETURN_IF_NOT(
      (out.batched && cos_shape[0] == batch_size && cos_shape[1] > 0) ||
          (cos_shape.NumDimensions() == 2 && cos_shape[0] > 0),
      "PackedSparseAttentionIndexer: cos_cache must have shape (max_position, rotary_width) or "
      "(batch_size, max_position, rotary_width), got ",
      cos_shape.ToString());
  out.max_rotary_length = out.batched ? cos_shape[1] : cos_shape[0];
  out.rotary_width = out.batched ? cos_shape[2] : cos_shape[1];
  ORT_RETURN_IF_NOT(out.max_rotary_length > 0 && out.rotary_width > 0,
                    "PackedSparseAttentionIndexer: invalid cos_cache shape ", cos_shape.ToString());
  ORT_RETURN_IF_NOT(sin_cache != nullptr && sin_cache->Shape() == cos_shape,
                    "PackedSparseAttentionIndexer: sin_cache must have the same shape as cos_cache");
  return Status::OK();
}

}  // namespace

Status PackedSparseAttentionIndexerCopyProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseUniform);
  const auto& dst = shader.AddOutput("dst", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  " << dst.SetByOffset("global_idx", "dst_element_t(" + src.GetByOffset("global_idx") + ")") << "\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerQsaUpdateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cumulative_sequence_lengths", ShaderUsage::UseUniform);
  const auto& past_kv_buffer = shader.AddInput("past_kv_buffer", ShaderUsage::UseUniform);
  const auto& past_state_lengths = shader.AddInput("past_state_lengths", ShaderUsage::UseUniform);
  const auto& present_key_state =
      shader.AddOutput("present_key_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_kv_buffer =
      shader.AddOutput("present_kv_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_state_lengths = shader.AddOutput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& overflow_flags = shader.AddOutput("overflow_flags", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn extended_key(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * uniforms.head_size + d;\n"
      << "    return f32(" << past_kv_buffer.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * uniforms.head_size + d;\n"
      << "  return f32(" << key.GetByOffset("idx2") << ");\n"
      << "}\n"
      << "fn pooled(b: u32, k: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  var sum = 0.0;\n"
      << "  for (var t = 0u; t < uniforms.compress_ratio; t++) {\n"
      << "    sum += extended_key(b, k * i32(uniforms.compress_ratio) + i32(t), old_buf_len, req_start, d);\n"
      << "  }\n"
      << "  return sum / f32(uniforms.compress_ratio);\n"
      << "}\n"
      << "fn normalized(b: u32, k: i32, old_buf_len: i32, req_start: i32, d: u32) -> f32 {\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var c = 0u; c < uniforms.head_size; c++) {\n"
      << "    let value = pooled(b, k, old_buf_len, req_start, c);\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  return pooled(b, k, old_buf_len, req_start, d) * "
         "inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon) * f32("
      << norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn rotated(b: u32, k: i32, old_buf_len: i32, req_start: i32, old_key_len: i32, d: u32) -> f32 {\n"
      << "  let value = normalized(b, k, old_buf_len, req_start, d);\n"
      << "  if (d >= uniforms.rotary_width) { return value; }\n"
      << "  let half = uniforms.rotary_width / 2u;\n"
      << "  let pair_d = select(d - half, d + half, d < half);\n"
      << "  let sign = select(1.0, -1.0, d < half);\n"
      << "  let paired = sign * normalized(b, k, old_buf_len, req_start, pair_d);\n"
      << "  let entry = old_key_len + k;\n"
      << "  let position = clamp_position(entry * i32(uniforms.compress_ratio));\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width + d;\n";
  } else {
    shader.AdditionalImplementation() << "  let cache = position * uniforms.rotary_width + d;\n";
  }
  shader.AdditionalImplementation()
      << "  return value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size || local_idx != 0u) { return; }\n"
      << "  let req_start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let req_end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  let req_len = max(req_end - req_start, 0);\n"
      << "  let old_key_len = clamp(" << past_state_lengths.GetByOffset("b * 2u")
      << ", 0, i32(uniforms.state_capacity));\n"
      << "  let old_buf_len = clamp(" << past_state_lengths.GetByOffset("b * 2u + 1u")
      << ", 0, i32(uniforms.compress_ratio) - 1);\n"
      << "  let pending = old_buf_len + req_len;\n"
      << "  let full_new_block_count = pending / i32(uniforms.compress_ratio);\n"
      << "  let capacity_left = max(i32(uniforms.state_capacity) - old_key_len, 0);\n"
      << "  let new_block_count = min(full_new_block_count, capacity_left);\n"
      << "  let overflowed = new_block_count < full_new_block_count;\n"
      << "  let new_buf_len = select(pending % i32(uniforms.compress_ratio), 0, overflowed);\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u", "old_key_len + new_block_count") << "\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u + 1u", "new_buf_len") << "\n"
      << "  for (var k = 0; k < new_block_count; k++) {\n"
      << "    let entry = u32(old_key_len + k);\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      let value = rotated(b, k, old_buf_len, req_start, old_key_len, d);\n"
      << "      "
      << present_key_state.SetByOffset("(b * uniforms.state_capacity + entry) * uniforms.head_size + d",
                                       "present_key_state_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n"
      << "  for (var t = 0; t < new_buf_len; t++) {\n"
      << "    let virtual_pos = new_block_count * i32(uniforms.compress_ratio) + t;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      let value = extended_key(b, virtual_pos, old_buf_len, req_start, d);\n"
      << "      "
      << present_kv_buffer.SetByOffset("(b * uniforms.buffer_capacity + u32(t)) * uniforms.head_size + d",
                                       "present_kv_buffer_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerQsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key_state = shader.AddInput("present_key_state", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cumulative_sequence_lengths", ShaderUsage::UseUniform);
  const auto& past_seqlens = shader.AddInput("past_sequence_lengths", ShaderUsage::UseUniform);
  const ShaderVariableHelper* position_ids = nullptr;
  if (has_position_ids_) {
    position_ids = &shader.AddInput("position_ids", ShaderUsage::UseUniform);
  }
  const auto& present_state_lengths = shader.AddInput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& selected_indices = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  const auto& selected_counts = shader.AddOutput("selected_counts", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn batch_of_token(token: u32) -> u32 {\n"
      << "  var b = 0u;\n"
      << "  for (var i = 0u; i < uniforms.batch_size; i++) {\n"
      << "    if (u32(" << cu_seqlens.GetByOffset("i") << ") <= token) { b = i; } else { break; }\n"
      << "  }\n"
      << "  return b;\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn causal_count(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  let p = u32(position);\n"
      << "  let cr = uniforms.compress_ratio;\n"
      << "  return p / cr + select(0u, 1u, p % cr == cr - 1u);\n"
      << "}\n";
  if (has_position_ids_) {
    shader.AdditionalImplementation()
        << "fn abs_position(token: u32, b: u32) -> i32 {\n"
        << "  let raw = " << position_ids->GetByOffset("token", true) << ";\n"
        << "  if ((raw.y & 0x80000000u) != 0u) { return -2147483648; }\n"
        << "  if (raw.y != 0u || raw.x > 2147483647u) { return 2147483647; }\n"
        << "  return i32(raw.x);\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
        << "fn abs_position(token: u32, b: u32) -> i32 {\n"
        << "  let req_start = " << cu_seqlens.GetByOffset("b") << ";\n"
        << "  return " << past_seqlens.GetByOffset("b") << " + (i32(token) - req_start);\n"
        << "}\n";
  }
  shader.AdditionalImplementation()
      << "fn query_value(token: u32, head: u32, d: u32, position: i32, b: u32) -> f32 {\n"
      << "  let base = (token * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = f32(" << query.GetByOffset("base + d") << ");\n"
      << "  if (d >= uniforms.rotary_width) { return value; }\n"
      << "  let half = uniforms.rotary_width / 2u;\n"
      << "  let pair_d = select(d - half, d + half, d < half);\n"
      << "  let sign = select(1.0, -1.0, d < half);\n"
      << "  let paired = sign * f32(" << query.GetByOffset("base + pair_d") << ");\n"
      << "  let position_clamped = clamp_position(position);\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position_clamped) * uniforms.rotary_width + d;\n";
  } else {
    shader.AdditionalImplementation() << "  let cache = position_clamped * uniforms.rotary_width + d;\n";
  }
  shader.AdditionalImplementation()
      << "  value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(token: u32, b: u32, block_index: u32, position: i32) -> f32 {\n"
      << "  let key_base = (b * uniforms.state_capacity + block_index) * uniforms.head_size;\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(token, head, d, position, b) * f32("
      << present_key_state.GetByOffset("key_base + d") << ");\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let token = workgroup_idx;\n"
      << "  if (token >= uniforms.total_tokens || local_idx != 0u) { return; }\n"
      << "  let output_base = token * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected_indices.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let b = batch_of_token(token);\n"
      << "  let key_len_after = u32(" << present_state_lengths.GetByOffset("b * 2u") << ");\n"
      << "  let position = abs_position(token, b);\n"
      << "  let causal = causal_count(position);\n"
      << "  let visible_block_count = min(key_len_after, causal);\n"
      << "  let selected = min(uniforms.block_topk, visible_block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  var emitted_blocks = 0u;\n"
      << "  for (var rank = 0u; rank < selected; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < visible_block_count; candidate++) {\n"
      << "      let score = block_score(token, b, candidate, position);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < "
         "best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    for (var t = 0u; t < uniforms.compress_ratio; t++) {\n"
      << "      "
      << selected_indices.SetByOffset("output_base + rank * uniforms.compress_ratio + t",
                                      "best_index * i32(uniforms.compress_ratio) + i32(t)")
      << "\n"
      << "    }\n"
      << "    emitted_blocks = rank + 1u;\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n"
      << "  let block_start = i32(visible_block_count * uniforms.compress_ratio);\n"
      << "  let natural_tail = select(0, position - block_start + 1, position >= block_start);\n"
      << "  let remaining_capacity = i32(uniforms.capacity) - i32(emitted_blocks * uniforms.compress_ratio);\n"
      << "  var tail_count = min(natural_tail, remaining_capacity);\n"
      << "  tail_count = max(tail_count, 0);\n"
      << "  for (var t = 0; t < tail_count; t++) {\n"
      << "    "
      << selected_indices.SetByOffset("output_base + emitted_blocks * uniforms.compress_ratio + u32(t)",
                                      "block_start + t")
      << "\n"
      << "  }\n"
      << "  "
      << selected_counts.SetByOffset("token", "i32(emitted_blocks * uniforms.compress_ratio) + tail_count")
      << "\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerCsaUpdateProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& position_bias = shader.AddInput("position_bias", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cumulative_sequence_lengths", ShaderUsage::UseUniform);
  const auto& past_kv_buffer = shader.AddInput("past_kv_buffer", ShaderUsage::UseUniform);
  const auto& past_gate_buffer = shader.AddInput("past_gate_buffer", ShaderUsage::UseUniform);
  const auto& past_state_lengths = shader.AddInput("past_state_lengths", ShaderUsage::UseUniform);
  const auto& present_key_state =
      shader.AddOutput("present_key_state", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_kv_buffer =
      shader.AddOutput("present_kv_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_gate_buffer =
      shader.AddOutput("present_gate_buffer", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_state_lengths = shader.AddOutput("present_state_lengths", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn extended_key(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, channel: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * width + channel;\n"
      << "    return f32(" << past_kv_buffer.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * width + channel;\n"
      << "  return f32(" << key.GetByOffset("idx2") << ");\n"
      << "}\n"
      << "fn extended_gate(b: u32, virtual_pos: i32, old_buf_len: i32, req_start: i32, channel: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  if (virtual_pos < old_buf_len) {\n"
      << "    let idx = (b * uniforms.buffer_capacity + u32(virtual_pos)) * width + channel;\n"
      << "    return f32(" << past_gate_buffer.GetByOffset("idx") << ");\n"
      << "  }\n"
      << "  let idx2 = u32(req_start + virtual_pos - old_buf_len) * width + channel;\n"
      << "  return f32(" << gate.GetByOffset("idx2") << ");\n"
      << "}\n"
      << "fn pooled(b: u32, k: i32, old_buf_len: i32, req_start: i32, overlap_length: i32, d: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  let has_previous = k >= 1 || overlap_length >= i32(uniforms.compress_ratio);\n"
      << "  let previous_base = overlap_length + (k - 1) * i32(uniforms.compress_ratio);\n"
      << "  let current_base = overlap_length + k * i32(uniforms.compress_ratio);\n"
      << "  var max_gate = -3.4028234663852886e+38;\n"
      << "  if (has_previous) {\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let v = extended_gate(b, previous_base + i32(slot), old_buf_len, req_start, d) + f32("
      << position_bias.GetByOffset("slot * width + d") << ");\n"
      << "      max_gate = max(max_gate, v);\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let v = extended_gate(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d) + f32("
      << position_bias.GetByOffset("slot * width + uniforms.head_size + d") << ");\n"
      << "    max_gate = max(max_gate, v);\n"
      << "  }\n"
      << "  var denominator = 0.0;\n"
      << "  var accumulator = 0.0;\n"
      << "  if (has_previous) {\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let logit = extended_gate(b, previous_base + i32(slot), old_buf_len, req_start, d) + f32("
      << position_bias.GetByOffset("slot * width + d") << ");\n"
      << "      let weight = exp(logit - max_gate);\n"
      << "      denominator += weight;\n"
      << "      accumulator += weight * extended_key(b, previous_base + i32(slot), old_buf_len, req_start, d);\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let logit = extended_gate(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d) + f32("
      << position_bias.GetByOffset("slot * width + uniforms.head_size + d") << ");\n"
      << "    let weight = exp(logit - max_gate);\n"
      << "    denominator += weight;\n"
      << "    accumulator += weight * extended_key(b, current_base + i32(slot), old_buf_len, req_start, "
         "uniforms.head_size + d);\n"
      << "  }\n"
      << "  return select(0.0, accumulator / denominator, denominator > 0.0);\n"
      << "}\n"
      << "fn clamp_position(position: i32) -> u32 {\n"
      << "  if (position < 0) { return 0u; }\n"
      << "  return min(u32(position), uniforms.max_rotary_length - 1u);\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let b = workgroup_idx;\n"
      << "  if (b >= uniforms.batch_size || local_idx != 0u) { return; }\n"
      << "  let req_start = " << cu_seqlens.GetByOffset("b") << ";\n"
      << "  let req_end = " << cu_seqlens.GetByOffset("b + 1u") << ";\n"
      << "  let req_len = max(req_end - req_start, 0);\n"
      << "  let old_key_len = clamp(" << past_state_lengths.GetByOffset("b * 2u")
      << ", 0, i32(uniforms.state_capacity));\n"
      << "  let old_buf_len = clamp(" << past_state_lengths.GetByOffset("b * 2u + 1u")
      << ", 0, i32(uniforms.buffer_capacity));\n"
      << "  let overlap_length = select(0, i32(uniforms.compress_ratio), old_buf_len >= "
         "i32(uniforms.compress_ratio));\n"
      << "  let leftover_length = old_buf_len - overlap_length;\n"
      << "  let pending = leftover_length + req_len;\n"
      << "  let full_new_window_count = pending / i32(uniforms.compress_ratio);\n"
      << "  let capacity_left = max(i32(uniforms.state_capacity) - old_key_len, 0);\n"
      << "  let new_window_count = min(full_new_window_count, capacity_left);\n"
      << "  let overflowed = new_window_count < full_new_window_count;\n"
      << "  var present_buffer_length = 0;\n"
      << "  var present_buffer_start = 0;\n"
      << "  if (!overflowed) {\n"
      << "    if (new_window_count > 0) {\n"
      << "      present_buffer_length = i32(uniforms.compress_ratio) + pending % i32(uniforms.compress_ratio);\n"
      << "      present_buffer_start = overlap_length + (new_window_count - 1) * i32(uniforms.compress_ratio);\n"
      << "    } else {\n"
      << "      present_buffer_length = old_buf_len + req_len;\n"
      << "      present_buffer_start = 0;\n"
      << "    }\n"
      << "    present_buffer_length = min(present_buffer_length, i32(uniforms.buffer_capacity));\n"
      << "  }\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u", "old_key_len + new_window_count") << "\n"
      << "  " << present_state_lengths.SetByOffset("b * 2u + 1u", "present_buffer_length") << "\n"
      << "  for (var k = 0; k < new_window_count; k++) {\n"
      << "    var square_sum = 0.0;\n"
      << "    for (var c = 0u; c < uniforms.head_size; c++) {\n"
      << "      let v = pooled(b, k, old_buf_len, req_start, overlap_length, c);\n"
      << "      square_sum += v * v;\n"
      << "    }\n"
      << "    let inverse_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "    let entry = u32(old_key_len + k);\n"
      << "    let position = clamp_position(i32(entry) * i32(uniforms.compress_ratio));\n";
  if (cos_cache_batched_) {
    shader.MainFunctionBody()
        << "    let cache_base = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width;\n";
  } else {
    shader.MainFunctionBody() << "    let cache_base = position * uniforms.rotary_width;\n";
  }
  shader.MainFunctionBody()
      << "    let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      var value = pooled(b, k, old_buf_len, req_start, overlap_length, d) * inverse_rms * f32("
      << norm.GetByOffset("d") << ");\n"
      << "      if (d >= rotary_base) {\n"
      << "        let offset = d - rotary_base;\n"
      << "        let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "        let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "        let paired = sign * pooled(b, k, old_buf_len, req_start, overlap_length, pair_d) * "
         "inverse_rms * f32("
      << norm.GetByOffset("pair_d") << ");\n"
      << "        value = value * f32(" << cos_cache.GetByOffset("cache_base + offset / 2u") << ") + paired * f32("
      << sin_cache.GetByOffset("cache_base + offset / 2u") << ");\n"
      << "      }\n"
      << "      "
      << present_key_state.SetByOffset("(b * uniforms.state_capacity + entry) * uniforms.head_size + d",
                                       "present_key_state_element_t(value)")
      << "\n"
      << "    }\n"
      << "  }\n"
      << "  for (var t = 0; t < present_buffer_length; t++) {\n"
      << "    let virtual_pos = present_buffer_start + t;\n"
      << "    for (var c = 0u; c < 2u * uniforms.head_size; c++) {\n"
      << "      let key_value = extended_key(b, virtual_pos, old_buf_len, req_start, c);\n"
      << "      let gate_value = extended_gate(b, virtual_pos, old_buf_len, req_start, c);\n"
      << "      "
      << present_kv_buffer.SetByOffset("(b * uniforms.buffer_capacity + u32(t)) * (2u * uniforms.head_size) + c",
                                       "present_kv_buffer_element_t(key_value)")
      << "\n"
      << "      "
      << present_gate_buffer.SetByOffset(
             "(b * uniforms.buffer_capacity + u32(t)) * (2u * uniforms.head_size) + c",
             "present_gate_buffer_element_t(gate_value)")
      << "\n"
      << "    }\n"
      << "  }\n";
  return Status::OK();
}

Status PackedSparseAttentionIndexerCsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key_state = shader.AddInput("present_key_state", ShaderUsage::UseUniform);
  const auto& head_weights = shader.AddInput("head_weights", ShaderUsage::UseUniform);
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& cu_seqlens = shader.AddInput("cumulative_sequence_lengths", ShaderUsage::UseUniform);
  const auto& present_state_lengths = shader.AddInput("present_state_lengths", ShaderUsage::UseUniform);
  const auto& selected_indices = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  const auto& selected_counts = shader.AddOutput("selected_counts", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn batch_of_token(token: u32) -> u32 {\n"
      << "  var b = 0u;\n"
      << "  for (var i = 0u; i < uniforms.batch_size; i++) {\n"
      << "    if (u32(" << cu_seqlens.GetByOffset("i") << ") <= token) { b = i; } else { break; }\n"
      << "  }\n"
      << "  return b;\n"
      << "}\n"
      << "fn clamped_position(raw: vec2<u32>) -> u32 {\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u) { return 0xffffffffu; }\n"
      << "  return raw.x;\n"
      << "}\n"
      << "fn causal_count(raw: vec2<u32>) -> u32 {\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u) { return 0xffffffffu; }\n"
      << "  let p = raw.x;\n"
      << "  let cr = uniforms.compress_ratio;\n"
      << "  return p / cr + select(0u, 1u, p % cr == cr - 1u);\n"
      << "}\n"
      << "fn query_value(token: u32, head: u32, d: u32, raw: vec2<u32>, b: u32) -> f32 {\n"
      << "  let base = (token * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = f32(" << query.GetByOffset("base + d") << ");\n"
      << "  let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "  if (d < rotary_base) { return value; }\n"
      << "  let offset = d - rotary_base;\n"
      << "  let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "  let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "  let paired = sign * f32(" << query.GetByOffset("base + pair_d") << ");\n"
      << "  let position = min(clamped_position(raw), uniforms.max_rotary_length - 1u);\n";
  if (cos_cache_batched_) {
    shader.AdditionalImplementation()
        << "  let cache = (b * uniforms.max_rotary_length + position) * uniforms.rotary_width + offset / 2u;\n";
  } else {
    shader.AdditionalImplementation()
        << "  let cache = position * uniforms.rotary_width + offset / 2u;\n";
  }
  shader.AdditionalImplementation()
      << "  value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  return value;\n"
      << "}\n"
      << "fn entry_score(token: u32, b: u32, entry: u32, raw: vec2<u32>) -> f32 {\n"
      << "  let key_base = (b * uniforms.state_capacity + entry) * uniforms.head_size;\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(token, head, d, raw, b) * f32("
      << present_key_state.GetByOffset("key_base + d") << ");\n"
      << "    }\n"
      << "    score += max(dot, 0.0) * f32(" << head_weights.GetByOffset("token * uniforms.num_heads + head")
      << ");\n"
      << "  }\n"
      << "  return score * uniforms.scale * uniforms.head_weight_scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let token = workgroup_idx;\n"
      << "  if (token >= uniforms.total_tokens || local_idx != 0u) { return; }\n"
      << "  let output_base = token * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected_indices.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let b = batch_of_token(token);\n"
      << "  let key_len_after = u32(" << present_state_lengths.GetByOffset("b * 2u") << ");\n"
      << "  let raw = " << position_ids.GetByOffset("token", true) << ";\n"
      << "  let threshold = min(causal_count(raw), key_len_after);\n"
      << "  let selected = min(uniforms.index_topk, threshold);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  var emitted = 0u;\n"
      << "  for (var rank = 0u; rank < selected; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < threshold; candidate++) {\n"
      << "      let score = entry_score(token, b, candidate, raw);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < "
         "best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    " << selected_indices.SetByOffset("output_base + rank", "best_index") << "\n"
      << "    emitted = rank + 1u;\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n"
      << "  " << selected_counts.SetByOffset("token", "i32(emitted)") << "\n";
  return Status::OK();
}

PackedSparseAttentionIndexer::PackedSparseAttentionIndexer(const OpKernelInfo& info) : WebGpuKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "PackedSparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(psai::TryParsePolicy(policy_mode, policy_), "PackedSparseAttentionIndexer: policy_mode must be '",
              psai::kPolicyModeQsa, "' or '", psai::kPolicyModeCsa, "', got '", policy_mode, "'");
  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "PackedSparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0, "PackedSparseAttentionIndexer: compress_ratio must be > 0");

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  has_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  has_head_weight_scale_ = info.GetAttr<float>("head_weight_scale", &head_weight_scale_).IsOK();
  if (policy_ == psai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget && token_budget_ > 0 && token_budget_ % compress_ratio_ == 0,
                "PackedSparseAttentionIndexer: token_budget must be > 0 and divisible by compress_ratio for qsa");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale_,
                "PackedSparseAttentionIndexer: csa attributes must be omitted for qsa");
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(has_index_topk && index_topk_ > 0, "PackedSparseAttentionIndexer: index_topk must be > 0 for csa");
    ORT_ENFORCE(!has_token_budget, "PackedSparseAttentionIndexer: token_budget must be omitted for csa");
    token_budget_ = 0;
  }
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "PackedSparseAttentionIndexer: epsilon must be >= 0");
}

Status PackedSparseAttentionIndexer::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const bool is_qsa = policy_ == psai::Policy::kQsa;
  constexpr int kCsaOnlyInputs[] = {psai::kGate, psai::kPositionBias, psai::kHeadWeights};
  for (int index : kCsaOnlyInputs) {
    const bool provided = index < context.InputCount() && context.Input(index) != nullptr;
    ORT_RETURN_IF(provided != !is_qsa, "PackedSparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode 'qsa'" : " is required for policy_mode 'csa'");
  }
  const bool position_ids_provided =
      psai::kPositionIds < context.InputCount() && context.Input(psai::kPositionIds) != nullptr;
  ORT_RETURN_IF(!is_qsa && !position_ids_provided,
                "PackedSparseAttentionIndexer: position_ids is required for "
                "policy_mode 'csa'");
  const bool gate_buffer_provided =
      psai::kPastGateBuffer < context.InputCount() && context.Input(psai::kPastGateBuffer) != nullptr;
  ORT_RETURN_IF(gate_buffer_provided != !is_qsa, "PackedSparseAttentionIndexer: past_gate_buffer ",
                gate_buffer_provided ? "must be omitted for policy_mode 'qsa'"
                                     : "is required for policy_mode 'csa'");
  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

Status PackedSparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(psai::kQuery);
  const Tensor* key = context.Input(psai::kKey);
  const Tensor* norm = context.Input(psai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(psai::kCosCache);
  const Tensor* sin_cache = context.Input(psai::kSinCache);
  const Tensor* cu_seqlens = context.Input(psai::kCumulativeSequenceLengths);
  const Tensor* past_seqlens = context.Input(psai::kPastSequenceLengths);
  const Tensor* position_ids = context.Input(psai::kPositionIds);
  const Tensor* past_key_state = context.Input(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context.Input(psai::kPastKvBuffer);
  const Tensor* past_state_lengths = context.Input(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "PackedSparseAttentionIndexer: query must have rank 3");
  const int64_t total_tokens = query_shape[0];
  const int64_t num_heads = query_shape[1];
  const int64_t head_size = query_shape[2];
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "PackedSparseAttentionIndexer: invalid query dimensions");

  ORT_RETURN_IF(cu_seqlens == nullptr, "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: invalid cumulative_sequence_lengths shape");
  const int64_t batch_size = cu_shape[0] - 1;

  ORT_RETURN_IF_ERROR(CheckShape(past_seqlens, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));
  if (position_ids != nullptr) {
    ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));
  }

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && rotary.rotary_width % 2 == 0 && rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: invalid qsa rotary_width");

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: invalid past_key_state shape");
  const int64_t state_capacity = key_state_shape[1];

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kQsa, token_budget_, index_topk_, compress_ratio_);
  Tensor* selected_indices = context.Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context.Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context.Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context.Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, head_size}));
  Tensor* present_state_lengths =
      context.Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));

  if (present_key_state->DataRaw() != past_key_state->DataRaw()) {
    const int64_t total = present_key_state->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_key_state, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_key_state, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }
  if (present_kv_buffer->DataRaw() != past_kv_buffer->DataRaw()) {
    const int64_t total = present_kv_buffer->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_kv_buffer, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_kv_buffer, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }
  if (present_state_lengths->DataRaw() != past_state_lengths->DataRaw()) {
    const int64_t total = present_state_lengths->Shape().Size();
    if (total > 0) {
      PackedSparseAttentionIndexerCopyProgram copy;
      copy.SetWorkgroupSize(kWorkgroupSize)
          .AddInput({past_state_lengths, ProgramTensorMetadataDependency::Type})
          .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
          .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
          .AddUniformVariables({{ToUint32(total)}});
      ORT_RETURN_IF_ERROR(context.RunProgram(copy));
    }
  }

  if (batch_size > 0) {
    PackedSparseAttentionIndexerQsaUpdateProgram update{rotary.batched};
    update.CacheHint(rotary.batched)
        .SetWorkgroupSize(kWorkgroupSize)
        .AddInputs({{key, ProgramTensorMetadataDependency::Type},
                    {norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type},
                    {cu_seqlens, ProgramTensorMetadataDependency::Type},
                    {past_kv_buffer, ProgramTensorMetadataDependency::Type}})
        .AddInput({past_state_lengths, ProgramTensorMetadataDependency::Type})
        .AddOutputs({{present_key_state, ProgramTensorMetadataDependency::Type},
                     {present_kv_buffer, ProgramTensorMetadataDependency::Type}})
        .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(ToUint32(batch_size))
        .AddUniformVariables({{ToUint32(batch_size)},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(state_capacity)},
                              {ToUint32(buffer_capacity)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary.rotary_width)},
                              {ToUint32(rotary.max_rotary_length)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(update));
  }

  if (total_tokens == 0) {
    return Status::OK();
  }

  PackedSparseAttentionIndexerQsaSelectProgram select{rotary.batched, position_ids != nullptr};
  select.CacheHint(rotary.batched, position_ids != nullptr)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                  {present_key_state, ProgramTensorMetadataDependency::Type},
                  {cos_cache, ProgramTensorMetadataDependency::Type},
                  {sin_cache, ProgramTensorMetadataDependency::Type},
                  {cu_seqlens, ProgramTensorMetadataDependency::Type},
                  {past_seqlens, ProgramTensorMetadataDependency::Type}});
  if (position_ids != nullptr) {
    select.AddInput({position_ids, ProgramTensorMetadataDependency::Type});
  }
  select.AddInput({present_state_lengths, ProgramTensorMetadataDependency::Type})
      .AddOutputs({{selected_indices, ProgramTensorMetadataDependency::Type},
                   {selected_counts, ProgramTensorMetadataDependency::Type}})
      .SetDispatchGroupSize(ToUint32(total_tokens))
      .AddUniformVariables({{ToUint32(total_tokens)},
                            {ToUint32(batch_size)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary.rotary_width)},
                            {ToUint32(rotary.max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(state_capacity)},
                            {ToUint32(capacity)},
                            {ToUint32(token_budget_ / compress_ratio_)},
                            {epsilon_},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
  return context.RunProgram(select);
}

Status PackedSparseAttentionIndexer::ComputeCsa(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(psai::kQuery);
  const Tensor* key = context.Input(psai::kKey);
  const Tensor* norm = context.Input(psai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(psai::kCosCache);
  const Tensor* sin_cache = context.Input(psai::kSinCache);
  const Tensor* cu_seqlens = context.Input(psai::kCumulativeSequenceLengths);
  const Tensor* past_seqlens = context.Input(psai::kPastSequenceLengths);
  const Tensor* gate = context.Input(psai::kGate);
  const Tensor* position_bias = context.Input(psai::kPositionBias);
  const Tensor* head_weights = context.Input(psai::kHeadWeights);
  const Tensor* position_ids = context.Input(psai::kPositionIds);
  const Tensor* past_key_state = context.Input(psai::kPastKeyState);
  const Tensor* past_kv_buffer = context.Input(psai::kPastKvBuffer);
  const Tensor* past_gate_buffer = context.Input(psai::kPastGateBuffer);
  const Tensor* past_state_lengths = context.Input(psai::kPastStateLengths);

  ORT_RETURN_IF(query == nullptr, "PackedSparseAttentionIndexer: query is required");
  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "PackedSparseAttentionIndexer: query must have rank 3");
  const int64_t total_tokens = query_shape[0];
  const int64_t num_heads = query_shape[1];
  const int64_t head_size = query_shape[2];
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "PackedSparseAttentionIndexer: invalid query dimensions");
  const int64_t width = 2 * head_size;

  ORT_RETURN_IF(cu_seqlens == nullptr, "PackedSparseAttentionIndexer: cumulative_sequence_lengths is required");
  const auto& cu_shape = cu_seqlens->Shape();
  ORT_RETURN_IF_NOT(cu_shape.NumDimensions() == 1 && cu_shape[0] >= 1,
                    "PackedSparseAttentionIndexer: invalid cumulative_sequence_lengths shape");
  const int64_t batch_size = cu_shape[0] - 1;

  ORT_RETURN_IF_ERROR(CheckShape(past_seqlens, "past_sequence_lengths", {batch_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {total_tokens, width}));
  ORT_RETURN_IF_ERROR(CheckShape(position_bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {total_tokens, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {total_tokens}));

  RotaryCacheShape rotary;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary));
  ORT_RETURN_IF_NOT(rotary.rotary_width > 0 && 2 * rotary.rotary_width <= head_size,
                    "PackedSparseAttentionIndexer: invalid csa rotary_width");

  ORT_RETURN_IF(past_key_state == nullptr, "PackedSparseAttentionIndexer: past_key_state is required");
  const auto& key_state_shape = past_key_state->Shape();
  ORT_RETURN_IF_NOT(key_state_shape.NumDimensions() == 3 && key_state_shape[0] == batch_size &&
                        key_state_shape[2] == head_size,
                    "PackedSparseAttentionIndexer: invalid past_key_state shape");
  const int64_t state_capacity = key_state_shape[1];

  const int64_t buffer_capacity = psai::GenericBufferCapacity(compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckShape(past_kv_buffer, "past_kv_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_gate_buffer, "past_gate_buffer", {batch_size, buffer_capacity, width}));
  ORT_RETURN_IF_ERROR(CheckShape(past_state_lengths, "past_state_lengths",
                                 {batch_size, psai::kStateLengthColumns}));

  const int64_t capacity = psai::SelectedCapacity(psai::Policy::kCsa, token_budget_, index_topk_, compress_ratio_);
  Tensor* selected_indices = context.Output(psai::kSelectedIndices, TensorShape({total_tokens, capacity}));
  Tensor* selected_counts = context.Output(psai::kSelectedCounts, TensorShape({total_tokens}));
  Tensor* present_key_state = context.Output(psai::kPresentKeyState, key_state_shape);
  Tensor* present_kv_buffer =
      context.Output(psai::kPresentKvBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_gate_buffer =
      context.Output(psai::kPresentGateBuffer, TensorShape({batch_size, buffer_capacity, width}));
  Tensor* present_state_lengths =
      context.Output(psai::kPresentStateLengths, TensorShape({batch_size, psai::kStateLengthColumns}));

  auto copy_if_needed = [&](const Tensor* src, Tensor* dst) -> Status {
    if (src->DataRaw() == dst->DataRaw()) {
      return Status::OK();
    }
    const int64_t total = dst->Shape().Size();
    if (total == 0) {
      return Status::OK();
    }
    PackedSparseAttentionIndexerCopyProgram copy;
    copy.SetWorkgroupSize(kWorkgroupSize)
        .AddInput({src, ProgramTensorMetadataDependency::Type})
        .AddOutput({dst, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(ToUint32((total + kWorkgroupSize - 1) / kWorkgroupSize))
        .AddUniformVariables({{ToUint32(total)}});
    return context.RunProgram(copy);
  };
  ORT_RETURN_IF_ERROR(copy_if_needed(past_key_state, present_key_state));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_kv_buffer, present_kv_buffer));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_gate_buffer, present_gate_buffer));
  ORT_RETURN_IF_ERROR(copy_if_needed(past_state_lengths, present_state_lengths));

  if (batch_size > 0) {
    PackedSparseAttentionIndexerCsaUpdateProgram update{rotary.batched};
    update.CacheHint(rotary.batched)
        .SetWorkgroupSize(kWorkgroupSize)
        .AddInputs({{key, ProgramTensorMetadataDependency::Type},
                    {gate, ProgramTensorMetadataDependency::Type},
                    {norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type},
                    {position_bias, ProgramTensorMetadataDependency::Type},
                    {cu_seqlens, ProgramTensorMetadataDependency::Type},
                    {past_kv_buffer, ProgramTensorMetadataDependency::Type},
                    {past_gate_buffer, ProgramTensorMetadataDependency::Type}})
        .AddInput({past_state_lengths, ProgramTensorMetadataDependency::Type})
        .AddOutputs({{present_key_state, ProgramTensorMetadataDependency::Type},
                     {present_kv_buffer, ProgramTensorMetadataDependency::Type},
                     {present_gate_buffer, ProgramTensorMetadataDependency::Type}})
        .AddOutput({present_state_lengths, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(ToUint32(batch_size))
        .AddUniformVariables({{ToUint32(batch_size)},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(state_capacity)},
                              {ToUint32(buffer_capacity)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary.rotary_width)},
                              {ToUint32(rotary.max_rotary_length)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(update));
  }

  if (total_tokens == 0) {
    return Status::OK();
  }

  PackedSparseAttentionIndexerCsaSelectProgram select{rotary.batched};
  select.CacheHint(rotary.batched)
      .SetWorkgroupSize(kWorkgroupSize)
      .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                  {present_key_state, ProgramTensorMetadataDependency::Type},
                  {head_weights, ProgramTensorMetadataDependency::Type},
                  {position_ids, ProgramTensorMetadataDependency::Type},
                  {cos_cache, ProgramTensorMetadataDependency::Type},
                  {sin_cache, ProgramTensorMetadataDependency::Type},
                  {cu_seqlens, ProgramTensorMetadataDependency::Type}})
      .AddInput({present_state_lengths, ProgramTensorMetadataDependency::Type})
      .AddOutputs({{selected_indices, ProgramTensorMetadataDependency::Type},
                   {selected_counts, ProgramTensorMetadataDependency::Type}})
      .SetDispatchGroupSize(ToUint32(total_tokens))
      .AddUniformVariables({{ToUint32(total_tokens)},
                            {ToUint32(batch_size)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary.rotary_width)},
                            {ToUint32(rotary.max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(state_capacity)},
                            {ToUint32(capacity)},
                            {ToUint32(index_topk_)},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))},
                            {has_head_weight_scale_ ? head_weight_scale_
                                                    : 1.0f / std::sqrt(static_cast<float>(num_heads))}});
  return context.RunProgram(select);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
