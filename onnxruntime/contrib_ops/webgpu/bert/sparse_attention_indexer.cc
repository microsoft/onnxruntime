// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/sparse_attention_indexer.h"

#include <cmath>
#include <limits>
#include <string>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace sai = onnxruntime::contrib::sparse_attention_indexer;

ONNX_OPERATOR_KERNEL_EX(
    SparseAttentionIndexer,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TB", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    SparseAttentionIndexer);

namespace {

constexpr uint32_t kWorkgroupSize = 64;

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "SparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "SparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

uint32_t ToUint32(int64_t value) {
  return onnxruntime::narrow<uint32_t>(value);
}

}  // namespace

Status SparseAttentionIndexerFillProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  " << output.SetByOffset("global_idx", "-1") << "\n";
  return Status::OK();
}

Status SparseAttentionIndexerQsaConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* past = nullptr;
  const ShaderVariableHelper* current = nullptr;
  if (has_past_) {
    past = &shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (has_current_) {
    current = &shader.AddInput("key", ShaderUsage::UseUniform);
  }
  const auto& present = shader.AddOutput("present_key",
                                         ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let d = global_idx % uniforms.head_size;\n"
      << "  let token_row = global_idx / uniforms.head_size;\n"
      << "  let token = token_row % uniforms.total_sequence_length;\n"
      << "  let batch = token_row / uniforms.total_sequence_length;\n";
  if (has_past_) {
    shader.MainFunctionBody()
        << "  if (token < uniforms.past_sequence_length) {\n"
        << "    " << present.SetByOffset("global_idx", "present_key_element_t(" + past->GetByOffset("(batch * uniforms.past_sequence_length + token) * uniforms.head_size + d") + ")")
        << "\n"
        << "    return;\n"
        << "  }\n";
  }
  if (has_current_) {
    shader.MainFunctionBody()
        << "  let current_token = token - uniforms.past_sequence_length;\n"
        << "  " << present.SetByOffset("global_idx", "present_key_element_t(" + current->GetByOffset("(batch * uniforms.sequence_length + current_token) * uniforms.head_size + d") + ")")
        << "\n";
  }
  return Status::OK();
}

Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform);
  const auto& selected = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn visible(row: u32, token: u32) -> bool {\n"
      << "  let offset = row * uniforms.total_sequence_length + token;\n"
      << "  return " << mask.GetByOffset("offset / 4u") << "[offset % 4u];\n"
      << "}\n"
      << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
      << "  var seen = 0u;\n"
      << "  for (var token = 0u; token < uniforms.total_sequence_length; token++) {\n"
      << "    if (visible(row, token)) {\n"
      << "      if (seen == ordinal) { return token; }\n"
      << "      seen++;\n"
      << "    }\n"
      << "  }\n"
      << "  return uniforms.total_sequence_length;\n"
      << "}\n"
      << "fn clamp_position(position: u32) -> u32 {\n"
      << "  return min(position, uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = (row * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = f32(" << query.GetByOffset("base + d") << ");\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * f32(" << query.GetByOffset("base + pair_d") << ");\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let token = row % uniforms.sequence_length;\n"
      << "    let position = clamp_position(uniforms.past_sequence_length + token);\n"
      << "    let cache = (batch * uniforms.max_rotary_length + position) * uniforms.rotary_width + d;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn pooled_value(row: u32, block: u32, d: u32) -> f32 {\n"
      << "  let batch = row / uniforms.sequence_length;\n"
      << "  let key_base = batch * uniforms.total_sequence_length * uniforms.head_size;\n"
      << "  var sum = 0.0;\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let token = visible_at(row, block * uniforms.compress_ratio + slot);\n"
      << "    sum += f32(" << present_key.GetByOffset("key_base + token * uniforms.head_size + d") << ");\n"
      << "  }\n"
      << "  return sum / f32(uniforms.compress_ratio);\n"
      << "}\n"
      << "fn normalized_value(row: u32, block: u32, d: u32) -> f32 {\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = pooled_value(row, block, k);\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  return pooled_value(row, block, d) * inverseSqrt(square_sum / f32(uniforms.head_size) + "
         "uniforms.epsilon) * f32("
      << norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn key_value(row: u32, block: u32, d: u32) -> f32 {\n"
      << "  var value = normalized_value(row, block, d);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_value(row, block, pair_d);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let position = clamp_position(visible_at(row, block * uniforms.compress_ratio));\n"
      << "    let cache = (batch * uniforms.max_rotary_length + position) * uniforms.rotary_width + d;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(row: u32, block: u32) -> f32 {\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(row, head, d) * key_value(row, block, d);\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows || local_idx != 0u) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  var visible_count = 0u;\n"
      << "  for (var token = 0u; token < uniforms.total_sequence_length; token++) {\n"
      << "    if (visible(row, token)) { visible_count++; }\n"
      << "  }\n"
      << "  let block_count = visible_count / uniforms.compress_ratio;\n"
      << "  let selected_blocks = min(uniforms.block_topk, block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < selected_blocks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < block_count; candidate++) {\n"
      << "      let score = block_score(row, candidate);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let token = visible_at(row, u32(best_index) * uniforms.compress_ratio + slot);\n"
      << "      " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
      << "    }\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n"
      << "  let tail_start = block_count * uniforms.compress_ratio;\n"
      << "  for (var tail = tail_start; tail < visible_count; tail++) {\n"
      << "    let output = selected_blocks * uniforms.compress_ratio + tail - tail_start;\n"
      << "    " << selected.SetByOffset("output_base + output", "i32(visible_at(row, tail))") << "\n"
      << "  }\n";
  return Status::OK();
}

Status SparseAttentionIndexerCsaCopyCompressedProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& past = shader.AddInput("past_compressed_key", ShaderUsage::UseUniform);
  const auto& present = shader.AddOutput("present_compressed_key",
                                         ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let d = global_idx % uniforms.head_size;\n"
      << "  let entry_row = global_idx / uniforms.head_size;\n"
      << "  let entry = entry_row % uniforms.past_length;\n"
      << "  let batch = entry_row / uniforms.past_length;\n"
      << "  let output = (batch * uniforms.present_length + entry) * uniforms.head_size + d;\n"
      << "  " << present.SetByOffset("output", "present_compressed_key_element_t(" + past.GetByOffset("global_idx") + ")")
      << "\n";
  return Status::OK();
}

Status SparseAttentionIndexerCsaCompressProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_kv = nullptr;
  const ShaderVariableHelper* past_gate = nullptr;
  if (has_past_buffer_) {
    past_kv = &shader.AddInput("past_kv_buffer", ShaderUsage::UseUniform);
    past_gate = &shader.AddInput("past_gate_buffer", ShaderUsage::UseUniform);
  }
  const auto& bias = shader.AddInput("position_bias", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& present = shader.AddOutput("present_compressed_key",
                                         ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << "fn kv_value(batch: u32, position: u32, channel: u32) -> f32 {\n";
  if (has_past_buffer_) {
    shader.AdditionalImplementation()
        << "  if (position < uniforms.past_buffer_length) {\n"
        << "    return f32(" << past_kv->GetByOffset(
                                    "(batch * uniforms.past_buffer_length + position) * "
                                    "(2u * uniforms.head_size) + channel")
        << ");\n"
        << "  }\n";
  }
  shader.AdditionalImplementation()
      << "  let token = position - uniforms.past_buffer_length;\n"
      << "  return f32(" << key.GetByOffset(
                                "(batch * uniforms.sequence_length + token) * "
                                "(2u * uniforms.head_size) + channel")
      << ");\n"
      << "}\n"
      << "fn gate_value(batch: u32, position: u32, channel: u32) -> f32 {\n";
  if (has_past_buffer_) {
    shader.AdditionalImplementation()
        << "  if (position < uniforms.past_buffer_length) {\n"
        << "    return f32(" << past_gate->GetByOffset(
                                    "(batch * uniforms.past_buffer_length + position) * "
                                    "(2u * uniforms.head_size) + channel")
        << ");\n"
        << "  }\n";
  }
  shader.AdditionalImplementation()
      << "  let token = position - uniforms.past_buffer_length;\n"
      << "  return f32(" << gate.GetByOffset(
                                "(batch * uniforms.sequence_length + token) * "
                                "(2u * uniforms.head_size) + channel")
      << ");\n"
      << "}\n"
      << "fn pooled_value(batch: u32, window: u32, d: u32) -> f32 {\n"
      << "  let width = 2u * uniforms.head_size;\n"
      << "  let current_base = uniforms.overlap_length + window * uniforms.compress_ratio;\n"
      << "  let has_previous = window >= 1u || uniforms.overlap_length >= uniforms.compress_ratio;\n"
      << "  var max_gate = -3.4028234663852886e+38;\n"
      << "  if (has_previous) {\n"
      << "    let previous_base = uniforms.overlap_length + (window - 1u) * uniforms.compress_ratio;\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      max_gate = max(max_gate, gate_value(batch, previous_base + slot, d) + f32("
      << bias.GetByOffset("slot * width + d") << "));\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    max_gate = max(max_gate, gate_value(batch, current_base + slot, uniforms.head_size + d) + f32("
      << bias.GetByOffset("slot * width + uniforms.head_size + d") << "));\n"
      << "  }\n"
      << "  var denominator = 0.0;\n"
      << "  var accumulator = 0.0;\n"
      << "  if (has_previous) {\n"
      << "    let previous_base = uniforms.overlap_length + (window - 1u) * uniforms.compress_ratio;\n"
      << "    for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "      let weight = exp(gate_value(batch, previous_base + slot, d) + f32("
      << bias.GetByOffset("slot * width + d") << ") - max_gate);\n"
      << "      denominator += weight;\n"
      << "      accumulator += weight * kv_value(batch, previous_base + slot, d);\n"
      << "    }\n"
      << "  }\n"
      << "  for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
      << "    let channel = uniforms.head_size + d;\n"
      << "    let weight = exp(gate_value(batch, current_base + slot, channel) + f32("
      << bias.GetByOffset("slot * width + uniforms.head_size + d") << ") - max_gate);\n"
      << "    denominator += weight;\n"
      << "    accumulator += weight * kv_value(batch, current_base + slot, channel);\n"
      << "  }\n"
      << "  return select(0.0, accumulator / denominator, denominator > 0.0);\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let work = workgroup_idx;\n"
      << "  if (work >= uniforms.work_items || local_idx != 0u) { return; }\n"
      << "  let window = work % uniforms.new_window_count;\n"
      << "  let batch = work / uniforms.new_window_count;\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "    let value = pooled_value(batch, window, d);\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  let inverse_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "  let entry = uniforms.past_compressed_length + window;\n"
      << "  let position = min(entry * uniforms.compress_ratio, uniforms.max_rotary_length - 1u);\n"
      << "  let cache_base = (batch * uniforms.max_rotary_length + position) * uniforms.rotary_width;\n"
      << "  let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "  for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "    var value = pooled_value(batch, window, d) * inverse_rms * f32(" << norm.GetByOffset("d") << ");\n"
      << "    if (d >= rotary_base) {\n"
      << "      let offset = d - rotary_base;\n"
      << "      let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "      let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "      let paired = sign * pooled_value(batch, window, pair_d) * inverse_rms * f32("
      << norm.GetByOffset("pair_d") << ");\n"
      << "      value = value * f32(" << cos_cache.GetByOffset("cache_base + offset / 2u")
      << ") + paired * f32(" << sin_cache.GetByOffset("cache_base + offset / 2u") << ");\n"
      << "    }\n"
      << "    let output = (batch * uniforms.present_compressed_length + entry) * uniforms.head_size + d;\n"
      << "    " << present.SetByOffset("output", "present_compressed_key_element_t(value)") << "\n"
      << "  }\n";
  return Status::OK();
}

Status SparseAttentionIndexerCsaCopyBufferProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* key = nullptr;
  const ShaderVariableHelper* gate = nullptr;
  if (has_current_) {
    key = &shader.AddInput("key", ShaderUsage::UseUniform);
    gate = &shader.AddInput("gate", ShaderUsage::UseUniform);
  }
  const ShaderVariableHelper* past_kv = nullptr;
  const ShaderVariableHelper* past_gate = nullptr;
  if (has_past_buffer_) {
    past_kv = &shader.AddInput("past_kv_buffer", ShaderUsage::UseUniform);
    past_gate = &shader.AddInput("past_gate_buffer", ShaderUsage::UseUniform);
  }
  const auto& present_kv = shader.AddOutput("present_kv_buffer",
                                            ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& present_gate = shader.AddOutput("present_gate_buffer",
                                              ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let width = 2u * uniforms.head_size;\n"
      << "  let channel = global_idx % width;\n"
      << "  let token_row = global_idx / width;\n"
      << "  let token = token_row % uniforms.present_buffer_length;\n"
      << "  let batch = token_row / uniforms.present_buffer_length;\n"
      << "  let source = uniforms.present_buffer_start + token;\n";
  if (has_past_buffer_) {
    shader.MainFunctionBody()
        << "  if (source < uniforms.past_buffer_length) {\n"
        << "    let input = (batch * uniforms.past_buffer_length + source) * width + channel;\n"
        << "    " << present_kv.SetByOffset("global_idx", "present_kv_buffer_element_t(" + past_kv->GetByOffset("input") + ")")
        << "\n"
        << "    " << present_gate.SetByOffset("global_idx", "present_gate_buffer_element_t(" + past_gate->GetByOffset("input") + ")")
        << "\n"
        << "    return;\n"
        << "  }\n";
  }
  if (has_current_) {
    shader.MainFunctionBody()
        << "  let current_token = source - uniforms.past_buffer_length;\n"
        << "  let input = (batch * uniforms.sequence_length + current_token) * width + channel;\n"
        << "  "
        << present_kv.SetByOffset("global_idx",
                                  "present_kv_buffer_element_t(" + key->GetByOffset("input") + ")")
        << "\n"
        << "  "
        << present_gate.SetByOffset("global_idx",
                                    "present_gate_buffer_element_t(" + gate->GetByOffset("input") + ")")
        << "\n";
  }
  return Status::OK();
}

Status SparseAttentionIndexerCsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& compressed_key = shader.AddInput("present_compressed_key", ShaderUsage::UseUniform);
  const auto& head_weights = shader.AddInput("head_weights", ShaderUsage::UseUniform);
  const auto& position_ids = shader.AddInput("position_ids", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& selected = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);

  shader.AdditionalImplementation()
      << "fn clamped_position(row: u32, limit: u32) -> u32 {\n"
      << "  let raw = " << position_ids.GetByOffset("row", true) << ";\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u || raw.x > limit) { return limit; }\n"
      << "  return raw.x;\n"
      << "}\n"
      << "fn visible_entry_count(row: u32, count: u32) -> u32 {\n"
      << "  let raw = " << position_ids.GetByOffset("row", true) << ";\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  if (raw.y != 0u) { return count; }\n"
      << "  let quotient = raw.x / uniforms.compress_ratio;\n"
      << "  if (quotient >= count) { return count; }\n"
      << "  let increment = select(0u, 1u, raw.x % uniforms.compress_ratio == uniforms.compress_ratio - 1u);\n"
      << "  return min(count, quotient + increment);\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = (row * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = f32(" << query.GetByOffset("base + d") << ");\n"
      << "  let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "  if (d >= rotary_base) {\n"
      << "    let offset = d - rotary_base;\n"
      << "    let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "    let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "    let paired = sign * f32(" << query.GetByOffset("base + pair_d") << ");\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let position = clamped_position(row, uniforms.max_rotary_length - 1u);\n"
      << "    let cache = (batch * uniforms.max_rotary_length + position) * uniforms.rotary_width + offset / 2u;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn entry_score(row: u32, entry: u32) -> f32 {\n"
      << "  let batch = row / uniforms.sequence_length;\n"
      << "  let key_base = (batch * uniforms.present_compressed_length + entry) * uniforms.head_size;\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(row, head, d) * f32(" << compressed_key.GetByOffset("key_base + d") << ");\n"
      << "    }\n"
      << "    score += max(dot, 0.0) * f32("
      << head_weights.GetByOffset("row * uniforms.num_heads + head") << ");\n"
      << "  }\n"
      << "  return score * uniforms.scale * uniforms.head_weight_scale;\n"
      << "}\n";

  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows || local_idx != 0u) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let count = uniforms.present_compressed_length;\n"
      << "  let threshold = visible_entry_count(row, count);\n"
      << "  let ranks = min(uniforms.capacity, count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < ranks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < count; candidate++) {\n"
      << "      let score = select(-3.4028234663852886e+38, entry_score(row, candidate), candidate < threshold);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    if (u32(best_index) < threshold) {\n"
      << "      " << selected.SetByOffset("output_base + rank", "best_index") << "\n"
      << "    }\n"
      << "    previous_score = best_score;\n"
      << "    previous_index = best_index;\n"
      << "  }\n";
  return Status::OK();
}

SparseAttentionIndexer::SparseAttentionIndexer(const OpKernelInfo& info) : WebGpuKernel(info) {
  std::string policy_mode;
  ORT_ENFORCE(info.GetAttr<std::string>("policy_mode", &policy_mode).IsOK(),
              "SparseAttentionIndexer: policy_mode is required");
  ORT_ENFORCE(sai::TryParsePolicy(policy_mode, policy_), "SparseAttentionIndexer: policy_mode must be '",
              sai::kPolicyModeQsa, "' or '", sai::kPolicyModeCsa, "', got '", policy_mode, "'");
  ORT_ENFORCE(info.GetAttr<int64_t>("compress_ratio", &compress_ratio_).IsOK(),
              "SparseAttentionIndexer: compress_ratio is required");
  ORT_ENFORCE(compress_ratio_ > 0, "SparseAttentionIndexer: compress_ratio must be > 0");

  const bool has_token_budget = info.GetAttr<int64_t>("token_budget", &token_budget_).IsOK();
  const bool has_index_topk = info.GetAttr<int64_t>("index_topk", &index_topk_).IsOK();
  has_scale_ = info.GetAttr<float>("scale", &scale_).IsOK();
  has_head_weight_scale_ = info.GetAttr<float>("head_weight_scale", &head_weight_scale_).IsOK();
  if (policy_ == sai::Policy::kQsa) {
    ORT_ENFORCE(has_token_budget && token_budget_ > 0 && token_budget_ % compress_ratio_ == 0,
                "SparseAttentionIndexer: token_budget must be > 0 and divisible by compress_ratio for qsa");
    ORT_ENFORCE(!has_index_topk && !has_head_weight_scale_,
                "SparseAttentionIndexer: csa attributes must be omitted for qsa");
    index_topk_ = 0;
  } else {
    ORT_ENFORCE(has_index_topk && index_topk_ > 0,
                "SparseAttentionIndexer: index_topk must be > 0 for csa");
    ORT_ENFORCE(!has_token_budget, "SparseAttentionIndexer: token_budget must be omitted for csa");
    token_budget_ = 0;
  }
  epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1.0e-6f);
  ORT_ENFORCE(epsilon_ >= 0.0f, "SparseAttentionIndexer: epsilon must be >= 0");
}

Status SparseAttentionIndexer::ComputeInternal(ComputeContext& context) const {
  const bool is_qsa = policy_ == sai::Policy::kQsa;
  for (int index = sai::kMask; index < sai::kInputCount; ++index) {
    const bool policy_owns_slot = is_qsa ? (index <= sai::kPastKey) : (index >= sai::kGate);
    const bool provided = index < context.InputCount() && context.Input(index) != nullptr;
    ORT_RETURN_IF(provided != policy_owns_slot, "SparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode '" : " is required for policy_mode '",
                  is_qsa ? sai::kPolicyModeQsa : sai::kPolicyModeCsa, "'");
  }
  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

Status SparseAttentionIndexer::ComputeQsa(ComputeContext& context) const {
  const Tensor* query = context.Input(sai::kQuery);
  const Tensor* key = context.Input(sai::kKey);
  const Tensor* norm = context.Input(sai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(sai::kCosCache);
  const Tensor* sin_cache = context.Input(sai::kSinCache);
  const Tensor* mask = context.Input(sai::kMask);
  const Tensor* past_key = context.Input(sai::kPastKey);

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 4, "SparseAttentionIndexer: query must have rank 4");
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const int64_t num_heads = query_shape[2];
  const int64_t head_size = query_shape[3];
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "SparseAttentionIndexer: invalid query dimensions");
  const auto& past_shape = past_key->Shape();
  ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3 && past_shape[0] == batch_size && past_shape[2] == head_size,
                    "SparseAttentionIndexer: invalid past_key shape");
  const int64_t past_length = past_shape[1];
  const int64_t total_length = past_length + sequence_length;
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));
  const auto& cos_shape = cos_cache->Shape();
  ORT_RETURN_IF_NOT(cos_shape.NumDimensions() == 3 && cos_shape[0] == batch_size && cos_shape[1] > 0,
                    "SparseAttentionIndexer: invalid cos_cache shape");
  const int64_t max_rotary_length = cos_shape[1];
  const int64_t rotary_width = cos_shape[2];
  ORT_RETURN_IF_NOT(sin_cache->Shape() == cos_shape && rotary_width > 0 && rotary_width % 2 == 0 &&
                        rotary_width <= head_size,
                    "SparseAttentionIndexer: invalid qsa rotary cache shape");
  const auto& mask_shape = mask->Shape();
  ORT_RETURN_IF_NOT(
      (mask_shape.NumDimensions() == 4 && mask_shape[0] == batch_size && mask_shape[1] == 1 &&
       mask_shape[2] == sequence_length && mask_shape[3] == total_length) ||
          (mask_shape.NumDimensions() == 3 && mask_shape[0] == batch_size &&
           mask_shape[1] == sequence_length && mask_shape[2] == total_length),
      "SparseAttentionIndexer: invalid qsa mask shape");

  const int64_t capacity = sai::SelectedCapacity(policy_, token_budget_, index_topk_, compress_ratio_);
  Tensor* selected =
      context.Output(sai::kSelectedIndices, TensorShape({batch_size, sequence_length, capacity}));
  Tensor* present = context.Output(sai::kPresentKey, TensorShape({batch_size, total_length, head_size}));
  const int64_t present_elements = present->Shape().Size();
  if (present_elements > 0) {
    const bool has_past = past_length > 0;
    const bool has_current = sequence_length > 0;
    SparseAttentionIndexerQsaConcatProgram concat{has_past, has_current};
    concat.CacheHint(has_past, has_current)
        .SetWorkgroupSize(kWorkgroupSize);
    if (has_past) {
      concat.AddInput({past_key, ProgramTensorMetadataDependency::Type});
    }
    if (has_current) {
      concat.AddInput({key, ProgramTensorMetadataDependency::Type});
    }
    concat.AddOutput({present, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize((ToUint32(present_elements) + kWorkgroupSize - 1) / kWorkgroupSize)
        .AddUniformVariables({{ToUint32(present_elements)},
                              {ToUint32(sequence_length)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(head_size)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(concat));
  }
  const int64_t rows = batch_size * sequence_length;
  if (rows == 0) {
    return Status::OK();
  }
  SparseAttentionIndexerQsaSelectProgram select;
  select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity)
      .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                  {present, ProgramTensorMetadataDependency::Type},
                  {norm, ProgramTensorMetadataDependency::Type},
                  {cos_cache, ProgramTensorMetadataDependency::Type},
                  {sin_cache, ProgramTensorMetadataDependency::Type}})
      .AddInput({mask, ProgramTensorMetadataDependency::Type, {(mask->Shape().Size() + 3) / 4}, 4})
      .AddOutput({selected, ProgramTensorMetadataDependency::Type})
      .SetWorkgroupSize(kWorkgroupSize)
      .SetDispatchGroupSize(ToUint32(rows))
      .AddUniformVariables({{ToUint32(rows)},
                            {ToUint32(sequence_length)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary_width)},
                            {ToUint32(max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(capacity)},
                            {ToUint32(past_length)},
                            {ToUint32(total_length)},
                            {ToUint32(token_budget_ / compress_ratio_)},
                            {epsilon_},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
  return context.RunProgram(select);
}

Status SparseAttentionIndexer::ComputeCsa(ComputeContext& context) const {
  const Tensor* query = context.Input(sai::kQuery);
  const Tensor* key = context.Input(sai::kKey);
  const Tensor* norm = context.Input(sai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(sai::kCosCache);
  const Tensor* sin_cache = context.Input(sai::kSinCache);
  const Tensor* gate = context.Input(sai::kGate);
  const Tensor* bias = context.Input(sai::kPositionBias);
  const Tensor* head_weights = context.Input(sai::kHeadWeights);
  const Tensor* position_ids = context.Input(sai::kPositionIds);
  const Tensor* past_compressed = context.Input(sai::kPastCompressedKey);
  const Tensor* past_kv = context.Input(sai::kPastKvBuffer);
  const Tensor* past_gate = context.Input(sai::kPastGateBuffer);

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 4, "SparseAttentionIndexer: query must have rank 4");
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const int64_t num_heads = query_shape[2];
  const int64_t head_size = query_shape[3];
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "SparseAttentionIndexer: invalid query dimensions");
  const int64_t width = 2 * head_size;
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(norm, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {batch_size, sequence_length, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {batch_size, sequence_length}));

  const auto& cos_shape = cos_cache->Shape();
  ORT_RETURN_IF_NOT(cos_shape.NumDimensions() == 3 && cos_shape[0] == batch_size && cos_shape[1] > 0,
                    "SparseAttentionIndexer: invalid cos_cache shape");
  const int64_t max_rotary_length = cos_shape[1];
  const int64_t rotary_width = cos_shape[2];
  ORT_RETURN_IF_NOT(sin_cache->Shape() == cos_shape && rotary_width > 0 && 2 * rotary_width <= head_size,
                    "SparseAttentionIndexer: invalid csa rotary cache shape");
  const auto& past_compressed_shape = past_compressed->Shape();
  ORT_RETURN_IF_NOT(past_compressed_shape.NumDimensions() == 3 &&
                        past_compressed_shape[0] == batch_size && past_compressed_shape[2] == head_size,
                    "SparseAttentionIndexer: invalid past_compressed_key shape");
  const int64_t past_compressed_length = past_compressed_shape[1];
  const auto& past_buffer_shape = past_kv->Shape();
  ORT_RETURN_IF_NOT(past_buffer_shape.NumDimensions() == 3 && past_buffer_shape[0] == batch_size &&
                        past_buffer_shape[2] == width,
                    "SparseAttentionIndexer: invalid past_kv_buffer shape");
  const int64_t past_buffer_length = past_buffer_shape[1];
  ORT_RETURN_IF_ERROR(CheckShape(past_gate, "past_gate_buffer", {batch_size, past_buffer_length, width}));

  sai::CsaWindowPlan plan;
  ORT_RETURN_IF_NOT(sai::TryComputeCsaWindowPlan(past_buffer_length, sequence_length, compress_ratio_, plan),
                    "SparseAttentionIndexer: invalid csa buffer length");
  const int64_t present_compressed_length = past_compressed_length + plan.new_window_count;
  const int64_t capacity = sai::SelectedCapacity(policy_, token_budget_, index_topk_, compress_ratio_);
  Tensor* selected =
      context.Output(sai::kSelectedIndices, TensorShape({batch_size, sequence_length, capacity}));
  Tensor* present_compressed = context.Output(
      sai::kPresentCompressedKey, TensorShape({batch_size, present_compressed_length, head_size}));
  Tensor* present_kv =
      context.Output(sai::kPresentKvBuffer, TensorShape({batch_size, plan.present_buffer_length, width}));
  Tensor* present_gate =
      context.Output(sai::kPresentGateBuffer, TensorShape({batch_size, plan.present_buffer_length, width}));

  const int64_t past_compressed_elements = batch_size * past_compressed_length * head_size;
  if (past_compressed_elements > 0) {
    SparseAttentionIndexerCsaCopyCompressedProgram copy;
    copy.CacheHint(query->GetElementType())
        .AddInput({past_compressed, ProgramTensorMetadataDependency::Type})
        .AddOutput({present_compressed, ProgramTensorMetadataDependency::Type})
        .SetWorkgroupSize(kWorkgroupSize)
        .SetDispatchGroupSize((ToUint32(past_compressed_elements) + kWorkgroupSize - 1) / kWorkgroupSize)
        .AddUniformVariables({{ToUint32(past_compressed_elements)},
                              {ToUint32(head_size)},
                              {ToUint32(past_compressed_length)},
                              {ToUint32(present_compressed_length)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(copy));
  }

  if (plan.new_window_count > 0) {
    const bool has_past_buffer = past_buffer_length > 0;
    SparseAttentionIndexerCsaCompressProgram compress{has_past_buffer};
    compress.CacheHint(query->GetElementType(), has_past_buffer, head_size, rotary_width, compress_ratio_)
        .AddInputs({{key, ProgramTensorMetadataDependency::Type},
                    {gate, ProgramTensorMetadataDependency::Type}});
    if (has_past_buffer) {
      compress.AddInputs({{past_kv, ProgramTensorMetadataDependency::Type},
                          {past_gate, ProgramTensorMetadataDependency::Type}});
    }
    compress.AddInputs({{bias, ProgramTensorMetadataDependency::Type},
                        {norm, ProgramTensorMetadataDependency::Type},
                        {cos_cache, ProgramTensorMetadataDependency::Type},
                        {sin_cache, ProgramTensorMetadataDependency::Type}})
        .AddOutput({present_compressed, ProgramTensorMetadataDependency::Type})
        .SetWorkgroupSize(kWorkgroupSize)
        .SetDispatchGroupSize(ToUint32(batch_size * plan.new_window_count))
        .AddUniformVariables({{ToUint32(batch_size * plan.new_window_count)},
                              {ToUint32(sequence_length)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(past_compressed_length)},
                              {ToUint32(present_compressed_length)},
                              {ToUint32(past_buffer_length)},
                              {ToUint32(plan.overlap_length)},
                              {ToUint32(plan.new_window_count)},
                              {epsilon_}});
    ORT_RETURN_IF_ERROR(context.RunProgram(compress));
  }

  const int64_t present_buffer_elements = batch_size * plan.present_buffer_length * width;
  if (present_buffer_elements > 0) {
    const bool has_past_buffer = past_buffer_length > 0;
    const bool has_current = sequence_length > 0;
    SparseAttentionIndexerCsaCopyBufferProgram copy_buffer{has_past_buffer, has_current};
    copy_buffer.CacheHint(query->GetElementType(), has_past_buffer, has_current);
    if (has_current) {
      copy_buffer.AddInputs({{key, ProgramTensorMetadataDependency::Type},
                             {gate, ProgramTensorMetadataDependency::Type}});
    }
    if (has_past_buffer) {
      copy_buffer.AddInputs({{past_kv, ProgramTensorMetadataDependency::Type},
                             {past_gate, ProgramTensorMetadataDependency::Type}});
    }
    copy_buffer.AddOutputs({{present_kv, ProgramTensorMetadataDependency::Type},
                            {present_gate, ProgramTensorMetadataDependency::Type}})
        .SetWorkgroupSize(kWorkgroupSize)
        .SetDispatchGroupSize((ToUint32(present_buffer_elements) + kWorkgroupSize - 1) / kWorkgroupSize)
        .AddUniformVariables({{ToUint32(present_buffer_elements)},
                              {ToUint32(sequence_length)},
                              {ToUint32(head_size)},
                              {ToUint32(past_buffer_length)},
                              {ToUint32(plan.present_buffer_length)},
                              {ToUint32(plan.present_buffer_start)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(copy_buffer));
  }

  const int64_t rows = batch_size * sequence_length;
  if (rows == 0) {
    return Status::OK();
  }
  if (present_compressed_length == 0) {
    const int64_t output_elements = selected->Shape().Size();
    SparseAttentionIndexerFillProgram fill;
    fill.AddOutput({selected, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(kWorkgroupSize)
        .SetDispatchGroupSize((ToUint32(output_elements) + kWorkgroupSize - 1) / kWorkgroupSize)
        .AddUniformVariable({ToUint32(output_elements)});
    return context.RunProgram(fill);
  }
  SparseAttentionIndexerCsaSelectProgram select;
  select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity)
      .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                  {present_compressed, ProgramTensorMetadataDependency::Type},
                  {head_weights, ProgramTensorMetadataDependency::Type},
                  {position_ids, ProgramTensorMetadataDependency::Type},
                  {cos_cache, ProgramTensorMetadataDependency::Type},
                  {sin_cache, ProgramTensorMetadataDependency::Type}})
      .AddOutput({selected, ProgramTensorMetadataDependency::Type})
      .SetWorkgroupSize(kWorkgroupSize)
      .SetDispatchGroupSize(ToUint32(rows))
      .AddUniformVariables({{ToUint32(rows)},
                            {ToUint32(sequence_length)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary_width)},
                            {ToUint32(max_rotary_length)},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(capacity)},
                            {ToUint32(present_compressed_length)},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))},
                            {has_head_weight_scale_
                                 ? head_weight_scale_
                                 : 1.0f / std::sqrt(static_cast<float>(num_heads))}});
  return context.RunProgram(select);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
