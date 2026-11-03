// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/bert/sparse_attention_indexer.h"

#include <bit>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/platform/env_var_utils.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

namespace sai = onnxruntime::contrib::sparse_attention_indexer;
using onnxruntime::webgpu::ProgramTensorMetadataDependency;
using onnxruntime::webgpu::ShaderUsage;
using onnxruntime::webgpu::ShaderVariableHelper;
using onnxruntime::webgpu::WebGpuSupportedFloatTypes;

ONNX_OPERATOR_KERNEL_EX(
    SparseAttentionIndexer,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("TB", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),
    SparseAttentionIndexer);

namespace {

constexpr uint32_t kElementwiseWorkgroupSize = 64;
constexpr uint32_t kSerialWorkgroupSize = 1;
constexpr std::string_view kWgslNegativeMax =
    "const NEGATIVE_MAX_F32: f32 = -3.4028234663852886e+38;\n";
constexpr uint32_t kQsaSelectWorkgroupSize = 64;
// Both caches and the count fit within WebGPU's guaranteed 16 KiB workgroup storage.
constexpr uint32_t kQsaCachedBlocks = 768;
constexpr uint32_t kQsaCachedVisibleTokens = 3072;
constexpr uint32_t kQsaSharedHeadSize = 128;

class SparseAttentionIndexerQsaParallelRankProgram final
    : public Program<SparseAttentionIndexerQsaParallelRankProgram> {
 public:
  SparseAttentionIndexerQsaParallelRankProgram() : Program{"SparseAttentionIndexerQsaParallelRank"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"score_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32});

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& visible = shader.AddInput("visible", ShaderUsage::UseUniform);
    const auto& scores = shader.AddInput("scores", ShaderUsage::UseUniform);
    const auto& selected = shader.AddOutput("selected", ShaderUsage::UseUniform);
    shader.AdditionalImplementation()
        << "var<workgroup> ranks: array<u32, 64>;\n"
        << "var<workgroup> valid_counts: array<u32, 64>;\n";
    shader.MainFunctionBody()
        << "  let candidate = workgroup_idx;\n"
        << "  if (candidate >= uniforms.score_stride) { return; }\n"
        << "  let visible_count = " << visible.GetByOffset("uniforms.total_sequence_length") << ";\n"
        << "  let block_count = visible_count / uniforms.compress_ratio;\n"
        << "  let selected_blocks = min(uniforms.block_topk, block_count);\n"
        << "  var first_score = 0.0;\n"
        << "  var score = 0.0;\n"
        << "  if (block_count > 0u) { first_score = " << scores.GetByOffset("0u") << "; }\n"
        << "  if (candidate < block_count) { score = " << scores.GetByOffset("candidate") << "; }\n"
        << "  var rank = 0u;\n"
        << "  var valid_count = 0u;\n"
        << "  if (candidate < block_count && first_score == first_score && score == score) {\n"
        << "    for (var other = local_idx; other < block_count; other += 64u) {\n"
        << "      let other_score = " << scores.GetByOffset("other") << ";\n"
        << "      if (candidate == 0u && other_score == other_score) { valid_count++; }\n"
        << "      if (other_score > score || (other_score == score && other < candidate)) { rank++; }\n"
        << "    }\n"
        << "  }\n"
        << "  ranks[local_idx] = rank;\n"
        << "  valid_counts[local_idx] = valid_count;\n"
        << "  workgroupBarrier();\n"
        << "  for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
        << "    if (local_idx < stride) {\n"
        << "      ranks[local_idx] += ranks[local_idx + stride];\n"
        << "      valid_counts[local_idx] += valid_counts[local_idx + stride];\n"
        << "    }\n"
        << "    workgroupBarrier();\n"
        << "  }\n"
        << "  if (local_idx == 0u) {\n"
        << "    if (candidate == 0u) {\n"
        << "      let tail_start = block_count * uniforms.compress_ratio;\n"
        << "      let tail_count = visible_count - tail_start;\n"
        << "      var valid_selected_blocks = min(selected_blocks, valid_counts[0]);\n"
        << "      if (first_score != first_score) { valid_selected_blocks = min(selected_blocks, 1u); }\n"
        // Initialization touches only slots that no candidate workgroup can select.
        << "      for (var slot = valid_selected_blocks * uniforms.compress_ratio;\n"
        << "           slot < selected_blocks * uniforms.compress_ratio; slot++) {\n"
        << "        " << selected.SetByOffset("slot", "-1") << "\n"
        << "      }\n"
        << "      for (var slot = selected_blocks * uniforms.compress_ratio + tail_count;\n"
        << "           slot < uniforms.capacity; slot++) {\n"
        << "        " << selected.SetByOffset("slot", "-1") << "\n"
        << "      }\n"
        << "      for (var tail = 0u; tail < visible_count - tail_start; tail++) {\n"
        << "        " << selected.SetByOffset(
                            "selected_blocks * uniforms.compress_ratio + tail",
                            "i32(" + visible.GetByOffset("tail_start + tail") + ")")
        << "\n"
        << "      }\n"
        << "    }\n"
        << "    if (candidate >= block_count || score != score) {\n"
        << "      if (!(candidate == 0u && block_count > 0u && first_score != first_score)) { return; }\n"
        << "    }\n"
        << "    if (first_score != first_score) {\n"
        << "      if (candidate != 0u) { return; }\n"
        << "      rank = 0u;\n"
        << "    } else { rank = ranks[0]; }\n"
        << "    if (rank < selected_blocks) {\n"
        << "      for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
        << "        " << selected.SetByOffset(
                            "rank * uniforms.compress_ratio + slot",
                            "i32(" + visible.GetByOffset("candidate * uniforms.compress_ratio + slot") + ")")
        << "\n"
        << "      }\n"
        << "    }\n"
        << "  }\n";
    return Status::OK();
  }
};

Status CheckShape(const Tensor* tensor, const char* name, std::initializer_list<int64_t> expected) {
  ORT_RETURN_IF(tensor == nullptr, "SparseAttentionIndexer: ", name, " is required");
  const TensorShape expected_shape(expected);
  ORT_RETURN_IF_NOT(tensor->Shape() == expected_shape, "SparseAttentionIndexer: ", name, " must have shape ",
                    expected_shape.ToString(), ", got ", tensor->Shape().ToString());
  return Status::OK();
}

struct RotaryCacheShape {
  bool batched;
  int64_t max_rotary_length;
  int64_t rotary_width;
};

Status CheckRotaryCache(const Tensor* cos_cache, const Tensor* sin_cache, int64_t batch_size,
                        RotaryCacheShape& out) {
  ORT_RETURN_IF(cos_cache == nullptr, "SparseAttentionIndexer: cos_cache is required");
  const auto& cos_shape = cos_cache->Shape();
  out.batched = cos_shape.NumDimensions() == 3;
  ORT_RETURN_IF_NOT(
      (out.batched && cos_shape[0] == batch_size && cos_shape[1] > 0) ||
          (cos_shape.NumDimensions() == 2 && cos_shape[0] > 0),
      "SparseAttentionIndexer: cos_cache must have shape (max_rotary_sequence_length, rotary_width) or "
      "(batch_size, max_rotary_sequence_length, rotary_width), got ",
      cos_shape.ToString());
  out.max_rotary_length = out.batched ? cos_shape[1] : cos_shape[0];
  out.rotary_width = out.batched ? cos_shape[2] : cos_shape[1];
  ORT_RETURN_IF_NOT(sin_cache != nullptr && sin_cache->Shape() == cos_shape,
                    "SparseAttentionIndexer: sin_cache must have the same shape as cos_cache");
  return Status::OK();
}

Status CheckUint32(std::initializer_list<std::pair<int64_t, const char*>> values) {
  for (const auto& [value, name] : values) {
    ORT_RETURN_IF(value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<uint32_t>::max(),
                  "SparseAttentionIndexer WebGPU: ", name, " must fit in uint32, got ", value);
  }
  return Status::OK();
}

Status CheckUint32Product(const char* name, std::initializer_list<int64_t> factors) {
  uint64_t product = 1;
  for (const int64_t factor : factors) {
    ORT_RETURN_IF(factor < 0 || (factor != 0 && product > std::numeric_limits<uint32_t>::max() /
                                                              static_cast<uint64_t>(factor)),
                  "SparseAttentionIndexer WebGPU: ", name, " must fit in uint32");
    product *= static_cast<uint64_t>(factor);
  }
  return Status::OK();
}

uint32_t ToUint32(int64_t value) {
  return static_cast<uint32_t>(value);
}

uint32_t DivideRoundUp(uint32_t value, uint32_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

}  // namespace

Status SparseAttentionIndexerFillProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& output = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  " << output.SetByOffset("global_idx", "-1") << "\n";
  return Status::OK();
}

class QsaVectorConcatProgram final : public Program<QsaVectorConcatProgram> {
 public:
  QsaVectorConcatProgram(bool has_past, bool has_current)
      : Program{"QsaConcatVector4"}, has_past_{has_past}, has_current_{has_current} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"key_row_stride", ProgramUniformVariableDataType::Uint32},
      {"key_offset", ProgramUniformVariableDataType::Uint32});
 private:
  bool has_past_;
  bool has_current_;
};

Status QsaVectorConcatProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper* past = nullptr;
  const ShaderVariableHelper* current = nullptr;
  if (has_past_) {
    past = &shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (has_current_) {
    current = &shader.AddInput("key", ShaderUsage::UseUniform);
  }
  const auto& present = shader.AddOutput("present_key",
                                         ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);

  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let d = global_idx % uniforms.head_size;\n"
      << "  let token_row = global_idx / uniforms.head_size;\n"
      << "  let token = token_row % uniforms.total_sequence_length;\n"
      << "  let batch = token_row / uniforms.total_sequence_length;\n";
  if (has_past_) {
    shader.MainFunctionBody()
        << "  if (token < uniforms.past_sequence_length) {\n"
        << "    " << present.SetByOffset("global_idx", "present_key_value_t(" + past->GetByOffset("(batch * uniforms.past_sequence_length + token) * uniforms.head_size + d") + ")")
        << "\n"
        << "    return;\n"
        << "  }\n";
  }
  if (has_current_) {
    shader.MainFunctionBody()
        << "  let current_token = token - uniforms.past_sequence_length;\n"
        << "  let current_row = batch * uniforms.sequence_length + current_token;\n"
        << "  " << present.SetByOffset("global_idx", "present_key_value_t(" + current->GetByOffset("current_row * uniforms.key_row_stride + uniforms.key_offset + d") + ")")
        << "\n";
  }
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
        << "  let current_row = batch * uniforms.sequence_length + current_token;\n"
        << "  " << present.SetByOffset("global_idx", "present_key_element_t(" + current->GetByOffset("current_row * uniforms.key_row_stride + uniforms.key_offset + d") + ")")
        << "\n";
  }
  return Status::OK();
}

class QsaPackedScore16Program final : public Program<QsaPackedScore16Program> {
 public:
  explicit QsaPackedScore16Program(bool has_mask)
      : Program{"QsaPackedScore16QueryWide"}, has_mask_{has_mask} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_row_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
 private:
  bool has_mask_;
  const bool use_global_cache_ = true;
  const QsaCachePhase phase_ = QsaCachePhase::Score;
  const uint32_t sort_capacity_ = 0;
  const bool dense_map_ = false;
};

Status QsaPackedScore16Program::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform);
  const auto& query_norm = shader.AddInput("query_norm_weight", ShaderUsage::UseUniform);
  const auto& key_norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& selected = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);

  const ShaderVariableHelper* visible_cache = nullptr;
  const ShaderVariableHelper* score_cache = nullptr;
  if (use_global_cache_) {
    visible_cache = &shader.AddOutput("visible_cache", ShaderUsage::UseUniform);
    score_cache = &shader.AddOutput("score_cache", ShaderUsage::UseUniform);
  } else {
    shader.AdditionalImplementation()
        << "var<workgroup> block_scores: array<f32, " << kQsaCachedBlocks << ">;\n"
        << "var<workgroup> visible_tokens: array<u32, " << kQsaCachedVisibleTokens << ">;\n";
  }
  shader.AdditionalImplementation()
      << "var<workgroup> visible_count_shared: u32;\n"
      << "fn visible(row: u32, token: u32) -> bool {\n"
      << "  let query = row % uniforms.sequence_length;\n";
  if (has_mask_) {
    const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform);
    shader.AdditionalImplementation()
        << "  let batch = row / uniforms.sequence_length;\n"
        << "  let offset = batch * uniforms.total_sequence_length + token;\n"
        << "  let mask_value = " << mask.GetByOffset("offset", true) << ";\n"
        << "  return token <= uniforms.past_sequence_length + query && "
           "(mask_value.x != 0u || mask_value.y != 0u);\n";
  } else {
    shader.AdditionalImplementation()
        << "  return token <= uniforms.past_sequence_length + query;\n";
  }
  shader.AdditionalImplementation() << "}\n";
  if (use_global_cache_) {
    shader.AdditionalImplementation()
        << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
        << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
        << "  return " << visible_cache->GetByOffset("row * (uniforms.total_sequence_length + 1u) + ordinal") << ";\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
      << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
      << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
      << "  if (ordinal < " << kQsaCachedVisibleTokens << "u) { return visible_tokens[ordinal]; }\n"
      << "  var seen = " << kQsaCachedVisibleTokens << "u;\n"
      << "  for (var token = visible_tokens[" << kQsaCachedVisibleTokens - 1 << "u] + 1u;\n"
      << "       token < uniforms.total_sequence_length; token++) {\n"
      << "    if (visible(row, token)) {\n"
      << "      if (seen == ordinal) { return token; }\n"
      << "      seen++;\n"
      << "    }\n"
      << "  }\n"
      << "  return uniforms.total_sequence_length;\n"
      << "}\n";
  }
  shader.AdditionalImplementation()
      << "fn clamp_position(position: u32) -> u32 {\n"
      << "  return min(position, uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn normalized_query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  let base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "  return f32(" << query.GetByOffset("base + d")
      << ") * inv_rms * f32("
      << query_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  var value = normalized_query_value(row, head, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_query_value(row, head, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let token = row % uniforms.sequence_length;\n"
      << "    let position = clamp_position(uniforms.past_sequence_length + token);\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
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
      << "fn normalized_value(row: u32, block: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  return pooled_value(row, block, d) * inv_rms * f32("
      << key_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn key_value(row: u32, block: u32, d: u32, inv_rms: f32, position: u32) -> f32 {\n"
      << "  var value = normalized_value(row, block, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_value(row, block, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(row: u32, block: u32) -> f32 {\n"
      << "  var key_square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = pooled_value(row, block, k);\n"
      << "    key_square_sum += value * value;\n"
      << "  }\n"
      << "  let key_inv_rms = inverseSqrt(key_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "  let position = clamp_position(visible_at(row, block * uniforms.compress_ratio));\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    let query_base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "    var query_square_sum = 0.0;\n"
      << "    for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "      let value = f32(" << query.GetByOffset("query_base + k") << ");\n"
      << "      query_square_sum += value * value;\n"
      << "    }\n"
      << "    let query_inv_rms = inverseSqrt(query_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(row, head, d, query_inv_rms) * key_value(row, block, d, key_inv_rms, position);\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  if (use_global_cache_) {
    const std::string score_offset =
        "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
    const std::string count_offset =
        "row * (uniforms.total_sequence_length + 1u) + uniforms.total_sequence_length";
    if (phase_ == QsaCachePhase::Score) {
      shader.AdditionalImplementation()
          << "var<workgroup> pooled_key: array<f32, " << 16 * (kQsaSharedHeadSize + 16) << ">;\n"
          << "var<workgroup> head_scores: array<f32, 64>;\n"
          << "var<workgroup> key_inverse_rms: array<f32, 16>;\n"
          << "var<workgroup> query_inverse_rms: array<f32, 4>;\n"
          << "var<workgroup> query_values: array<f32, " << 4 * (kQsaSharedHeadSize + 1) << ">;\n";
      shader.MainFunctionBody()
          << "  let stride = max(1u, uniforms.total_sequence_length / uniforms.compress_ratio);\n"
          << "  let group = local_idx / 16u;\n"
          << "  let local_lane = local_idx % 16u;\n"
          << "  let packed_stride = (stride + 15u) / 16u;\n"
          << "  let row = workgroup_idx / packed_stride;\n"
          << "  let candidate = (workgroup_idx % packed_stride) * 16u + group;\n"
          << "  if (row >= uniforms.rows) { return; }\n"
          << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_idx < uniforms.num_heads) {\n"
          << "    let query_base = row * uniforms.query_row_stride + local_idx * uniforms.head_size;\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      let value = f32(" << query.GetByOffset("query_base + d") << ");\n"
          << "      square_sum += value * value;\n"
          << "    }\n"
          << "    query_inverse_rms[local_idx] = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var i = local_idx; i < uniforms.num_heads * uniforms.head_size; i += 256u) {\n"
          << "    let head = i / uniforms.head_size;\n"
          << "    let d = i % uniforms.head_size;\n"
          << "    query_values[head * 129u + d] = query_value(row, head, d, query_inverse_rms[head]);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  let is_valid_block = candidate < visible_count_shared / uniforms.compress_ratio;\n"
          << "  for (var d = local_lane; d < uniforms.head_size; d += 16u) {\n"
          << "    pooled_key[group * 144u + d] = 0.0;\n"
          << "    if (is_valid_block) { pooled_key[group * 144u + d] = pooled_value(row, candidate, d); }\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane == 0u) {\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) { square_sum += pooled_key[group * 144u + d] * pooled_key[group * 144u + d]; }\n"
          << "    key_inverse_rms[group] = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var d = local_lane; d < uniforms.head_size; d += 16u) {\n"
          << "    pooled_key[group * 144u + d] = pooled_key[group * 144u + d] * key_inverse_rms[group] * f32(" << key_norm.GetByOffset("d") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  var position = 0u;\n"
          << "  if (is_valid_block) { position = clamp_position(visible_at(row, candidate * uniforms.compress_ratio)); }\n"
          << "  let half = uniforms.rotary_width / 2u;\n"
          << "  let cache_base = ((row / uniforms.sequence_length) * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width;\n"
          << "  for (var d = local_lane; d < half; d += 16u) {\n"
          << "    let left = pooled_key[group * 144u + d];\n"
          << "    let right = pooled_key[group * 144u + d + half];\n"
          << "    pooled_key[group * 144u + d] = left * f32(" << cos_cache.GetByOffset("cache_base + d") << ") + "
                                                                                             "(-1.0 * right) * f32("
          << sin_cache.GetByOffset("cache_base + d") << ");\n"
          << "    pooled_key[group * 144u + d + half] = right * f32(" << cos_cache.GetByOffset("cache_base + d + half") << ") + "
                                                                                                            "left * f32("
          << sin_cache.GetByOffset("cache_base + d + half") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane < uniforms.num_heads && is_valid_block) {\n"
          << "    var dot = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      dot += query_values[local_lane * 129u + d] * pooled_key[group * 144u + d];\n"
          << "    }\n"
          << "    head_scores[group * 4u + local_lane] = max(dot, 0.0);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane == 0u && is_valid_block) {\n"
          << "    var score = 0.0;\n"
          << "    for (var head = 0u; head < uniforms.num_heads; head++) { score += head_scores[group * 4u + head]; }\n"
          << "    " << score_cache->SetByOffset(score_offset, "score * uniforms.scale") << "\n"
          << "  }\n";
      return Status::OK();
    }
    if (phase_ == QsaCachePhase::Map) {
      shader.AdditionalImplementation() << "var<workgroup> visible_flags: array<u32, 64>;\n";
    } else if (sort_capacity_ > 0) {
      shader.AdditionalImplementation()
          << "var<workgroup> sort_scores: array<f32, " << sort_capacity_ << ">;\n"
          << "var<workgroup> sort_indices: array<i32, " << sort_capacity_ << ">;\n"
          << "fn qsa_before(left_score: f32, left_index: i32, right_score: f32, right_index: i32) -> bool {\n"
          << "  if (left_index < 0) { return false; }\n"
          << "  if (right_index < 0) { return true; }\n"
          << "  if (left_score != left_score) { return true; }\n"
          << "  if (right_score != right_score) { return false; }\n"
          << "  return left_score > right_score || (left_score == right_score && left_index < right_index);\n"
          << "}\n";
    } else {
      shader.AdditionalImplementation()
          << "var<workgroup> best_scores: array<f32, 64>;\n"
          << "var<workgroup> best_indices: array<i32, 64>;\n"
          << "var<workgroup> previous_score_shared: f32;\n"
          << "var<workgroup> previous_index_shared: i32;\n";
    }
    shader.MainFunctionBody()
        << "  let row = workgroup_idx;\n"
        << "  if (row >= uniforms.rows) { return; }\n";
    if (phase_ == QsaCachePhase::Map) {
      if (dense_map_) {
        shader.MainFunctionBody()
            << "  var lane_count = 0u;\n"
            << "  for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "    lane_count += select(0u, 1u, visible(row, token));\n"
            << "  }\n"
            << "  visible_flags[local_idx] = lane_count;\n"
            << "  workgroupBarrier();\n"
            << "  for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
            << "    if (local_idx < stride) { visible_flags[local_idx] += visible_flags[local_idx + stride]; }\n"
            << "    workgroupBarrier();\n"
            << "  }\n"
            << "  if (local_idx == 0u) { visible_count_shared = visible_flags[0]; }\n"
            << "  let dense_count = workgroupUniformLoad(&visible_count_shared);\n"
            << "  if (dense_count == 0u || dense_count == uniforms.total_sequence_length) {\n"
            << "    if (dense_count != 0u) {\n"
            << "      for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "        " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + token", "token") << "\n"
            << "      }\n"
            << "    }\n"
            << "    storageBarrier();\n"
            << "    if (local_idx == 0u) { "
            << visible_cache->SetByOffset(count_offset, "dense_count") << " }\n"
            << "    return;\n"
            << "  }\n";
      }
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) { visible_count_shared = 0u; }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length; base += 64u) {\n"
          << "    let token = base + local_idx;\n"
          << "    visible_flags[local_idx] = select(0u, 1u, "
             "token < uniforms.total_sequence_length && visible(row, token));\n"
          << "    workgroupBarrier();\n"
          << "    var prefix = 0u;\n"
          << "    for (var lane = 0u; lane < local_idx; lane++) { prefix += visible_flags[lane]; }\n"
          << "    let chunk_base = visible_count_shared;\n"
          << "    if (visible_flags[local_idx] != 0u) {\n"
          << "      " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + chunk_base + prefix", "token") << "\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    if (local_idx == 63u) {\n"
          << "      visible_count_shared = chunk_base + prefix + visible_flags[local_idx];\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n"
          << "  storageBarrier();\n"
          << "  if (local_idx == 0u) { "
          << visible_cache->SetByOffset(count_offset, "visible_count_shared") << " }\n";
      return Status::OK();
    }
    shader.MainFunctionBody()
        << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
        << "  workgroupBarrier();\n"
        << "  let visible_count = visible_count_shared;\n"
        << "  let block_count = visible_count / uniforms.compress_ratio;\n"
        << "  let output_base = row * uniforms.capacity;\n"
        << "  for (var i = local_idx; i < uniforms.capacity; i += 64u) {\n"
        << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
        << "  }\n";
    if (sort_capacity_ == 0) {
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) {\n"
          << "    previous_score_shared = 0.0;\n"
          << "    previous_index_shared = -1i;\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  workgroupBarrier();\n"
        << "  storageBarrier();\n"
        << "  let selected_blocks = min(uniforms.block_topk, block_count);\n";
    if (sort_capacity_ > 0) {
      const uint32_t half_capacity = sort_capacity_ / 2;
      shader.MainFunctionBody()
          << "  for (var item = local_idx; item < " << sort_capacity_ << "u; item += 64u) {\n"
          << "    sort_scores[item] = 0.0;\n"
          << "    sort_indices[item] = -1i;\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length / uniforms.compress_ratio; base += "
          << half_capacity << "u) {\n"
          << "    for (var item = local_idx; item < " << half_capacity << "u; item += 64u) {\n"
          << "      let candidate = base + item;\n"
          << "      var score = 0.0;\n"
          << "      var index = -1i;\n"
          << "      if (candidate < block_count) {\n"
          << "        score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "        if (score == score || candidate == 0u) { index = i32(candidate); }\n"
          << "      }\n"
          << "      sort_scores[" << half_capacity << "u + item] = score;\n"
          << "      sort_indices[" << half_capacity << "u + item] = index;\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    for (var stage_size = 2u; stage_size <= " << sort_capacity_ << "u; stage_size *= 2u) {\n"
          << "      for (var distance = stage_size / 2u; distance > 0u; distance /= 2u) {\n"
          << "        for (var pair = local_idx; pair < " << half_capacity << "u; pair += 64u) {\n"
          << "          let left = (pair / distance) * (2u * distance) + pair % distance;\n"
          << "          let right = left + distance;\n"
          << "          let left_score = sort_scores[left];\n"
          << "          let right_score = sort_scores[right];\n"
          << "          let left_index = sort_indices[left];\n"
          << "          let right_index = sort_indices[right];\n"
          << "          let forward = (left & stage_size) == 0u;\n"
          << "          if ((forward && qsa_before(right_score, right_index, left_score, left_index)) ||\n"
          << "              (!forward && qsa_before(left_score, left_index, right_score, right_index))) {\n"
          << "            sort_scores[left] = right_score;\n"
          << "            sort_scores[right] = left_score;\n"
          << "            sort_indices[left] = right_index;\n"
          << "            sort_indices[right] = left_index;\n"
          << "          }\n"
          << "        }\n"
          << "        workgroupBarrier();\n"
          << "      }\n"
          << "    }\n"
          << "  }\n"
          << "  var first_is_nan = false;\n"
          << "  if (block_count > 0u) {\n"
          << "    let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "    first_is_nan = first_score != first_score;\n"
          << "  }\n"
          << "  for (var rank = local_idx; rank < selected_blocks; rank += 64u) {\n"
          << "    let chosen = sort_indices[rank];\n"
          << "    if (chosen >= 0 && (!first_is_nan || rank == 0u)) {\n"
          << "      for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "    }\n"
          << "  }\n";
    } else {
      shader.MainFunctionBody()
          << "  for (var rank = 0u; rank < uniforms.block_topk; rank++) {\n"
          << "    var best_score = 0.0;\n"
          << "    var best_index = -1i;\n"
          << "    for (var candidate = local_idx; candidate < block_count; candidate += 64u) {\n"
          << "      let score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "      if (score != score) { continue; }\n"
          << "      if (previous_index_shared >= 0 && !(score < previous_score_shared || "
             "(score == previous_score_shared && i32(candidate) > previous_index_shared))) { continue; }\n"
          << "      if (best_index < 0 || score > best_score || "
             "(score == best_score && i32(candidate) < best_index)) {\n"
          << "        best_score = score;\n"
          << "        best_index = i32(candidate);\n"
          << "      }\n"
          << "    }\n"
          << "    best_scores[local_idx] = best_score;\n"
          << "    best_indices[local_idx] = best_index;\n"
          << "    workgroupBarrier();\n"
          << "    for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
          << "      if (local_idx < stride) {\n"
          << "        let other = local_idx + stride;\n"
          << "        if (best_indices[other] >= 0 && (best_indices[local_idx] < 0 || "
             "best_scores[other] > best_scores[local_idx] || "
             "(best_scores[other] == best_scores[local_idx] && best_indices[other] < best_indices[local_idx]))) {\n"
          << "          best_scores[local_idx] = best_scores[other];\n"
          << "          best_indices[local_idx] = best_indices[other];\n"
          << "        }\n"
          << "      }\n"
          << "      workgroupBarrier();\n"
          << "    }\n"
          << "    if (local_idx == 0u && rank == 0u && block_count > 0u) {\n"
          << "      let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "      if (first_score != first_score) { best_scores[0] = first_score; best_indices[0] = 0i; }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    let chosen = best_indices[0];\n"
          << "    if (chosen >= 0) {\n"
          << "      for (var slot = local_idx; slot < uniforms.compress_ratio; slot += 64u) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "      if (local_idx == 0u) {\n"
          << "        previous_score_shared = best_scores[0];\n"
          << "        previous_index_shared = chosen;\n"
          << "      }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  let tail_start = block_count * uniforms.compress_ratio;\n"
        << "  for (var tail = local_idx; tail < visible_count - tail_start; tail += 64u) {\n"
        << "    let output = selected_blocks * uniforms.compress_ratio + tail;\n"
        << "    " << selected.SetByOffset("output_base + output", "i32(visible_at(row, tail_start + tail))") << "\n"
        << "  }\n";
    return Status::OK();
  }

  const std::string cache_condition =
      use_global_cache_ ? "true" : "block_count <= " + std::to_string(kQsaCachedBlocks) + "u";
  const std::string visible_store = use_global_cache_
                                        ? visible_cache->SetByOffset("row * uniforms.total_sequence_length + count", "token")
                                        : "if (count < " + std::to_string(kQsaCachedVisibleTokens) + "u) { visible_tokens[count] = token; }";
  const std::string score_offset =
      "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
  const std::string score_store = use_global_cache_
                                      ? score_cache->SetByOffset(score_offset, "block_score(row, candidate)")
                                      : "block_scores[candidate] = block_score(row, candidate);";
  const std::string score_load =
      use_global_cache_ ? score_cache->GetByOffset(score_offset) : "block_scores[candidate]";
  const std::string storage_barrier = use_global_cache_ ? "  storageBarrier();\n" : "";
  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  if (local_idx == 0u) {\n"
      << "    var count = 0u;\n"
      << "    for (var token = 0u; token < uniforms.total_sequence_length; token++) {\n"
      << "      if (visible(row, token)) {\n"
      << "        " << visible_store << "\n"
      << "        count++;\n"
      << "      }\n"
      << "    }\n"
      << "    visible_count_shared = count;\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  let visible_count = visible_count_shared;\n"
      << "  let block_count = visible_count / uniforms.compress_ratio;\n"
      << "  if (" << cache_condition << ") {\n"
      << "    for (var candidate = local_idx; candidate < block_count; candidate += "
      << kQsaSelectWorkgroupSize << "u) {\n"
      << "      " << score_store << "\n"
      << "    }\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  if (local_idx != 0u) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let selected_blocks = min(uniforms.block_topk, block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < selected_blocks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < block_count; candidate++) {\n"
      << "      var score = 0.0;\n"
      << "      if (" << cache_condition << ") {\n"
      << "        score = " << score_load << ";\n"
      << "      } else {\n"
      << "        score = block_score(row, candidate);\n"
      << "      }\n"
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
class QsaPackedScoreProgram final : public Program<QsaPackedScoreProgram> {
 public:
  explicit QsaPackedScoreProgram(bool has_mask)
      : Program{"QsaPackedScore4"}, has_mask_{has_mask} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"num_heads", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"query_row_stride", ProgramUniformVariableDataType::Uint32},
      {"rotary_width", ProgramUniformVariableDataType::Uint32},
      {"max_rotary_length", ProgramUniformVariableDataType::Uint32},
      {"rotary_cache_batch_stride", ProgramUniformVariableDataType::Uint32},
      {"compress_ratio", ProgramUniformVariableDataType::Uint32},
      {"capacity", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"block_topk", ProgramUniformVariableDataType::Uint32},
      {"epsilon", ProgramUniformVariableDataType::Float32},
      {"scale", ProgramUniformVariableDataType::Float32});
 private:
  bool has_mask_;
  const bool use_global_cache_ = true;
  const QsaCachePhase phase_ = QsaCachePhase::Score;
  const uint32_t sort_capacity_ = 0;
  const bool dense_map_ = false;
};

Status QsaPackedScoreProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform);
  const auto& query_norm = shader.AddInput("query_norm_weight", ShaderUsage::UseUniform);
  const auto& key_norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& selected = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);

  const ShaderVariableHelper* visible_cache = nullptr;
  const ShaderVariableHelper* score_cache = nullptr;
  if (use_global_cache_) {
    visible_cache = &shader.AddOutput("visible_cache", ShaderUsage::UseUniform);
    score_cache = &shader.AddOutput("score_cache", ShaderUsage::UseUniform);
  } else {
    shader.AdditionalImplementation()
        << "var<workgroup> block_scores: array<f32, " << kQsaCachedBlocks << ">;\n"
        << "var<workgroup> visible_tokens: array<u32, " << kQsaCachedVisibleTokens << ">;\n";
  }
  shader.AdditionalImplementation()
      << "var<workgroup> visible_count_shared: u32;\n"
      << "fn visible(row: u32, token: u32) -> bool {\n"
      << "  let query = row % uniforms.sequence_length;\n";
  if (has_mask_) {
    const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform);
    shader.AdditionalImplementation()
        << "  let batch = row / uniforms.sequence_length;\n"
        << "  let offset = batch * uniforms.total_sequence_length + token;\n"
        << "  let mask_value = " << mask.GetByOffset("offset", true) << ";\n"
        << "  return token <= uniforms.past_sequence_length + query && "
           "(mask_value.x != 0u || mask_value.y != 0u);\n";
  } else {
    shader.AdditionalImplementation()
        << "  return token <= uniforms.past_sequence_length + query;\n";
  }
  shader.AdditionalImplementation() << "}\n";
  if (use_global_cache_) {
    shader.AdditionalImplementation()
        << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
        << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
        << "  return " << visible_cache->GetByOffset("row * (uniforms.total_sequence_length + 1u) + ordinal") << ";\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
      << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
      << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
      << "  if (ordinal < " << kQsaCachedVisibleTokens << "u) { return visible_tokens[ordinal]; }\n"
      << "  var seen = " << kQsaCachedVisibleTokens << "u;\n"
      << "  for (var token = visible_tokens[" << kQsaCachedVisibleTokens - 1 << "u] + 1u;\n"
      << "       token < uniforms.total_sequence_length; token++) {\n"
      << "    if (visible(row, token)) {\n"
      << "      if (seen == ordinal) { return token; }\n"
      << "      seen++;\n"
      << "    }\n"
      << "  }\n"
      << "  return uniforms.total_sequence_length;\n"
      << "}\n";
  }
  shader.AdditionalImplementation()
      << "fn clamp_position(position: u32) -> u32 {\n"
      << "  return min(position, uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn normalized_query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  let base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "  return f32(" << query.GetByOffset("base + d")
      << ") * inv_rms * f32("
      << query_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  var value = normalized_query_value(row, head, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_query_value(row, head, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let token = row % uniforms.sequence_length;\n"
      << "    let position = clamp_position(uniforms.past_sequence_length + token);\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
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
      << "fn normalized_value(row: u32, block: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  return pooled_value(row, block, d) * inv_rms * f32("
      << key_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn key_value(row: u32, block: u32, d: u32, inv_rms: f32, position: u32) -> f32 {\n"
      << "  var value = normalized_value(row, block, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_value(row, block, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(row: u32, block: u32) -> f32 {\n"
      << "  var key_square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = pooled_value(row, block, k);\n"
      << "    key_square_sum += value * value;\n"
      << "  }\n"
      << "  let key_inv_rms = inverseSqrt(key_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "  let position = clamp_position(visible_at(row, block * uniforms.compress_ratio));\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    let query_base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "    var query_square_sum = 0.0;\n"
      << "    for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "      let value = f32(" << query.GetByOffset("query_base + k") << ");\n"
      << "      query_square_sum += value * value;\n"
      << "    }\n"
      << "    let query_inv_rms = inverseSqrt(query_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(row, head, d, query_inv_rms) * key_value(row, block, d, key_inv_rms, position);\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  if (use_global_cache_) {
    const std::string score_offset =
        "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
    const std::string count_offset =
        "row * (uniforms.total_sequence_length + 1u) + uniforms.total_sequence_length";
    if (phase_ == QsaCachePhase::Score) {
      shader.AdditionalImplementation()
          << "var<workgroup> pooled_key: array<f32, " << 4 * kQsaSharedHeadSize << ">;\n"
          << "var<workgroup> head_scores: array<f32, 256>;\n"
          << "var<workgroup> key_inverse_rms: array<f32, 4>;\n";
      shader.MainFunctionBody()
          << "  let stride = max(1u, uniforms.total_sequence_length / uniforms.compress_ratio);\n"
          << "  let group = local_idx / 16u;\n"
          << "  let local_lane = local_idx % 16u;\n"
          << "  let packed_stride = (stride + 3u) / 4u;\n"
          << "  let row = workgroup_idx / packed_stride;\n"
          << "  let candidate = (workgroup_idx % packed_stride) * 4u + group;\n"
          << "  if (row >= uniforms.rows) { return; }\n"
          << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
          << "  workgroupBarrier();\n"
          << "  let is_valid_block = candidate < visible_count_shared / uniforms.compress_ratio;\n"
          << "  for (var d = local_lane; d < uniforms.head_size; d += 16u) {\n"
          << "    pooled_key[group * 128u + d] = 0.0;\n"
          << "    if (is_valid_block) { pooled_key[group * 128u + d] = pooled_value(row, candidate, d); }\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane == 0u) {\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) { square_sum += pooled_key[group * 128u + d] * pooled_key[group * 128u + d]; }\n"
          << "    key_inverse_rms[group] = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var d = local_lane; d < uniforms.head_size; d += 16u) {\n"
          << "    pooled_key[group * 128u + d] = pooled_key[group * 128u + d] * key_inverse_rms[group] * f32(" << key_norm.GetByOffset("d") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  var position = 0u;\n"
          << "  if (is_valid_block) { position = clamp_position(visible_at(row, candidate * uniforms.compress_ratio)); }\n"
          << "  let half = uniforms.rotary_width / 2u;\n"
          << "  let cache_base = ((row / uniforms.sequence_length) * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width;\n"
          << "  for (var d = local_lane; d < half; d += 16u) {\n"
          << "    let left = pooled_key[group * 128u + d];\n"
          << "    let right = pooled_key[group * 128u + d + half];\n"
          << "    pooled_key[group * 128u + d] = left * f32(" << cos_cache.GetByOffset("cache_base + d") << ") + "
                                                                                             "(-1.0 * right) * f32("
          << sin_cache.GetByOffset("cache_base + d") << ");\n"
          << "    pooled_key[group * 128u + d + half] = right * f32(" << cos_cache.GetByOffset("cache_base + d + half") << ") + "
                                                                                                            "left * f32("
          << sin_cache.GetByOffset("cache_base + d + half") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane < uniforms.num_heads && is_valid_block) {\n"
          << "    let query_base = row * uniforms.query_row_stride + local_lane * uniforms.head_size;\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      let value = f32(" << query.GetByOffset("query_base + d") << ");\n"
          << "      square_sum += value * value;\n"
          << "    }\n"
          << "    let inv_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "    var dot = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      dot += query_value(row, local_lane, d, inv_rms) * pooled_key[group * 128u + d];\n"
          << "    }\n"
          << "    head_scores[group * 64u + local_lane] = max(dot, 0.0);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_lane == 0u && is_valid_block) {\n"
          << "    var score = 0.0;\n"
          << "    for (var head = 0u; head < uniforms.num_heads; head++) { score += head_scores[group * 64u + head]; }\n"
          << "    " << score_cache->SetByOffset(score_offset, "score * uniforms.scale") << "\n"
          << "  }\n";
      return Status::OK();
    }
    if (phase_ == QsaCachePhase::Map) {
      shader.AdditionalImplementation() << "var<workgroup> visible_flags: array<u32, 64>;\n";
    } else if (sort_capacity_ > 0) {
      shader.AdditionalImplementation()
          << "var<workgroup> sort_scores: array<f32, " << sort_capacity_ << ">;\n"
          << "var<workgroup> sort_indices: array<i32, " << sort_capacity_ << ">;\n"
          << "fn qsa_before(left_score: f32, left_index: i32, right_score: f32, right_index: i32) -> bool {\n"
          << "  if (left_index < 0) { return false; }\n"
          << "  if (right_index < 0) { return true; }\n"
          << "  if (left_score != left_score) { return true; }\n"
          << "  if (right_score != right_score) { return false; }\n"
          << "  return left_score > right_score || (left_score == right_score && left_index < right_index);\n"
          << "}\n";
    } else {
      shader.AdditionalImplementation()
          << "var<workgroup> best_scores: array<f32, 64>;\n"
          << "var<workgroup> best_indices: array<i32, 64>;\n"
          << "var<workgroup> previous_score_shared: f32;\n"
          << "var<workgroup> previous_index_shared: i32;\n";
    }
    shader.MainFunctionBody()
        << "  let row = workgroup_idx;\n"
        << "  if (row >= uniforms.rows) { return; }\n";
    if (phase_ == QsaCachePhase::Map) {
      if (dense_map_) {
        shader.MainFunctionBody()
            << "  var lane_count = 0u;\n"
            << "  for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "    lane_count += select(0u, 1u, visible(row, token));\n"
            << "  }\n"
            << "  visible_flags[local_idx] = lane_count;\n"
            << "  workgroupBarrier();\n"
            << "  for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
            << "    if (local_idx < stride) { visible_flags[local_idx] += visible_flags[local_idx + stride]; }\n"
            << "    workgroupBarrier();\n"
            << "  }\n"
            << "  if (local_idx == 0u) { visible_count_shared = visible_flags[0]; }\n"
            << "  let dense_count = workgroupUniformLoad(&visible_count_shared);\n"
            << "  if (dense_count == 0u || dense_count == uniforms.total_sequence_length) {\n"
            << "    if (dense_count != 0u) {\n"
            << "      for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "        " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + token", "token") << "\n"
            << "      }\n"
            << "    }\n"
            << "    storageBarrier();\n"
            << "    if (local_idx == 0u) { "
            << visible_cache->SetByOffset(count_offset, "dense_count") << " }\n"
            << "    return;\n"
            << "  }\n";
      }
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) { visible_count_shared = 0u; }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length; base += 64u) {\n"
          << "    let token = base + local_idx;\n"
          << "    visible_flags[local_idx] = select(0u, 1u, "
             "token < uniforms.total_sequence_length && visible(row, token));\n"
          << "    workgroupBarrier();\n"
          << "    var prefix = 0u;\n"
          << "    for (var lane = 0u; lane < local_idx; lane++) { prefix += visible_flags[lane]; }\n"
          << "    let chunk_base = visible_count_shared;\n"
          << "    if (visible_flags[local_idx] != 0u) {\n"
          << "      " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + chunk_base + prefix", "token") << "\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    if (local_idx == 63u) {\n"
          << "      visible_count_shared = chunk_base + prefix + visible_flags[local_idx];\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n"
          << "  storageBarrier();\n"
          << "  if (local_idx == 0u) { "
          << visible_cache->SetByOffset(count_offset, "visible_count_shared") << " }\n";
      return Status::OK();
    }
    shader.MainFunctionBody()
        << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
        << "  workgroupBarrier();\n"
        << "  let visible_count = visible_count_shared;\n"
        << "  let block_count = visible_count / uniforms.compress_ratio;\n"
        << "  let output_base = row * uniforms.capacity;\n"
        << "  for (var i = local_idx; i < uniforms.capacity; i += 64u) {\n"
        << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
        << "  }\n";
    if (sort_capacity_ == 0) {
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) {\n"
          << "    previous_score_shared = 0.0;\n"
          << "    previous_index_shared = -1i;\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  workgroupBarrier();\n"
        << "  storageBarrier();\n"
        << "  let selected_blocks = min(uniforms.block_topk, block_count);\n";
    if (sort_capacity_ > 0) {
      const uint32_t half_capacity = sort_capacity_ / 2;
      shader.MainFunctionBody()
          << "  for (var item = local_idx; item < " << sort_capacity_ << "u; item += 64u) {\n"
          << "    sort_scores[item] = 0.0;\n"
          << "    sort_indices[item] = -1i;\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length / uniforms.compress_ratio; base += "
          << half_capacity << "u) {\n"
          << "    for (var item = local_idx; item < " << half_capacity << "u; item += 64u) {\n"
          << "      let candidate = base + item;\n"
          << "      var score = 0.0;\n"
          << "      var index = -1i;\n"
          << "      if (candidate < block_count) {\n"
          << "        score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "        if (score == score || candidate == 0u) { index = i32(candidate); }\n"
          << "      }\n"
          << "      sort_scores[" << half_capacity << "u + item] = score;\n"
          << "      sort_indices[" << half_capacity << "u + item] = index;\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    for (var stage_size = 2u; stage_size <= " << sort_capacity_ << "u; stage_size *= 2u) {\n"
          << "      for (var distance = stage_size / 2u; distance > 0u; distance /= 2u) {\n"
          << "        for (var pair = local_idx; pair < " << half_capacity << "u; pair += 64u) {\n"
          << "          let left = (pair / distance) * (2u * distance) + pair % distance;\n"
          << "          let right = left + distance;\n"
          << "          let left_score = sort_scores[left];\n"
          << "          let right_score = sort_scores[right];\n"
          << "          let left_index = sort_indices[left];\n"
          << "          let right_index = sort_indices[right];\n"
          << "          let forward = (left & stage_size) == 0u;\n"
          << "          if ((forward && qsa_before(right_score, right_index, left_score, left_index)) ||\n"
          << "              (!forward && qsa_before(left_score, left_index, right_score, right_index))) {\n"
          << "            sort_scores[left] = right_score;\n"
          << "            sort_scores[right] = left_score;\n"
          << "            sort_indices[left] = right_index;\n"
          << "            sort_indices[right] = left_index;\n"
          << "          }\n"
          << "        }\n"
          << "        workgroupBarrier();\n"
          << "      }\n"
          << "    }\n"
          << "  }\n"
          << "  var first_is_nan = false;\n"
          << "  if (block_count > 0u) {\n"
          << "    let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "    first_is_nan = first_score != first_score;\n"
          << "  }\n"
          << "  for (var rank = local_idx; rank < selected_blocks; rank += 64u) {\n"
          << "    let chosen = sort_indices[rank];\n"
          << "    if (chosen >= 0 && (!first_is_nan || rank == 0u)) {\n"
          << "      for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "    }\n"
          << "  }\n";
    } else {
      shader.MainFunctionBody()
          << "  for (var rank = 0u; rank < uniforms.block_topk; rank++) {\n"
          << "    var best_score = 0.0;\n"
          << "    var best_index = -1i;\n"
          << "    for (var candidate = local_idx; candidate < block_count; candidate += 64u) {\n"
          << "      let score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "      if (score != score) { continue; }\n"
          << "      if (previous_index_shared >= 0 && !(score < previous_score_shared || "
             "(score == previous_score_shared && i32(candidate) > previous_index_shared))) { continue; }\n"
          << "      if (best_index < 0 || score > best_score || "
             "(score == best_score && i32(candidate) < best_index)) {\n"
          << "        best_score = score;\n"
          << "        best_index = i32(candidate);\n"
          << "      }\n"
          << "    }\n"
          << "    best_scores[local_idx] = best_score;\n"
          << "    best_indices[local_idx] = best_index;\n"
          << "    workgroupBarrier();\n"
          << "    for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
          << "      if (local_idx < stride) {\n"
          << "        let other = local_idx + stride;\n"
          << "        if (best_indices[other] >= 0 && (best_indices[local_idx] < 0 || "
             "best_scores[other] > best_scores[local_idx] || "
             "(best_scores[other] == best_scores[local_idx] && best_indices[other] < best_indices[local_idx]))) {\n"
          << "          best_scores[local_idx] = best_scores[other];\n"
          << "          best_indices[local_idx] = best_indices[other];\n"
          << "        }\n"
          << "      }\n"
          << "      workgroupBarrier();\n"
          << "    }\n"
          << "    if (local_idx == 0u && rank == 0u && block_count > 0u) {\n"
          << "      let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "      if (first_score != first_score) { best_scores[0] = first_score; best_indices[0] = 0i; }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    let chosen = best_indices[0];\n"
          << "    if (chosen >= 0) {\n"
          << "      for (var slot = local_idx; slot < uniforms.compress_ratio; slot += 64u) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "      if (local_idx == 0u) {\n"
          << "        previous_score_shared = best_scores[0];\n"
          << "        previous_index_shared = chosen;\n"
          << "      }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  let tail_start = block_count * uniforms.compress_ratio;\n"
        << "  for (var tail = local_idx; tail < visible_count - tail_start; tail += 64u) {\n"
        << "    let output = selected_blocks * uniforms.compress_ratio + tail;\n"
        << "    " << selected.SetByOffset("output_base + output", "i32(visible_at(row, tail_start + tail))") << "\n"
        << "  }\n";
    return Status::OK();
  }

  const std::string cache_condition =
      use_global_cache_ ? "true" : "block_count <= " + std::to_string(kQsaCachedBlocks) + "u";
  const std::string visible_store = use_global_cache_
                                        ? visible_cache->SetByOffset("row * uniforms.total_sequence_length + count", "token")
                                        : "if (count < " + std::to_string(kQsaCachedVisibleTokens) + "u) { visible_tokens[count] = token; }";
  const std::string score_offset =
      "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
  const std::string score_store = use_global_cache_
                                      ? score_cache->SetByOffset(score_offset, "block_score(row, candidate)")
                                      : "block_scores[candidate] = block_score(row, candidate);";
  const std::string score_load =
      use_global_cache_ ? score_cache->GetByOffset(score_offset) : "block_scores[candidate]";
  const std::string storage_barrier = use_global_cache_ ? "  storageBarrier();\n" : "";
  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  if (local_idx == 0u) {\n"
      << "    var count = 0u;\n"
      << "    for (var token = 0u; token < uniforms.total_sequence_length; token++) {\n"
      << "      if (visible(row, token)) {\n"
      << "        " << visible_store << "\n"
      << "        count++;\n"
      << "      }\n"
      << "    }\n"
      << "    visible_count_shared = count;\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  let visible_count = visible_count_shared;\n"
      << "  let block_count = visible_count / uniforms.compress_ratio;\n"
      << "  if (" << cache_condition << ") {\n"
      << "    for (var candidate = local_idx; candidate < block_count; candidate += "
      << kQsaSelectWorkgroupSize << "u) {\n"
      << "      " << score_store << "\n"
      << "    }\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  if (local_idx != 0u) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let selected_blocks = min(uniforms.block_topk, block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < selected_blocks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < block_count; candidate++) {\n"
      << "      var score = 0.0;\n"
      << "      if (" << cache_condition << ") {\n"
      << "        score = " << score_load << ";\n"
      << "      } else {\n"
      << "        score = block_score(row, candidate);\n"
      << "      }\n"
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
Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& present_key = shader.AddInput("present_key", ShaderUsage::UseUniform);
  const auto& query_norm = shader.AddInput("query_norm_weight", ShaderUsage::UseUniform);
  const auto& key_norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& selected = shader.AddOutput("selected_indices", ShaderUsage::UseUniform);

  const ShaderVariableHelper* visible_cache = nullptr;
  const ShaderVariableHelper* score_cache = nullptr;
  if (use_global_cache_) {
    visible_cache = &shader.AddOutput("visible_cache", ShaderUsage::UseUniform);
    score_cache = &shader.AddOutput("score_cache", ShaderUsage::UseUniform);
  } else {
    shader.AdditionalImplementation()
        << "var<workgroup> block_scores: array<f32, " << kQsaCachedBlocks << ">;\n"
        << "var<workgroup> visible_tokens: array<u32, " << kQsaCachedVisibleTokens << ">;\n";
  }
  shader.AdditionalImplementation()
      << "var<workgroup> visible_count_shared: u32;\n"
      << "fn visible(row: u32, token: u32) -> bool {\n"
      << "  let query = row % uniforms.sequence_length;\n";
  if (has_mask_) {
    const auto& mask = shader.AddInput("mask", ShaderUsage::UseUniform);
    shader.AdditionalImplementation()
        << "  let batch = row / uniforms.sequence_length;\n"
        << "  let offset = batch * uniforms.total_sequence_length + token;\n"
        << "  let mask_value = " << mask.GetByOffset("offset", true) << ";\n"
        << "  return token <= uniforms.past_sequence_length + query && "
           "(mask_value.x != 0u || mask_value.y != 0u);\n";
  } else {
    shader.AdditionalImplementation()
        << "  return token <= uniforms.past_sequence_length + query;\n";
  }
  shader.AdditionalImplementation() << "}\n";
  if (use_global_cache_) {
    shader.AdditionalImplementation()
        << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
        << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
        << "  return " << visible_cache->GetByOffset("row * (uniforms.total_sequence_length + 1u) + ordinal") << ";\n"
        << "}\n";
  } else {
    shader.AdditionalImplementation()
      << "fn visible_at(row: u32, ordinal: u32) -> u32 {\n"
      << "  if (ordinal >= visible_count_shared) { return uniforms.total_sequence_length; }\n"
      << "  if (ordinal < " << kQsaCachedVisibleTokens << "u) { return visible_tokens[ordinal]; }\n"
      << "  var seen = " << kQsaCachedVisibleTokens << "u;\n"
      << "  for (var token = visible_tokens[" << kQsaCachedVisibleTokens - 1 << "u] + 1u;\n"
      << "       token < uniforms.total_sequence_length; token++) {\n"
      << "    if (visible(row, token)) {\n"
      << "      if (seen == ordinal) { return token; }\n"
      << "      seen++;\n"
      << "    }\n"
      << "  }\n"
      << "  return uniforms.total_sequence_length;\n"
      << "}\n";
  }
  shader.AdditionalImplementation()
      << "fn clamp_position(position: u32) -> u32 {\n"
      << "  return min(position, uniforms.max_rotary_length - 1u);\n"
      << "}\n"
      << "fn normalized_query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  let base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "  return f32(" << query.GetByOffset("base + d")
      << ") * inv_rms * f32("
      << query_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  var value = normalized_query_value(row, head, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_query_value(row, head, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let token = row % uniforms.sequence_length;\n"
      << "    let position = clamp_position(uniforms.past_sequence_length + token);\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
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
      << "fn normalized_value(row: u32, block: u32, d: u32, inv_rms: f32) -> f32 {\n"
      << "  return pooled_value(row, block, d) * inv_rms * f32("
      << key_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn key_value(row: u32, block: u32, d: u32, inv_rms: f32, position: u32) -> f32 {\n"
      << "  var value = normalized_value(row, block, d, inv_rms);\n"
      << "  if (d < uniforms.rotary_width) {\n"
      << "    let half = uniforms.rotary_width / 2u;\n"
      << "    let pair_d = select(d - half, d + half, d < half);\n"
      << "    let sign = select(1.0, -1.0, d < half);\n"
      << "    let paired = sign * normalized_value(row, block, pair_d, inv_rms);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + d;\n"
      << "    value = value * f32(" << cos_cache.GetByOffset("cache") << ") + paired * f32("
      << sin_cache.GetByOffset("cache") << ");\n"
      << "  }\n"
      << "  return value;\n"
      << "}\n"
      << "fn block_score(row: u32, block: u32) -> f32 {\n"
      << "  var key_square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = pooled_value(row, block, k);\n"
      << "    key_square_sum += value * value;\n"
      << "  }\n"
      << "  let key_inv_rms = inverseSqrt(key_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "  let position = clamp_position(visible_at(row, block * uniforms.compress_ratio));\n"
      << "  var score = 0.0;\n"
      << "  for (var head = 0u; head < uniforms.num_heads; head++) {\n"
      << "    let query_base = row * uniforms.query_row_stride + head * uniforms.head_size;\n"
      << "    var query_square_sum = 0.0;\n"
      << "    for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "      let value = f32(" << query.GetByOffset("query_base + k") << ");\n"
      << "      query_square_sum += value * value;\n"
      << "    }\n"
      << "    let query_inv_rms = inverseSqrt(query_square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "    var dot = 0.0;\n"
      << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "      dot += query_value(row, head, d, query_inv_rms) * key_value(row, block, d, key_inv_rms, position);\n"
      << "    }\n"
      << "    score += max(dot, 0.0);\n"
      << "  }\n"
      << "  return score * uniforms.scale;\n"
      << "}\n";

  if (use_global_cache_) {
    const std::string score_offset =
        "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
    const std::string count_offset =
        "row * (uniforms.total_sequence_length + 1u) + uniforms.total_sequence_length";
    if (phase_ == QsaCachePhase::Score) {
      shader.AdditionalImplementation()
          << "var<workgroup> pooled_key: array<f32, " << kQsaSharedHeadSize << ">;\n"
          << "var<workgroup> head_scores: array<f32, 64>;\n"
          << "var<workgroup> key_inverse_rms: f32;\n";
      shader.MainFunctionBody()
          << "  let stride = max(1u, uniforms.total_sequence_length / uniforms.compress_ratio);\n"
          << "  let row = workgroup_idx / stride;\n"
          << "  let candidate = workgroup_idx % stride;\n"
          << "  if (row >= uniforms.rows) { return; }\n"
          << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
          << "  workgroupBarrier();\n"
          << "  let is_valid_block = candidate < visible_count_shared / uniforms.compress_ratio;\n"
          << "  for (var d = local_idx; d < uniforms.head_size; d += 64u) {\n"
          << "    pooled_key[d] = 0.0;\n"
          << "    if (is_valid_block) { pooled_key[d] = pooled_value(row, candidate, d); }\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_idx == 0u) {\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) { square_sum += pooled_key[d] * pooled_key[d]; }\n"
          << "    key_inverse_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var d = local_idx; d < uniforms.head_size; d += 64u) {\n"
          << "    pooled_key[d] = pooled_key[d] * key_inverse_rms * f32(" << key_norm.GetByOffset("d") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  var position = 0u;\n"
          << "  if (is_valid_block) { position = clamp_position(visible_at(row, candidate * uniforms.compress_ratio)); }\n"
          << "  let half = uniforms.rotary_width / 2u;\n"
          << "  let cache_base = ((row / uniforms.sequence_length) * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width;\n"
          << "  for (var d = local_idx; d < half; d += 64u) {\n"
          << "    let left = pooled_key[d];\n"
          << "    let right = pooled_key[d + half];\n"
          << "    pooled_key[d] = left * f32(" << cos_cache.GetByOffset("cache_base + d") << ") + "
                                                                                             "(-1.0 * right) * f32("
          << sin_cache.GetByOffset("cache_base + d") << ");\n"
          << "    pooled_key[d + half] = right * f32(" << cos_cache.GetByOffset("cache_base + d + half") << ") + "
                                                                                                            "left * f32("
          << sin_cache.GetByOffset("cache_base + d + half") << ");\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_idx < uniforms.num_heads && is_valid_block) {\n"
          << "    let query_base = row * uniforms.query_row_stride + local_idx * uniforms.head_size;\n"
          << "    var square_sum = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      let value = f32(" << query.GetByOffset("query_base + d") << ");\n"
          << "      square_sum += value * value;\n"
          << "    }\n"
          << "    let inv_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
          << "    var dot = 0.0;\n"
          << "    for (var d = 0u; d < uniforms.head_size; d++) {\n"
          << "      dot += query_value(row, local_idx, d, inv_rms) * pooled_key[d];\n"
          << "    }\n"
          << "    head_scores[local_idx] = max(dot, 0.0);\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  if (local_idx == 0u && is_valid_block) {\n"
          << "    var score = 0.0;\n"
          << "    for (var head = 0u; head < uniforms.num_heads; head++) { score += head_scores[head]; }\n"
          << "    " << score_cache->SetByOffset(score_offset, "score * uniforms.scale") << "\n"
          << "  }\n";
      return Status::OK();
    }
    if (phase_ == QsaCachePhase::Map) {
      shader.AdditionalImplementation() << "var<workgroup> visible_flags: array<u32, 64>;\n";
    } else if (sort_capacity_ > 0) {
      shader.AdditionalImplementation()
          << "var<workgroup> sort_scores: array<f32, " << sort_capacity_ << ">;\n"
          << "var<workgroup> sort_indices: array<i32, " << sort_capacity_ << ">;\n"
          << "fn qsa_before(left_score: f32, left_index: i32, right_score: f32, right_index: i32) -> bool {\n"
          << "  if (left_index < 0) { return false; }\n"
          << "  if (right_index < 0) { return true; }\n"
          << "  if (left_score != left_score) { return true; }\n"
          << "  if (right_score != right_score) { return false; }\n"
          << "  return left_score > right_score || (left_score == right_score && left_index < right_index);\n"
          << "}\n";
    } else {
      shader.AdditionalImplementation()
          << "var<workgroup> best_scores: array<f32, 64>;\n"
          << "var<workgroup> best_indices: array<i32, 64>;\n"
          << "var<workgroup> previous_score_shared: f32;\n"
          << "var<workgroup> previous_index_shared: i32;\n";
    }
    shader.MainFunctionBody()
        << "  let row = workgroup_idx;\n"
        << "  if (row >= uniforms.rows) { return; }\n";
    if (phase_ == QsaCachePhase::Map) {
      if (dense_map_) {
        shader.MainFunctionBody()
            << "  var lane_count = 0u;\n"
            << "  for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "    lane_count += select(0u, 1u, visible(row, token));\n"
            << "  }\n"
            << "  visible_flags[local_idx] = lane_count;\n"
            << "  workgroupBarrier();\n"
            << "  for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
            << "    if (local_idx < stride) { visible_flags[local_idx] += visible_flags[local_idx + stride]; }\n"
            << "    workgroupBarrier();\n"
            << "  }\n"
            << "  if (local_idx == 0u) { visible_count_shared = visible_flags[0]; }\n"
            << "  let dense_count = workgroupUniformLoad(&visible_count_shared);\n"
            << "  if (dense_count == 0u || dense_count == uniforms.total_sequence_length) {\n"
            << "    if (dense_count != 0u) {\n"
            << "      for (var token = local_idx; token < uniforms.total_sequence_length; token += 64u) {\n"
            << "        " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + token", "token") << "\n"
            << "      }\n"
            << "    }\n"
            << "    storageBarrier();\n"
            << "    if (local_idx == 0u) { "
            << visible_cache->SetByOffset(count_offset, "dense_count") << " }\n"
            << "    return;\n"
            << "  }\n";
      }
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) { visible_count_shared = 0u; }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length; base += 64u) {\n"
          << "    let token = base + local_idx;\n"
          << "    visible_flags[local_idx] = select(0u, 1u, "
             "token < uniforms.total_sequence_length && visible(row, token));\n"
          << "    workgroupBarrier();\n"
          << "    var prefix = 0u;\n"
          << "    for (var lane = 0u; lane < local_idx; lane++) { prefix += visible_flags[lane]; }\n"
          << "    let chunk_base = visible_count_shared;\n"
          << "    if (visible_flags[local_idx] != 0u) {\n"
          << "      " << visible_cache->SetByOffset("row * (uniforms.total_sequence_length + 1u) + chunk_base + prefix", "token") << "\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    if (local_idx == 63u) {\n"
          << "      visible_count_shared = chunk_base + prefix + visible_flags[local_idx];\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n"
          << "  storageBarrier();\n"
          << "  if (local_idx == 0u) { "
          << visible_cache->SetByOffset(count_offset, "visible_count_shared") << " }\n";
      return Status::OK();
    }
    shader.MainFunctionBody()
        << "  if (local_idx == 0u) { visible_count_shared = " << visible_cache->GetByOffset(count_offset) << "; }\n"
        << "  workgroupBarrier();\n"
        << "  let visible_count = visible_count_shared;\n"
        << "  let block_count = visible_count / uniforms.compress_ratio;\n"
        << "  let output_base = row * uniforms.capacity;\n"
        << "  for (var i = local_idx; i < uniforms.capacity; i += 64u) {\n"
        << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
        << "  }\n";
    if (sort_capacity_ == 0) {
      shader.MainFunctionBody()
          << "  if (local_idx == 0u) {\n"
          << "    previous_score_shared = 0.0;\n"
          << "    previous_index_shared = -1i;\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  workgroupBarrier();\n"
        << "  storageBarrier();\n"
        << "  let selected_blocks = min(uniforms.block_topk, block_count);\n";
    if (sort_capacity_ > 0) {
      const uint32_t half_capacity = sort_capacity_ / 2;
      shader.MainFunctionBody()
          << "  for (var item = local_idx; item < " << sort_capacity_ << "u; item += 64u) {\n"
          << "    sort_scores[item] = 0.0;\n"
          << "    sort_indices[item] = -1i;\n"
          << "  }\n"
          << "  workgroupBarrier();\n"
          << "  for (var base = 0u; base < uniforms.total_sequence_length / uniforms.compress_ratio; base += "
          << half_capacity << "u) {\n"
          << "    for (var item = local_idx; item < " << half_capacity << "u; item += 64u) {\n"
          << "      let candidate = base + item;\n"
          << "      var score = 0.0;\n"
          << "      var index = -1i;\n"
          << "      if (candidate < block_count) {\n"
          << "        score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "        if (score == score || candidate == 0u) { index = i32(candidate); }\n"
          << "      }\n"
          << "      sort_scores[" << half_capacity << "u + item] = score;\n"
          << "      sort_indices[" << half_capacity << "u + item] = index;\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    for (var stage_size = 2u; stage_size <= " << sort_capacity_ << "u; stage_size *= 2u) {\n"
          << "      for (var distance = stage_size / 2u; distance > 0u; distance /= 2u) {\n"
          << "        for (var pair = local_idx; pair < " << half_capacity << "u; pair += 64u) {\n"
          << "          let left = (pair / distance) * (2u * distance) + pair % distance;\n"
          << "          let right = left + distance;\n"
          << "          let left_score = sort_scores[left];\n"
          << "          let right_score = sort_scores[right];\n"
          << "          let left_index = sort_indices[left];\n"
          << "          let right_index = sort_indices[right];\n"
          << "          let forward = (left & stage_size) == 0u;\n"
          << "          if ((forward && qsa_before(right_score, right_index, left_score, left_index)) ||\n"
          << "              (!forward && qsa_before(left_score, left_index, right_score, right_index))) {\n"
          << "            sort_scores[left] = right_score;\n"
          << "            sort_scores[right] = left_score;\n"
          << "            sort_indices[left] = right_index;\n"
          << "            sort_indices[right] = left_index;\n"
          << "          }\n"
          << "        }\n"
          << "        workgroupBarrier();\n"
          << "      }\n"
          << "    }\n"
          << "  }\n"
          << "  var first_is_nan = false;\n"
          << "  if (block_count > 0u) {\n"
          << "    let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "    first_is_nan = first_score != first_score;\n"
          << "  }\n"
          << "  for (var rank = local_idx; rank < selected_blocks; rank += 64u) {\n"
          << "    let chosen = sort_indices[rank];\n"
          << "    if (chosen >= 0 && (!first_is_nan || rank == 0u)) {\n"
          << "      for (var slot = 0u; slot < uniforms.compress_ratio; slot++) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "    }\n"
          << "  }\n";
    } else {
      shader.MainFunctionBody()
          << "  for (var rank = 0u; rank < uniforms.block_topk; rank++) {\n"
          << "    var best_score = 0.0;\n"
          << "    var best_index = -1i;\n"
          << "    for (var candidate = local_idx; candidate < block_count; candidate += 64u) {\n"
          << "      let score = " << score_cache->GetByOffset(score_offset) << ";\n"
          << "      if (score != score) { continue; }\n"
          << "      if (previous_index_shared >= 0 && !(score < previous_score_shared || "
             "(score == previous_score_shared && i32(candidate) > previous_index_shared))) { continue; }\n"
          << "      if (best_index < 0 || score > best_score || "
             "(score == best_score && i32(candidate) < best_index)) {\n"
          << "        best_score = score;\n"
          << "        best_index = i32(candidate);\n"
          << "      }\n"
          << "    }\n"
          << "    best_scores[local_idx] = best_score;\n"
          << "    best_indices[local_idx] = best_index;\n"
          << "    workgroupBarrier();\n"
          << "    for (var stride = 32u; stride > 0u; stride /= 2u) {\n"
          << "      if (local_idx < stride) {\n"
          << "        let other = local_idx + stride;\n"
          << "        if (best_indices[other] >= 0 && (best_indices[local_idx] < 0 || "
             "best_scores[other] > best_scores[local_idx] || "
             "(best_scores[other] == best_scores[local_idx] && best_indices[other] < best_indices[local_idx]))) {\n"
          << "          best_scores[local_idx] = best_scores[other];\n"
          << "          best_indices[local_idx] = best_indices[other];\n"
          << "        }\n"
          << "      }\n"
          << "      workgroupBarrier();\n"
          << "    }\n"
          << "    if (local_idx == 0u && rank == 0u && block_count > 0u) {\n"
          << "      let first_score = " << score_cache->GetByOffset("row * (uniforms.total_sequence_length / uniforms.compress_ratio)") << ";\n"
          << "      if (first_score != first_score) { best_scores[0] = first_score; best_indices[0] = 0i; }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "    let chosen = best_indices[0];\n"
          << "    if (chosen >= 0) {\n"
          << "      for (var slot = local_idx; slot < uniforms.compress_ratio; slot += 64u) {\n"
          << "        let token = visible_at(row, u32(chosen) * uniforms.compress_ratio + slot);\n"
          << "        " << selected.SetByOffset("output_base + rank * uniforms.compress_ratio + slot", "i32(token)") << "\n"
          << "      }\n"
          << "      if (local_idx == 0u) {\n"
          << "        previous_score_shared = best_scores[0];\n"
          << "        previous_index_shared = chosen;\n"
          << "      }\n"
          << "    }\n"
          << "    workgroupBarrier();\n"
          << "  }\n";
    }
    shader.MainFunctionBody()
        << "  let tail_start = block_count * uniforms.compress_ratio;\n"
        << "  for (var tail = local_idx; tail < visible_count - tail_start; tail += 64u) {\n"
        << "    let output = selected_blocks * uniforms.compress_ratio + tail;\n"
        << "    " << selected.SetByOffset("output_base + output", "i32(visible_at(row, tail_start + tail))") << "\n"
        << "  }\n";
    return Status::OK();
  }

  const std::string cache_condition =
      use_global_cache_ ? "true" : "block_count <= " + std::to_string(kQsaCachedBlocks) + "u";
  const std::string visible_store = use_global_cache_
                                        ? visible_cache->SetByOffset("row * uniforms.total_sequence_length + count", "token")
                                        : "if (count < " + std::to_string(kQsaCachedVisibleTokens) + "u) { visible_tokens[count] = token; }";
  const std::string score_offset =
      "row * (uniforms.total_sequence_length / uniforms.compress_ratio) + candidate";
  const std::string score_store = use_global_cache_
                                      ? score_cache->SetByOffset(score_offset, "block_score(row, candidate)")
                                      : "block_scores[candidate] = block_score(row, candidate);";
  const std::string score_load =
      use_global_cache_ ? score_cache->GetByOffset(score_offset) : "block_scores[candidate]";
  const std::string storage_barrier = use_global_cache_ ? "  storageBarrier();\n" : "";
  shader.MainFunctionBody()
      << "  let row = workgroup_idx;\n"
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  if (local_idx == 0u) {\n"
      << "    var count = 0u;\n"
      << "    for (var token = 0u; token < uniforms.total_sequence_length; token++) {\n"
      << "      if (visible(row, token)) {\n"
      << "        " << visible_store << "\n"
      << "        count++;\n"
      << "      }\n"
      << "    }\n"
      << "    visible_count_shared = count;\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  let visible_count = visible_count_shared;\n"
      << "  let block_count = visible_count / uniforms.compress_ratio;\n"
      << "  if (" << cache_condition << ") {\n"
      << "    for (var candidate = local_idx; candidate < block_count; candidate += "
      << kQsaSelectWorkgroupSize << "u) {\n"
      << "      " << score_store << "\n"
      << "    }\n"
      << "  }\n"
      << "  workgroupBarrier();\n"
      << storage_barrier
      << "  if (local_idx != 0u) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let selected_blocks = min(uniforms.block_topk, block_count);\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < selected_blocks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < block_count; candidate++) {\n"
      << "      var score = 0.0;\n"
      << "      if (" << cache_condition << ") {\n"
      << "        score = " << score_load << ";\n"
      << "      } else {\n"
      << "        score = block_score(row, candidate);\n"
      << "      }\n"
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
  const auto& scoring_keys = shader.AddOutput("scoring_compressed_keys",
                                              ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  shader.MainFunctionBody()
      << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.total")
      << "  let d = global_idx % uniforms.head_size;\n"
      << "  let entry_row = global_idx / uniforms.head_size;\n"
      << "  let entry = entry_row % uniforms.past_length;\n"
      << "  let batch = entry_row / uniforms.past_length;\n"
      << "  let output = (batch * uniforms.present_length + entry) * uniforms.head_size + d;\n"
      << "  " << present.SetByOffset("output", "present_compressed_key_element_t(" + past.GetByOffset("global_idx") + ")")
      << "\n"
      << "  " << scoring_keys.SetByOffset("output", "scoring_compressed_keys_element_t(" + past.GetByOffset("global_idx") + ")")
      << "\n";
  return Status::OK();
}

Status SparseAttentionIndexerCsaCompressProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& key = shader.AddInput("key", ShaderUsage::UseUniform);
  const auto& gate = shader.AddInput("gate", ShaderUsage::UseUniform);
  const ShaderVariableHelper* past_proj = nullptr;
  if (has_past_buffer_) {
    past_proj = &shader.AddInput("past_proj_buffer", ShaderUsage::UseUniform);
  }
  const auto& bias = shader.AddInput("position_bias", ShaderUsage::UseUniform);
  const auto& norm = shader.AddInput("key_norm_weight", ShaderUsage::UseUniform);
  const auto& cos_cache = shader.AddInput("cos_cache", ShaderUsage::UseUniform);
  const auto& sin_cache = shader.AddInput("sin_cache", ShaderUsage::UseUniform);
  const auto& present = shader.AddOutput("present_compressed_key",
                                         ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);
  const auto& scoring_keys = shader.AddOutput("scoring_compressed_keys",
                                              ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  shader.AdditionalImplementation()
      << kWgslNegativeMax
      << "fn kv_value(batch: u32, position: u32, channel: u32) -> f32 {\n";
  if (has_past_buffer_) {
    shader.AdditionalImplementation()
        << "  if (position < uniforms.past_buffer_length) {\n"
        << "    return f32(" << past_proj->GetByOffset(
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
        << "    return f32(" << past_proj->GetByOffset(
                                    "uniforms.past_buffer_length * uniforms.head_size * 2u * "
                                    "(uniforms.work_items / uniforms.new_window_count) + "
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
      << "  var max_gate = NEGATIVE_MAX_F32;\n"
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
      << "  if (work >= uniforms.work_items) { return; }\n"
      << "  let window = work % uniforms.new_window_count;\n"
      << "  let batch = work / uniforms.new_window_count;\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var d = 0u; d < uniforms.head_size; d++) {\n"
      << "    let value = pooled_value(batch, window, d);\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  let inverse_rms = inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon);\n"
      << "  let entry = uniforms.past_compressed_length + window;\n"
      << "  let max_position = uniforms.max_rotary_length - 1u;\n"
      << "  let position = select(entry * uniforms.compress_ratio, max_position, "
         "entry > max_position / uniforms.compress_ratio);\n"
      << "  let cache_base = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width;\n"
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
      << "    " << scoring_keys.SetByOffset("output", "scoring_compressed_keys_element_t(value)") << "\n"
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
  const ShaderVariableHelper* past_proj = nullptr;
  if (has_past_buffer_) {
    past_proj = &shader.AddInput("past_proj_buffer", ShaderUsage::UseUniform);
  }
  const auto& present_proj = shader.AddOutput("present_proj_buffer",
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
        << "    " << present_proj.SetByOffset("global_idx", "present_proj_buffer_element_t(" + past_proj->GetByOffset("input") + ")")
        << "\n"
        << "    " << present_proj.SetByOffset("uniforms.total + global_idx", "present_proj_buffer_element_t(" + past_proj->GetByOffset("(uniforms.total / uniforms.present_buffer_length) * "
                                                                                                                                       "uniforms.past_buffer_length + input") +
                                                                                 ")")
        << "\n"
        << "    return;\n"
        << "  }\n";
  }
  if (has_current_) {
    shader.MainFunctionBody()
        << "  let current_token = source - uniforms.past_buffer_length;\n"
        << "  let input = (batch * uniforms.sequence_length + current_token) * width + channel;\n"
        << "  "
        << present_proj.SetByOffset("global_idx",
                                    "present_proj_buffer_element_t(" + key->GetByOffset("input") + ")")
        << "\n"
        << "  "
        << present_proj.SetByOffset("uniforms.total + global_idx",
                                    "present_proj_buffer_element_t(" + gate->GetByOffset("input") + ")")
        << "\n";
  }
  return Status::OK();
}

Status SparseAttentionIndexerCsaSelectProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& query = shader.AddInput("query", ShaderUsage::UseUniform);
  const auto& query_norm = shader.AddInput("query_norm_weight", ShaderUsage::UseUniform);
  const auto& compressed_key = shader.AddInput("scoring_compressed_keys", ShaderUsage::UseUniform);
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
      << "  var raw = " << position_ids.GetByOffset("row", true) << ";\n"
      << "  if ((raw.y & 0x80000000u) != 0u) { return 0u; }\n"
      << "  raw.x += 1u;\n"
      << "  raw.y += select(0u, 1u, raw.x == 0u);\n"
      << "  if (raw.y / uniforms.compress_ratio != 0u) { return count; }\n"
      << "  var remainder = raw.y;\n"
      << "  var quotient = 0u;\n"
      << "  for (var bit_index = 32u; bit_index > 0u; bit_index--) {\n"
      << "    let bit = bit_index - 1u;\n"
      << "    let carry = remainder >> 31u;\n"
      << "    let doubled = (remainder << 1u) | ((raw.x >> bit) & 1u);\n"
      << "    if (carry != 0u || doubled >= uniforms.compress_ratio) {\n"
      << "      remainder = doubled - uniforms.compress_ratio;\n"
      << "      quotient |= 1u << bit;\n"
      << "    } else {\n"
      << "      remainder = doubled;\n"
      << "    }\n"
      << "  }\n"
      << "  return min(count, quotient);\n"
      << "}\n"
      << "fn normalized_query_value(row: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = (row * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var square_sum = 0.0;\n"
      << "  for (var k = 0u; k < uniforms.head_size; k++) {\n"
      << "    let value = f32(" << query.GetByOffset("base + k") << ");\n"
      << "    square_sum += value * value;\n"
      << "  }\n"
      << "  return f32(" << query.GetByOffset("base + d")
      << ") * inverseSqrt(square_sum / f32(uniforms.head_size) + uniforms.epsilon) * f32("
      << query_norm.GetByOffset("d") << ");\n"
      << "}\n"
      << "fn query_value(row: u32, head: u32, d: u32) -> f32 {\n"
      << "  let base = (row * uniforms.num_heads + head) * uniforms.head_size;\n"
      << "  var value = normalized_query_value(row, head, d);\n"
      << "  let rotary_base = uniforms.head_size - 2u * uniforms.rotary_width;\n"
      << "  if (d >= rotary_base) {\n"
      << "    let offset = d - rotary_base;\n"
      << "    let pair_d = select(d - 1u, d + 1u, (offset & 1u) == 0u);\n"
      << "    let sign = select(1.0, -1.0, (offset & 1u) == 0u);\n"
      << "    let paired = sign * normalized_query_value(row, head, pair_d);\n"
      << "    let batch = row / uniforms.sequence_length;\n"
      << "    let position = clamped_position(row, uniforms.max_rotary_length - 1u);\n"
      << "    let cache = (batch * uniforms.rotary_cache_batch_stride + position) * uniforms.rotary_width + offset / 2u;\n"
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
      << "  if (row >= uniforms.rows) { return; }\n"
      << "  let output_base = row * uniforms.capacity;\n"
      << "  for (var i = 0u; i < uniforms.capacity; i++) {\n"
      << "    " << selected.SetByOffset("output_base + i", "-1") << "\n"
      << "  }\n"
      << "  let count = uniforms.present_compressed_length;\n"
      << "  let threshold = visible_entry_count(row, count);\n"
      << "  let ranks = min(uniforms.capacity, threshold);\n"
      << "  if (threshold == 0u || ranks == 0u) { return; }\n"
      << "  var previous_score = 0.0;\n"
      << "  var previous_index = -1i;\n"
      << "  for (var rank = 0u; rank < ranks; rank++) {\n"
      << "    var best_score = 0.0;\n"
      << "    var best_index = -1i;\n"
      << "    for (var candidate = 0u; candidate < threshold; candidate++) {\n"
      << "      let score = entry_score(row, candidate);\n"
      << "      if (previous_index >= 0 && !(score < previous_score || "
         "(score == previous_score && i32(candidate) > previous_index))) { continue; }\n"
      << "      if (best_index < 0 || score > best_score || (score == best_score && i32(candidate) < best_index)) {\n"
      << "        best_score = score;\n"
      << "        best_index = i32(candidate);\n"
      << "      }\n"
      << "    }\n"
      << "    if (best_index < 0) { break; }\n"
      << "    " << selected.SetByOffset("output_base + rank", "best_index") << "\n"
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

Status SparseAttentionIndexer::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const bool is_qsa = policy_ == sai::Policy::kQsa;
  ORT_RETURN_IF(!is_qsa && context.Input(sai::kKey) == nullptr,
                "SparseAttentionIndexer: key is required for policy_mode 'csa'");
  for (int index : {sai::kGate, sai::kPositionBias, sai::kHeadWeights,
                    sai::kPositionIds, sai::kPastProjBuffer}) {
    const bool policy_owns_slot = !is_qsa;
    const bool provided = index < context.InputCount() && context.Input(index) != nullptr;
    ORT_RETURN_IF(provided != policy_owns_slot, "SparseAttentionIndexer: input ", index,
                  provided ? " must be omitted for policy_mode '" : " is required for policy_mode '",
                  is_qsa ? sai::kPolicyModeQsa : sai::kPolicyModeCsa, "'");
  }
  ORT_RETURN_IF(!is_qsa && context.Input(sai::kMask) != nullptr,
                "SparseAttentionIndexer: mask must be omitted for policy_mode 'csa'");
  return is_qsa ? ComputeQsa(context) : ComputeCsa(context);
}

#include "qsa_parallel_map.inc"

Status SparseAttentionIndexer::ComputeQsa(onnxruntime::webgpu::ComputeContext& context) const {
  const int prefill_score16 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_PREFILL_SCORE16", 0);
  ORT_RETURN_IF_NOT(prefill_score16 == 0 || prefill_score16 == 1, "ORT_WEBGPU_QSA_PREFILL_SCORE16 must be 0 or 1");

  const Tensor* query = context.Input(sai::kQuery);
  const Tensor* key = context.Input(sai::kKey);
  const Tensor* query_norm = context.Input(sai::kQueryNormWeight);
  const Tensor* key_norm = context.Input(sai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(sai::kCosCache);
  const Tensor* sin_cache = context.Input(sai::kSinCache);
  const Tensor* mask = context.Input(sai::kMask);
  const Tensor* past_key = context.Input(sai::kPastKey);
  ORT_RETURN_IF(context.InputCount() > sai::kPastSequenceLength &&
                    context.Input(sai::kPastSequenceLength) != nullptr,
                "SparseAttentionIndexer WebGPU does not support fixed-capacity caches");

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "SparseAttentionIndexer: query must have rank 3");
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const auto& query_norm_shape = query_norm->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1 && query_norm_shape[0] > 0,
                    "SparseAttentionIndexer: invalid query_norm_weight shape");
  const int64_t head_size = query_norm_shape[0];
  ORT_RETURN_IF_NOT(query_shape[2] > 0 && query_shape[2] % head_size == 0,
                    "SparseAttentionIndexer: query width must be positive and divisible by head_size");
  const bool packed_qk = key == nullptr;
  const int64_t packed_head_count = query_shape[2] / head_size;
  ORT_RETURN_IF(packed_qk && packed_head_count < 2,
                "SparseAttentionIndexer: packed QK input must contain at least one query head and one key");
  const int64_t num_heads = packed_head_count - (packed_qk ? 1 : 0);
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "SparseAttentionIndexer: invalid query dimensions");
  const auto& past_shape = past_key->Shape();
  ORT_RETURN_IF_NOT(past_shape.NumDimensions() == 3 && past_shape[0] == batch_size && past_shape[2] == head_size,
                    "SparseAttentionIndexer: invalid past_key shape");
  const int64_t past_length = past_shape[1];
  ORT_RETURN_IF_ERROR(CheckUint32({{batch_size, "batch_size"},
                                   {sequence_length, "sequence_length"},
                                   {num_heads, "num_heads"},
                                   {head_size, "head_size"},
                                   {query_shape[2], "query width"},
                                   {past_length, "past sequence length"},
                                   {compress_ratio_, "compress_ratio"},
                                   {token_budget_, "token_budget"}}));
  ORT_RETURN_IF(past_length > std::numeric_limits<uint32_t>::max() - sequence_length,
                "SparseAttentionIndexer WebGPU: total sequence length must fit in uint32");
  const int64_t total_length = past_length + sequence_length;
  if (!packed_qk) {
    ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, head_size}));
  }
  ORT_RETURN_IF_ERROR(CheckShape(query_norm, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm, "key_norm_weight", {head_size}));
  RotaryCacheShape rotary_cache_shape;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary_cache_shape));
  const int64_t max_rotary_length = rotary_cache_shape.max_rotary_length;
  const int64_t rotary_width = rotary_cache_shape.rotary_width;
  ORT_RETURN_IF_NOT(rotary_width > 0 && rotary_width % 2 == 0 && rotary_width <= head_size,
                    "SparseAttentionIndexer: invalid qsa rotary cache shape");
  if (mask != nullptr) {
    const auto& mask_shape = mask->Shape();
    ORT_RETURN_IF_NOT(
        mask_shape.NumDimensions() == 2 && mask_shape[0] == batch_size && mask_shape[1] == total_length,
        "SparseAttentionIndexer: qsa mask must be INT64 with shape (batch_size, total_sequence_length)");
  }

  const int64_t capacity = sai::SelectedCapacity(policy_, token_budget_, index_topk_, compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckUint32({{max_rotary_length, "rotary cache length"},
                                   {rotary_width, "rotary width"},
                                   {total_length, "total sequence length"},
                                   {capacity, "selected capacity"}}));
  ORT_RETURN_IF(total_length > std::numeric_limits<int32_t>::max(),
                "SparseAttentionIndexer WebGPU: qsa selected-index range must fit in int32");
  ORT_RETURN_IF_ERROR(CheckUint32Product("query element count", {batch_size, sequence_length, query_shape[2]}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("present key element count", {batch_size, total_length, head_size}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("selected index element count", {batch_size, sequence_length, capacity}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("rotary cache element count",
                                         {rotary_cache_shape.batched ? batch_size : 1,
                                          max_rotary_length, rotary_width}));
  if (mask != nullptr) {
    ORT_RETURN_IF_ERROR(CheckUint32Product("mask element count", {batch_size, total_length}));
  }
  Tensor* selected =
      context.Output(sai::kSelectedIndices, TensorShape({batch_size, sequence_length, capacity}));
  Tensor* present = context.Output(sai::kPresentKey, TensorShape({batch_size, total_length, head_size}));
  const int64_t present_elements = present->Shape().Size();
  const int score16 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_SCORE16", 0);
  ORT_RETURN_IF_NOT(score16 == 0 || score16 == 1, "ORT_WEBGPU_QSA_SCORE16 must be 0 or 1");
  const int indexer_stack = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_INDEXER_STACK", 0);
  ORT_RETURN_IF_NOT(indexer_stack == 0 || indexer_stack == 1, "ORT_WEBGPU_QSA_INDEXER_STACK must be 0 or 1");
  const int concat_vector = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_CONCAT_VECTOR", 0);
  ORT_RETURN_IF_NOT(concat_vector == 0 || concat_vector == 1, "ORT_WEBGPU_QSA_CONCAT_VECTOR must be 0 or 1");
  if ((concat_vector == 1 || indexer_stack == 1) && head_size % 4 == 0 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"}) {
  if (present_elements > 0) {
    const bool has_past = past_length > 0;
    const bool has_current = sequence_length > 0;
    const Tensor* current_key = packed_qk ? query : key;
    QsaVectorConcatProgram concat{has_past, has_current};
    concat.CacheHint(has_past, has_current)
        .SetWorkgroupSize(kElementwiseWorkgroupSize);
    if (has_past) {
      concat.AddInput({past_key, ProgramTensorMetadataDependency::Type, {past_key->Shape().Size() / 4}, 4});
    }
    if (has_current) {
      concat.AddInput({current_key, ProgramTensorMetadataDependency::Type, {current_key->Shape().Size() / 4}, 4});
    }
    concat.AddOutput({present, ProgramTensorMetadataDependency::Type, {present_elements / 4}, 4})
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(present_elements / 4), kElementwiseWorkgroupSize))
        .AddUniformVariables({{ToUint32(present_elements / 4)},
                              {ToUint32(sequence_length)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(head_size / 4)},
                              {ToUint32((packed_qk ? query_shape[2] : head_size) / 4)},
                              {ToUint32((packed_qk ? num_heads * head_size : 0) / 4)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(concat));
  }
  } else {
  if (present_elements > 0) {
    const bool has_past = past_length > 0;
    const bool has_current = sequence_length > 0;
    const Tensor* current_key = packed_qk ? query : key;
    SparseAttentionIndexerQsaConcatProgram concat{has_past, has_current};
    concat.CacheHint(has_past, has_current)
        .SetWorkgroupSize(kElementwiseWorkgroupSize);
    if (has_past) {
      concat.AddInput({past_key, ProgramTensorMetadataDependency::Type});
    }
    if (has_current) {
      concat.AddInput({current_key, ProgramTensorMetadataDependency::Type});
    }
    concat.AddOutput({present, ProgramTensorMetadataDependency::Type})
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(present_elements), kElementwiseWorkgroupSize))
        .AddUniformVariables({{ToUint32(present_elements)},
                              {ToUint32(sequence_length)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(head_size)},
                              {ToUint32(packed_qk ? query_shape[2] : head_size)},
                              {ToUint32(packed_qk ? num_heads * head_size : 0)}});
    ORT_RETURN_IF_ERROR(context.RunProgram(concat));
  }
  }
  const int64_t rows = batch_size * sequence_length;
  if (rows == 0) {
    return Status::OK();
  }
  // Each visible-map row reserves its final uint32 for the compacted token count.
  const uint64_t visible_elements = static_cast<uint64_t>(rows) * (total_length + 1);
  const int64_t score_stride = std::max<int64_t>(1, total_length / compress_ratio_);
  const uint64_t score_elements = static_cast<uint64_t>(rows) * score_stride;
  const uint64_t max_binding_size = context.DeviceLimits().maxStorageBufferBindingSize;
  const bool use_global_cache =
      (total_length > kQsaCachedVisibleTokens || total_length / compress_ratio_ > kQsaCachedBlocks ||
       (num_heads >= 2 && total_length >= compress_ratio_)) &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_GLOBAL_SCORE_CACHE", 0) != 0 &&
      head_size <= kQsaSharedHeadSize && num_heads <= kQsaSelectWorkgroupSize &&
      context.DeviceLimits().maxStorageBuffersPerShaderStage >= (mask != nullptr ? 10u : 9u) &&
      visible_elements <= std::numeric_limits<uint32_t>::max() &&
      score_elements <= std::numeric_limits<uint32_t>::max() &&
      visible_elements * sizeof(uint32_t) <= max_binding_size &&
      score_elements * sizeof(float) <= max_binding_size;
  Tensor visible_cache, score_cache;
  if (use_global_cache) {
    visible_cache = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{rows, total_length + 1});
    score_cache = context.CreateGPUTensor(DataTypeImpl::GetType<float>(), TensorShape{rows, score_stride});
  }
  uint32_t sort_capacity = 0;
  const int64_t block_topk = token_budget_ / compress_ratio_;
  if (use_global_cache && block_topk > 0 && block_topk <= 1024) {
    const uint32_t capacity = 2u * std::bit_ceil(ToUint32(block_topk));
    const uint32_t stages = std::bit_width(capacity) - 1u;
    const uint32_t sort_passes = stages * (stages + 1u) / 2u;
    const uint32_t sort_barriers = 1u + DivideRoundUp(ToUint32(score_stride), capacity / 2u) * (sort_passes + 1u);
    // Rank reduction has nine workgroup barriers per requested rank.
    if (sort_barriers < 9u * block_topk &&
        capacity * (sizeof(float) + sizeof(int32_t)) + sizeof(uint32_t) <=
        context.DeviceLimits().maxComputeWorkgroupStorageSize) {
      sort_capacity = capacity;
    }
  }
  const int packed_score = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_PACKED_SCORE", 0);
  ORT_RETURN_IF_NOT(packed_score == 0 || packed_score == 1, "ORT_WEBGPU_QSA_PACKED_SCORE must be 0 or 1");
  // Bound all-pairs ranking while exposing each candidate to an independent workgroup.
  const bool use_parallel_rank =
      use_global_cache && rows == 1 && block_topk > 0 && block_topk <= 1024 &&
      score_stride >= 512 && score_stride <= 16384 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_PARALLEL_RANK", 0) != 0;
  const bool use_dense_map =
      use_global_cache && rows == 1 && total_length >= 512 && total_length <= 65536 &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_DENSE_MAP", 0) != 0;
  const int parallel_map = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QSA_PARALLEL_MAP", 0);
  ORT_RETURN_IF_NOT(parallel_map == 0 || parallel_map == 1, "ORT_WEBGPU_QSA_PARALLEL_MAP must be 0 or 1");
  const bool use_parallel_map =
      parallel_map == 1 && use_dense_map && sequence_length == 1 && total_length >= 8192;
  if (use_parallel_map) {
    const uint32_t chunks = DivideRoundUp(ToUint32(total_length), 512u);
    auto counts = context.CreateGPUTensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{chunks});
    for (bool count_only : {true, false}) {
      QsaParallelMapProgram map{mask != nullptr, count_only};
      map.CacheHint(mask != nullptr);
      if (mask != nullptr) {
        map.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
      }
      if (!count_only) map.AddInput({&counts, ProgramTensorMetadataDependency::Type});
      map.AddOutput({count_only ? &counts : &visible_cache, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{ToUint32(total_length)}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(chunks);
      ORT_RETURN_IF_ERROR(context.RunProgram(map));
    }
  }
  for (auto phase : {QsaCachePhase::Map, QsaCachePhase::Score, QsaCachePhase::Select}) {
    if (use_parallel_map && phase == QsaCachePhase::Map) continue;
    if (!use_global_cache && phase != QsaCachePhase::Select) {
      continue;
    }
    if (prefill_score16 == 1 && indexer_stack == 1 && use_global_cache &&
        batch_size == 1 && (rows == 512 || rows == 1024) && rows == sequence_length &&
        query->IsDataType<MLFloat16>() && present->IsDataType<MLFloat16>() &&
        num_heads == 4 && head_size == 128 && query_shape[2] == 640 &&
        compress_ratio_ == 4 && token_budget_ == 2048 &&
        rows <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&
        DivideRoundUp(ToUint32(score_stride), 16u) <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&
        score_stride >= 128 && score_stride <= 16384 &&
        context.DeviceLimits().maxComputeWorkgroupStorageSize >= 11620 &&
        context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        phase == QsaCachePhase::Score) {
      const uint32_t phase_sort_capacity = 0;
      const bool phase_dense_map = false;
    QsaPackedScore16Program select{mask != nullptr};
    select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity,
                     mask != nullptr, use_global_cache, static_cast<int>(phase), phase_sort_capacity, phase_dense_map, "prefill-score16-2d")
        .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                    {present, ProgramTensorMetadataDependency::Type},
                    {query_norm, ProgramTensorMetadataDependency::Type},
                    {key_norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type}});
    if (mask != nullptr) {
      select.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
    }
    select.AddOutput({selected, ProgramTensorMetadataDependency::Type});
    if (use_global_cache) {
      select.AddOutputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                         {&score_cache, ProgramTensorMetadataDependency::Type}});
    }
    select.SetWorkgroupSize(256)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 16u), ToUint32(rows))
        .AddUniformVariables({{ToUint32(rows)},
                              {ToUint32(sequence_length)},
                              {ToUint32(num_heads)},
                              {ToUint32(head_size)},
                              {ToUint32(query_shape[2])},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(capacity)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(token_budget_ / compress_ratio_)},
                              {epsilon_},
                              {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
    ORT_RETURN_IF_ERROR(context.RunProgram(select));
      continue;
    }
    if (score16 == 1 && (packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&
        num_heads <= 4 && score_stride >= 2048 &&
        context.DeviceLimits().maxComputeWorkgroupStorageSize >= 11620 &&
        context.DeviceLimits().maxComputeWorkgroupSizeX >= 256 &&
        context.DeviceLimits().maxComputeInvocationsPerWorkgroup >= 256 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        phase == QsaCachePhase::Score) {
      const uint32_t phase_sort_capacity = 0;
      const bool phase_dense_map = false;
    QsaPackedScore16Program select{mask != nullptr};
    select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity,
                     mask != nullptr, use_global_cache, static_cast<int>(phase), phase_sort_capacity, phase_dense_map)
        .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                    {present, ProgramTensorMetadataDependency::Type},
                    {query_norm, ProgramTensorMetadataDependency::Type},
                    {key_norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type}});
    if (mask != nullptr) {
      select.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
    }
    select.AddOutput({selected, ProgramTensorMetadataDependency::Type});
    if (use_global_cache) {
      select.AddOutputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                         {&score_cache, ProgramTensorMetadataDependency::Type}});
    }
    select.SetWorkgroupSize(256)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 16u))
        .AddUniformVariables({{ToUint32(rows)},
                              {ToUint32(sequence_length)},
                              {ToUint32(num_heads)},
                              {ToUint32(head_size)},
                              {ToUint32(query_shape[2])},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(capacity)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(token_budget_ / compress_ratio_)},
                              {epsilon_},
                              {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
    ORT_RETURN_IF_ERROR(context.RunProgram(select));
      continue;
    }
    if ((packed_score == 1 || indexer_stack == 1) && use_global_cache && rows == 1 &&
        num_heads <= 16 && score_stride >= 2048 &&
        context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
        phase == QsaCachePhase::Score) {
      const uint32_t phase_sort_capacity = 0;
      const bool phase_dense_map = false;
    QsaPackedScoreProgram select{mask != nullptr};
    select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity,
                     mask != nullptr, use_global_cache, static_cast<int>(phase), phase_sort_capacity, phase_dense_map)
        .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                    {present, ProgramTensorMetadataDependency::Type},
                    {query_norm, ProgramTensorMetadataDependency::Type},
                    {key_norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type}});
    if (mask != nullptr) {
      select.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
    }
    select.AddOutput({selected, ProgramTensorMetadataDependency::Type});
    if (use_global_cache) {
      select.AddOutputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                         {&score_cache, ProgramTensorMetadataDependency::Type}});
    }
    select.SetWorkgroupSize(kQsaSelectWorkgroupSize)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(score_stride), 4u))
        .AddUniformVariables({{ToUint32(rows)},
                              {ToUint32(sequence_length)},
                              {ToUint32(num_heads)},
                              {ToUint32(head_size)},
                              {ToUint32(query_shape[2])},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(capacity)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(token_budget_ / compress_ratio_)},
                              {epsilon_},
                              {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
    ORT_RETURN_IF_ERROR(context.RunProgram(select));
      continue;
    }
    if (use_parallel_rank && phase == QsaCachePhase::Select) {
      SparseAttentionIndexerQsaParallelRankProgram rank;
      rank.AddInputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                      {&score_cache, ProgramTensorMetadataDependency::Type}})
          .AddOutput({selected, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{ToUint32(total_length)}, {ToUint32(score_stride)},
                                {ToUint32(compress_ratio_)}, {ToUint32(block_topk)},
                                {ToUint32(capacity)}})
          .SetWorkgroupSize(kQsaSelectWorkgroupSize)
          .SetDispatchGroupSize(ToUint32(score_stride));
      return context.RunProgram(rank);
    }
    const uint32_t phase_sort_capacity = phase == QsaCachePhase::Select ? sort_capacity : 0;
    const bool phase_dense_map = use_dense_map && phase == QsaCachePhase::Map;
    SparseAttentionIndexerQsaSelectProgram select{
        mask != nullptr, use_global_cache, phase, phase_sort_capacity, phase_dense_map};
    select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity,
                     mask != nullptr, use_global_cache, static_cast<int>(phase), phase_sort_capacity, phase_dense_map)
        .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                    {present, ProgramTensorMetadataDependency::Type},
                    {query_norm, ProgramTensorMetadataDependency::Type},
                    {key_norm, ProgramTensorMetadataDependency::Type},
                    {cos_cache, ProgramTensorMetadataDependency::Type},
                    {sin_cache, ProgramTensorMetadataDependency::Type}});
    if (mask != nullptr) {
      select.AddInput({mask, ProgramTensorMetadataDependency::Type, {mask->Shape().Size()}, 1});
    }
    select.AddOutput({selected, ProgramTensorMetadataDependency::Type});
    if (use_global_cache) {
      select.AddOutputs({{&visible_cache, ProgramTensorMetadataDependency::Type},
                         {&score_cache, ProgramTensorMetadataDependency::Type}});
    }
    select.SetWorkgroupSize(kQsaSelectWorkgroupSize)
        .SetDispatchGroupSize(ToUint32(use_global_cache && phase == QsaCachePhase::Score ? score_elements : rows))
        .AddUniformVariables({{ToUint32(rows)},
                              {ToUint32(sequence_length)},
                              {ToUint32(num_heads)},
                              {ToUint32(head_size)},
                              {ToUint32(query_shape[2])},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
                              {ToUint32(compress_ratio_)},
                              {ToUint32(capacity)},
                              {ToUint32(past_length)},
                              {ToUint32(total_length)},
                              {ToUint32(token_budget_ / compress_ratio_)},
                              {epsilon_},
                              {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))}});
    ORT_RETURN_IF_ERROR(context.RunProgram(select));
  }
  return Status::OK();
}

Status SparseAttentionIndexer::ComputeCsa(onnxruntime::webgpu::ComputeContext& context) const {
  const Tensor* query = context.Input(sai::kQuery);
  const Tensor* key = context.Input(sai::kKey);
  const Tensor* query_norm = context.Input(sai::kQueryNormWeight);
  const Tensor* key_norm = context.Input(sai::kKeyNormWeight);
  const Tensor* cos_cache = context.Input(sai::kCosCache);
  const Tensor* sin_cache = context.Input(sai::kSinCache);
  const Tensor* gate = context.Input(sai::kGate);
  const Tensor* bias = context.Input(sai::kPositionBias);
  const Tensor* head_weights = context.Input(sai::kHeadWeights);
  const Tensor* position_ids = context.Input(sai::kPositionIds);
  const Tensor* past_compressed = context.Input(sai::kPastKey);
  const Tensor* past_proj = context.Input(sai::kPastProjBuffer);
  ORT_RETURN_IF(context.InputCount() > sai::kPastSequenceLength &&
                    context.Input(sai::kPastSequenceLength) != nullptr,
                "SparseAttentionIndexer WebGPU does not support fixed-capacity caches");

  const auto& query_shape = query->Shape();
  ORT_RETURN_IF_NOT(query_shape.NumDimensions() == 3, "SparseAttentionIndexer: query must have rank 3");
  const int64_t batch_size = query_shape[0];
  const int64_t sequence_length = query_shape[1];
  const auto& query_norm_shape = query_norm->Shape();
  ORT_RETURN_IF_NOT(query_norm_shape.NumDimensions() == 1 && query_norm_shape[0] > 0,
                    "SparseAttentionIndexer: invalid query_norm_weight shape");
  const int64_t head_size = query_norm_shape[0];
  ORT_RETURN_IF_NOT(query_shape[2] > 0 && query_shape[2] % head_size == 0,
                    "SparseAttentionIndexer: query width must be positive and divisible by head_size");
  const int64_t num_heads = query_shape[2] / head_size;
  ORT_RETURN_IF_NOT(num_heads > 0 && head_size > 0, "SparseAttentionIndexer: invalid query dimensions");
  ORT_RETURN_IF(head_size > std::numeric_limits<uint32_t>::max() / 2,
                "SparseAttentionIndexer WebGPU: twice head_size must fit in uint32");
  const int64_t width = 2 * head_size;
  ORT_RETURN_IF_ERROR(CheckShape(key, "key", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(query_norm, "query_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(key_norm, "key_norm_weight", {head_size}));
  ORT_RETURN_IF_ERROR(CheckShape(gate, "gate", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckShape(bias, "position_bias", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckShape(head_weights, "head_weights", {batch_size, sequence_length, num_heads}));
  ORT_RETURN_IF_ERROR(CheckShape(position_ids, "position_ids", {batch_size, sequence_length}));

  RotaryCacheShape rotary_cache_shape;
  ORT_RETURN_IF_ERROR(CheckRotaryCache(cos_cache, sin_cache, batch_size, rotary_cache_shape));
  const int64_t max_rotary_length = rotary_cache_shape.max_rotary_length;
  const int64_t rotary_width = rotary_cache_shape.rotary_width;
  ORT_RETURN_IF_NOT(rotary_width > 0 && 2 * rotary_width <= head_size,
                    "SparseAttentionIndexer: invalid csa rotary cache shape");
  const auto& past_compressed_shape = past_compressed->Shape();
  ORT_RETURN_IF_NOT(past_compressed_shape.NumDimensions() == 3 &&
                        past_compressed_shape[0] == batch_size && past_compressed_shape[2] == head_size,
                    "SparseAttentionIndexer: invalid past_compressed_key shape");
  const int64_t past_compressed_length = past_compressed_shape[1];
  const auto& past_buffer_shape = past_proj->Shape();
  ORT_RETURN_IF_NOT(past_buffer_shape.NumDimensions() == 4 && past_buffer_shape[0] == 2 &&
                        past_buffer_shape[1] == batch_size && past_buffer_shape[3] == width,
                    "SparseAttentionIndexer: invalid past_proj_buffer shape");
  const int64_t past_buffer_length = past_buffer_shape[2];
  ORT_RETURN_IF_ERROR(CheckUint32({{batch_size, "batch_size"},
                                   {sequence_length, "sequence_length"},
                                   {num_heads, "num_heads"},
                                   {head_size, "head_size"},
                                   {width, "projection width"},
                                   {max_rotary_length, "rotary cache length"},
                                   {rotary_width, "rotary width"},
                                   {compress_ratio_, "compress_ratio"},
                                   {index_topk_, "index_topk"},
                                   {past_compressed_length, "past compressed length"},
                                   {past_buffer_length, "past projection buffer length"}}));
  ORT_RETURN_IF(past_buffer_length > std::numeric_limits<uint32_t>::max() - sequence_length,
                "SparseAttentionIndexer WebGPU: combined projection sequence length must fit in uint32");

  sai::CsaWindowPlan plan;
  ORT_RETURN_IF_NOT(sai::TryComputeCsaWindowPlan(past_buffer_length, sequence_length, compress_ratio_, plan),
                    "SparseAttentionIndexer: invalid csa buffer length");
  const int64_t present_compressed_length = past_compressed_length + plan.new_window_count;
  const int64_t capacity = sai::SelectedCapacity(policy_, token_budget_, index_topk_, compress_ratio_);
  ORT_RETURN_IF_ERROR(CheckUint32({{plan.overlap_length, "overlap length"},
                                   {plan.new_window_count, "new window count"},
                                   {plan.present_buffer_length, "present projection buffer length"},
                                   {plan.present_buffer_start, "present projection buffer start"},
                                   {present_compressed_length, "present compressed length"},
                                   {capacity, "selected capacity"}}));
  ORT_RETURN_IF(present_compressed_length > std::numeric_limits<int32_t>::max(),
                "SparseAttentionIndexer WebGPU: csa selected-index range must fit in int32");
  ORT_RETURN_IF_ERROR(CheckUint32Product("query element count", {batch_size, sequence_length, query_shape[2]}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("key and gate element count", {batch_size, sequence_length, width}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("position bias element count", {compress_ratio_, width}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("head weights element count", {batch_size, sequence_length, num_heads}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("position IDs element count", {batch_size, sequence_length}));
  ORT_RETURN_IF_ERROR(
      CheckUint32Product("past compressed key element count", {batch_size, past_compressed_length, head_size}));
  ORT_RETURN_IF_ERROR(
      CheckUint32Product("past projection buffer element count", {2, batch_size, past_buffer_length, width}));
  ORT_RETURN_IF_ERROR(
      CheckUint32Product("compressed key element count", {batch_size, present_compressed_length, head_size}));
  ORT_RETURN_IF_ERROR(
      CheckUint32Product("projection buffer element count", {2, batch_size, plan.present_buffer_length, width}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("selected index element count", {batch_size, sequence_length, capacity}));
  ORT_RETURN_IF_ERROR(CheckUint32Product("rotary cache element count",
                                         {rotary_cache_shape.batched ? batch_size : 1,
                                          max_rotary_length, rotary_width}));
  Tensor* selected =
      context.Output(sai::kSelectedIndices, TensorShape({batch_size, sequence_length, capacity}));
  Tensor* present_compressed =
      context.Output(sai::kPresentKey, TensorShape({batch_size, present_compressed_length, head_size}));
  Tensor* present_proj = context.Output(
      sai::kPresentProjBuffer, TensorShape({2, batch_size, plan.present_buffer_length, width}));
  std::optional<Tensor> scoring_compressed_keys;
  if (present_compressed_length > 0) {
    scoring_compressed_keys.emplace(context.CreateGPUTensor(
        DataTypeImpl::GetType<float>(), TensorShape({batch_size, present_compressed_length, head_size})));
  }

  const int64_t past_compressed_elements = batch_size * past_compressed_length * head_size;
  if (past_compressed_elements > 0) {
    SparseAttentionIndexerCsaCopyCompressedProgram copy;
    copy.CacheHint(query->GetElementType())
        .AddInput({past_compressed, ProgramTensorMetadataDependency::Type})
        .AddOutputs({{present_compressed, ProgramTensorMetadataDependency::Type},
                     {&*scoring_compressed_keys, ProgramTensorMetadataDependency::Type}})
        .SetWorkgroupSize(kElementwiseWorkgroupSize)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(past_compressed_elements), kElementwiseWorkgroupSize))
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
      compress.AddInput({past_proj, ProgramTensorMetadataDependency::Type});
    }
    compress.AddInputs({{bias, ProgramTensorMetadataDependency::Type},
                        {key_norm, ProgramTensorMetadataDependency::Type},
                        {cos_cache, ProgramTensorMetadataDependency::Type},
                        {sin_cache, ProgramTensorMetadataDependency::Type}})
        .AddOutputs({{present_compressed, ProgramTensorMetadataDependency::Type},
                     {&*scoring_compressed_keys, ProgramTensorMetadataDependency::Type}})
        .SetWorkgroupSize(kSerialWorkgroupSize)
        .SetDispatchGroupSize(ToUint32(batch_size * plan.new_window_count))
        .AddUniformVariables({{ToUint32(batch_size * plan.new_window_count)},
                              {ToUint32(sequence_length)},
                              {ToUint32(head_size)},
                              {ToUint32(rotary_width)},
                              {ToUint32(max_rotary_length)},
                              {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
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
      copy_buffer.AddInput({past_proj, ProgramTensorMetadataDependency::Type});
    }
    copy_buffer.AddOutput({present_proj, ProgramTensorMetadataDependency::Type})
        .SetWorkgroupSize(kElementwiseWorkgroupSize)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(present_buffer_elements), kElementwiseWorkgroupSize))
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
        .SetWorkgroupSize(kElementwiseWorkgroupSize)
        .SetDispatchGroupSize(DivideRoundUp(ToUint32(output_elements), kElementwiseWorkgroupSize))
        .AddUniformVariable({ToUint32(output_elements)});
    return context.RunProgram(fill);
  }
  SparseAttentionIndexerCsaSelectProgram select;
  select.CacheHint(query->GetElementType(), num_heads, head_size, rotary_width, compress_ratio_, capacity)
      .AddInputs({{query, ProgramTensorMetadataDependency::Type},
                  {query_norm, ProgramTensorMetadataDependency::Type},
                  {&*scoring_compressed_keys, ProgramTensorMetadataDependency::Type},
                  {head_weights, ProgramTensorMetadataDependency::Type},
                  {position_ids, ProgramTensorMetadataDependency::Type},
                  {cos_cache, ProgramTensorMetadataDependency::Type},
                  {sin_cache, ProgramTensorMetadataDependency::Type}})
      .AddOutput({selected, ProgramTensorMetadataDependency::Type})
      .SetWorkgroupSize(kSerialWorkgroupSize)
      .SetDispatchGroupSize(ToUint32(rows))
      .AddUniformVariables({{ToUint32(rows)},
                            {ToUint32(sequence_length)},
                            {ToUint32(num_heads)},
                            {ToUint32(head_size)},
                            {ToUint32(rotary_width)},
                            {ToUint32(max_rotary_length)},
                            {rotary_cache_shape.batched ? ToUint32(max_rotary_length) : 0u},
                            {ToUint32(compress_ratio_)},
                            {ToUint32(capacity)},
                            {ToUint32(present_compressed_length)},
                            {epsilon_},
                            {has_scale_ ? scale_ : 1.0f / std::sqrt(static_cast<float>(head_size))},
                            {has_head_weight_scale_
                                 ? head_weight_scale_
                                 : 1.0f / std::sqrt(static_cast<float>(num_heads))}});
  return context.RunProgram(select);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
