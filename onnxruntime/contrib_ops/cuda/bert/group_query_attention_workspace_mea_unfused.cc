// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#include <array>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr GQAWorkspaceStatus Ok() noexcept {
  return {};
}

constexpr GQAWorkspaceStatus Invalid(const char* message) noexcept {
  return {GQAWorkspaceError::InvalidArgument, message};
}

constexpr GQAWorkspaceStatus Unavailable(const char* message) noexcept {
  return {GQAWorkspaceError::Unavailable, message};
}

GQAWorkspaceStatus ValidatePositiveInt32(int64_t value, const char* message) noexcept {
  if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
    return Invalid(message);
  }
  return Ok();
}

GQAWorkspaceStatus ValidateBackendGeometry(const GQAWorkspaceProblem& problem) noexcept {
  if (problem.qkv_element_size != 2) {
    return Invalid("GQA MEA and unfused recipes require FP16 or BF16 storage.");
  }

  auto status = ValidatePositiveInt32(problem.batch_size, "GQA batch size must be positive and fit int32.");
  if (!status.IsOK()) return status;
  status = ValidatePositiveInt32(problem.sequence_length, "GQA sequence length must be positive and fit int32.");
  if (!status.IsOK()) return status;
  status = ValidatePositiveInt32(problem.num_heads, "GQA query head count must be positive and fit int32.");
  if (!status.IsOK()) return status;
  status = ValidatePositiveInt32(problem.kv_num_heads, "GQA KV head count must be positive and fit int32.");
  if (!status.IsOK()) return status;
  status = ValidatePositiveInt32(problem.head_size, "GQA head size must be positive and fit int32.");
  if (!status.IsOK()) return status;
  if (problem.num_heads % problem.kv_num_heads != 0) {
    return Invalid("GQA query head count must be a multiple of KV head count.");
  }
  return Ok();
}

GQAWorkspaceStatus MultiplyMany(
    size_t first, size_t second, size_t third, size_t fourth, size_t& result) noexcept {
  auto status = CheckedGQAWorkspaceMultiply(first, second, result);
  if (!status.IsOK()) return status;
  status = CheckedGQAWorkspaceMultiply(result, third, result);
  if (!status.IsOK()) return status;
  return CheckedGQAWorkspaceMultiply(result, fourth, result);
}

GQAWorkspaceStatus AppendAllocation(
    size_t bytes, size_t& cursor, size_t& offset) noexcept {
  if (bytes == 0) {
    offset = 0;
    return Ok();
  }

  auto status = CheckedGQAWorkspaceAlign(cursor, kGQAWorkspaceAlignment, offset);
  if (!status.IsOK()) return status;
  return CheckedGQAWorkspaceAdd(offset, bytes, cursor);
}

GQAWorkspaceStatus ValidateAllocationRanges(
    const std::array<std::array<size_t, 2>, 3>& ranges,
    size_t total,
    const char* message) noexcept {
  size_t previous_end = 0;
  for (const auto& range : ranges) {
    if (range[1] == 0) {
      if (range[0] != 0) return Invalid(message);
      continue;
    }
    if (range[0] % kGQAWorkspaceAlignment != 0 || range[0] < previous_end) {
      return Invalid(message);
    }
    auto status = CheckedGQAWorkspaceAdd(range[0], range[1], previous_end);
    if (!status.IsOK()) return status;
    if (previous_end > total) return Invalid(message);
  }
  return previous_end == total ? Ok() : Invalid(message);
}

}  // namespace

GQAMemoryEfficientWorkspaceResult GetGQAMemoryEfficientWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    int64_t effective_kv_cache_capacity) noexcept {
  GQAMemoryEfficientWorkspaceResult result;
  result.status = ValidateBackendGeometry(problem);
  if (!result.status.IsOK()) return result;
  if (problem.k_quantization != GQAKvQuantizationType::None ||
      problem.v_quantization != GQAKvQuantizationType::None) {
    result.status = Unavailable("GQA MEA does not support a quantized KV cache.");
    return result;
  }
  result.status = ValidatePositiveInt32(
      effective_kv_cache_capacity,
      "MEA effective KV cache capacity must be positive and fit int32.");
  if (!result.status.IsOK()) return result;

  GQAMemoryEfficientWorkspaceRecipe recipe;
  recipe.effective_kv_cache_capacity = effective_kv_cache_capacity;
  if (problem.num_heads != problem.kv_num_heads) {
    result.status = MultiplyMany(
        problem.qkv_element_size,
        static_cast<size_t>(problem.batch_size),
        static_cast<size_t>(problem.num_heads),
        static_cast<size_t>(effective_kv_cache_capacity),
        recipe.expanded_key_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceMultiply(
        recipe.expanded_key_bytes,
        static_cast<size_t>(problem.head_size),
        recipe.expanded_key_bytes);
    if (!result.status.IsOK()) return result;
    recipe.expanded_value_bytes = recipe.expanded_key_bytes;
  }

  // Matches MemoryEfficientAttentionParams::need_workspace(head_size, is_float).
  if (problem.head_size > 128 && problem.qkv_element_size != sizeof(float)) {
    result.status = MultiplyMany(
        sizeof(float),
        static_cast<size_t>(problem.batch_size),
        static_cast<size_t>(problem.sequence_length),
        static_cast<size_t>(problem.num_heads),
        recipe.output_accumulator_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceMultiply(
        recipe.output_accumulator_bytes,
        static_cast<size_t>(problem.head_size),
        recipe.output_accumulator_bytes);
    if (!result.status.IsOK()) return result;
  }

  size_t cursor = 0;
  result.status = AppendAllocation(
      recipe.expanded_key_bytes, cursor, recipe.expanded_key_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = AppendAllocation(
      recipe.expanded_value_bytes, cursor, recipe.expanded_value_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = AppendAllocation(
      recipe.output_accumulator_bytes, cursor, recipe.output_accumulator_offset_bytes);
  if (!result.status.IsOK()) return result;
  recipe.total_backend_bytes = cursor;

  result.status = ValidateGQAMemoryEfficientWorkspaceRecipe(recipe);
  if (result.status.IsOK()) result.recipe = recipe;
  return result;
}

GQAUnfusedWorkspaceResult GetGQAUnfusedWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    int64_t total_sequence_length) noexcept {
  GQAUnfusedWorkspaceResult result;
  result.status = ValidateBackendGeometry(problem);
  if (!result.status.IsOK()) return result;
  if (problem.k_quantization != GQAKvQuantizationType::None ||
      problem.v_quantization != GQAKvQuantizationType::None) {
    result.status = Unavailable(
        "GQA unfused attention does not support a quantized KV cache.");
    return result;
  }
  result.status = ValidatePositiveInt32(
      total_sequence_length,
      "Unfused total KV sequence length must be positive and fit int32.");
  if (!result.status.IsOK()) return result;
  GQAUnfusedWorkspaceRecipe recipe;
  size_t raw_q_bytes = 0;
  result.status = MultiplyMany(
      problem.qkv_element_size,
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.num_heads),
      static_cast<size_t>(problem.sequence_length),
      raw_q_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceMultiply(
      raw_q_bytes, static_cast<size_t>(problem.head_size), raw_q_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAlign(
      raw_q_bytes, kGQAWorkspaceAlignment, recipe.q_bnsh_bytes);
  if (!result.status.IsOK()) return result;

  size_t raw_y_bytes = 0;
  result.status = MultiplyMany(
      problem.qkv_element_size,
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.num_heads),
      static_cast<size_t>(problem.sequence_length),
      raw_y_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceMultiply(
      raw_y_bytes, static_cast<size_t>(problem.head_size), raw_y_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAlign(
      raw_y_bytes, kGQAWorkspaceAlignment, recipe.y_bnsh_bytes);
  if (!result.status.IsOK()) return result;

  size_t qk_elements = 0;
  result.status = MultiplyMany(
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.num_heads),
      static_cast<size_t>(problem.sequence_length),
      static_cast<size_t>(total_sequence_length),
      qk_elements);
  if (!result.status.IsOK()) return result;
  size_t raw_qk_bytes = 0;
  result.status = CheckedGQAWorkspaceMultiply(
      qk_elements, sizeof(float), raw_qk_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAlign(
      raw_qk_bytes, kGQAWorkspaceAlignment, recipe.qk_bytes);
  if (!result.status.IsOK()) return result;
  recipe.softmax_bytes = recipe.qk_bytes;

  recipe.q_bnsh_offset_bytes = 0;
  recipe.y_bnsh_offset_bytes = recipe.q_bnsh_bytes;
  result.status = CheckedGQAWorkspaceAdd(
      recipe.y_bnsh_offset_bytes, recipe.y_bnsh_bytes, recipe.qk_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAdd(
      recipe.qk_offset_bytes, recipe.qk_bytes, recipe.softmax_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAdd(
      recipe.softmax_offset_bytes, recipe.softmax_bytes, recipe.total_backend_bytes);
  if (!result.status.IsOK()) return result;

  result.status = ValidateGQAUnfusedWorkspaceRecipe(recipe);
  if (result.status.IsOK()) result.recipe = recipe;
  return result;
}

GQAWorkspaceStatus ValidateGQAMemoryEfficientWorkspaceRecipe(
    const GQAMemoryEfficientWorkspaceRecipe& recipe) noexcept {
  if (recipe.effective_kv_cache_capacity <= 0) {
    return Invalid("MEA workspace recipe has invalid effective capacity.");
  }
  if (recipe.expanded_key_bytes != recipe.expanded_value_bytes) {
    return Invalid("MEA expanded K and V allocation sizes differ.");
  }
  return ValidateAllocationRanges(
      {{{recipe.expanded_key_offset_bytes, recipe.expanded_key_bytes},
        {recipe.expanded_value_offset_bytes, recipe.expanded_value_bytes},
        {recipe.output_accumulator_offset_bytes, recipe.output_accumulator_bytes}}},
      recipe.total_backend_bytes,
      "MEA allocation regions overlap, are misaligned, or exceed the backend root.");
}

GQAWorkspaceStatus ValidateGQAUnfusedWorkspaceRecipe(
    const GQAUnfusedWorkspaceRecipe& recipe) noexcept {
  if (recipe.q_bnsh_bytes == 0 || recipe.y_bnsh_bytes == 0 ||
      recipe.qk_bytes == 0 || recipe.softmax_bytes == 0) {
    return Invalid("Unfused workspace recipe regions must be nonempty.");
  }
  if (recipe.q_bnsh_offset_bytes != 0 ||
      recipe.y_bnsh_offset_bytes != recipe.q_bnsh_bytes ||
      recipe.q_bnsh_bytes % kGQAWorkspaceAlignment != 0 ||
      recipe.y_bnsh_bytes % kGQAWorkspaceAlignment != 0 ||
      recipe.qk_bytes % kGQAWorkspaceAlignment != 0 ||
      recipe.softmax_bytes % kGQAWorkspaceAlignment != 0 ||
      recipe.softmax_bytes != recipe.qk_bytes) {
    return Invalid("Unfused combined allocation layout is inconsistent.");
  }

  size_t expected_qk_offset = 0;
  auto status = CheckedGQAWorkspaceAdd(
      recipe.y_bnsh_offset_bytes, recipe.y_bnsh_bytes, expected_qk_offset);
  if (!status.IsOK()) return status;
  size_t expected_softmax_offset = 0;
  status = CheckedGQAWorkspaceAdd(
      expected_qk_offset, recipe.qk_bytes, expected_softmax_offset);
  if (!status.IsOK()) return status;
  size_t expected_total = 0;
  status = CheckedGQAWorkspaceAdd(
      expected_softmax_offset, recipe.softmax_bytes, expected_total);
  if (!status.IsOK()) return status;
  return recipe.qk_offset_bytes == expected_qk_offset &&
                 recipe.softmax_offset_bytes == expected_softmax_offset &&
                 recipe.total_backend_bytes == expected_total
             ? Ok()
             : Invalid("Unfused combined allocation offsets are inconsistent.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
