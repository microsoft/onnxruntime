// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr size_t kXqaAlignment = 128;
constexpr size_t kXqaCtaTile = 256;
constexpr size_t kFlashMaxSplits = 128;

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

GQAWorkspaceStatus ValidateCommonGeometry(const GQAWorkspaceProblem& problem) noexcept {
  if (problem.qkv_element_size != 2) {
    return Invalid("GQA fused backends require FP16 or BF16 storage.");
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

GQAWorkspaceStatus Add(size_t left, size_t right, size_t& result) noexcept {
  return CheckedGQAWorkspaceAdd(left, right, result);
}

GQAWorkspaceStatus Multiply(size_t left, size_t right, size_t& result) noexcept {
  return CheckedGQAWorkspaceMultiply(left, right, result);
}

GQAWorkspaceStatus MultiplyMany(
    size_t first, size_t second, size_t third, size_t fourth, size_t& result) noexcept {
  auto status = Multiply(first, second, result);
  if (!status.IsOK()) return status;
  status = Multiply(result, third, result);
  if (!status.IsOK()) return status;
  return Multiply(result, fourth, result);
}

GQAWorkspaceStatus AppendAllocation(
    size_t bytes, size_t& cursor, size_t& offset) noexcept {
  if (bytes == 0) {
    offset = 0;
    return Ok();
  }

  auto status = CheckedGQAWorkspaceAlign(cursor, kGQAWorkspaceAlignment, offset);
  if (!status.IsOK()) return status;
  return Add(offset, bytes, cursor);
}

GQAWorkspaceStatus CheckedCeilDivide(
    size_t value, size_t divisor, size_t& result) noexcept {
  if (divisor == 0) {
    return Invalid("GQA workspace division requires a nonzero divisor.");
  }

  size_t numerator = 0;
  auto status = Add(value, divisor - 1, numerator);
  if (!status.IsOK()) return status;
  result = numerator / divisor;
  return Ok();
}

bool IsSupportedXqaGroup(size_t group_size, GQAXqaKvType kv_type) noexcept {
  if (kv_type == GQAXqaKvType::None) {
    return group_size == 1 || group_size == 2 || group_size == 4 || group_size == 5 ||
           group_size == 8 || group_size == 16 || group_size == 32;
  }
  return group_size == 4 || group_size == 8 || group_size == 16 || group_size == 32;
}

GQAWorkspaceStatus ValidateRange(
    size_t offset, size_t bytes, size_t total, const char* message) noexcept {
  if (bytes == 0) {
    return offset == 0 ? Ok() : Invalid(message);
  }
  size_t end = 0;
  auto status = Add(offset, bytes, end);
  if (!status.IsOK()) return status;
  return end <= total ? Ok() : Invalid(message);
}

GQAWorkspaceStatus ComputeFlashSplitCount(
    size_t batch_size,
    size_t sequence_length,
    size_t kv_length,
    size_t head_count,
    size_t head_size,
    size_t multi_processor_count,
    size_t& split_count) noexcept {
  const size_t block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  size_t kv_blocks = 0;
  auto status = CheckedCeilDivide(kv_length, block_n, kv_blocks);
  if (!status.IsOK()) return status;

  size_t query_blocks = 0;
  status = CheckedCeilDivide(sequence_length, 64, query_blocks);
  if (!status.IsOK()) return status;

  size_t work_items = 0;
  status = MultiplyMany(batch_size, head_count, query_blocks, 1, work_items);
  if (!status.IsOK()) return status;

  if (static_cast<float>(work_items) >= 0.8f * static_cast<float>(multi_processor_count)) {
    split_count = 1;
    return Ok();
  }

  const size_t max_splits = std::min({kFlashMaxSplits, multi_processor_count, kv_blocks});
  if (max_splits <= 1) {
    split_count = 1;
    return Ok();
  }

  std::array<float, kFlashMaxSplits> efficiencies{};
  float maximum_efficiency = 0.0f;
  size_t previous_blocks_per_split = 0;
  for (size_t candidate = 1; candidate <= max_splits; ++candidate) {
    size_t blocks_per_split = 0;
    status = CheckedCeilDivide(kv_blocks, candidate, blocks_per_split);
    if (!status.IsOK()) return status;

    const bool eligible = candidate == 1 || blocks_per_split != previous_blocks_per_split;
    previous_blocks_per_split = blocks_per_split;
    if (!eligible) continue;

    size_t split_work_items = 0;
    status = Multiply(work_items, candidate, split_work_items);
    if (!status.IsOK()) return status;
    const float waves = static_cast<float>(split_work_items) /
                        static_cast<float>(multi_processor_count);
    const float efficiency = waves / std::ceil(waves);
    efficiencies[candidate - 1] = efficiency;
    maximum_efficiency = std::max(maximum_efficiency, efficiency);
  }

  previous_blocks_per_split = 0;
  for (size_t candidate = 1; candidate <= max_splits; ++candidate) {
    size_t blocks_per_split = 0;
    status = CheckedCeilDivide(kv_blocks, candidate, blocks_per_split);
    if (!status.IsOK()) return status;
    const bool eligible = candidate == 1 || blocks_per_split != previous_blocks_per_split;
    previous_blocks_per_split = blocks_per_split;
    // Match flash::num_splits_heuristic exactly: the unsuffixed literal promotes
    // maximum_efficiency to double at threshold boundaries.
    if (eligible && efficiencies[candidate - 1] >= 0.85 * maximum_efficiency) {
      split_count = candidate;
      return Ok();
    }
  }

  split_count = 1;
  return Ok();
}

}  // namespace

GQAXqaWorkspaceResult GetGQAXqaWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAXqaConfig& config) noexcept {
  GQAXqaWorkspaceResult result;
  result.status = ValidateCommonGeometry(problem);
  if (!result.status.IsOK()) return result;
  result.status = ValidatePositiveInt32(
      problem.present_kv_cache_capacity,
      "XQA present KV cache capacity must be positive and fit int32.");
  if (!result.status.IsOK()) return result;
  if (problem.sequence_length != 1) {
    result.status = Unavailable("XQA supports single-token decode.");
    return result;
  }
  if (problem.is_first_prompt) {
    result.status = Invalid("A selected XQA route cannot be the first prompt.");
    return result;
  }

  if (config.device_major < 8) {
    result.status = Unavailable("XQA requires device major version 8 or newer.");
    return result;
  }
  if (config.device_minor < 0 || config.device_minor > 99) {
    result.status = Invalid("XQA device minor version is invalid.");
    return result;
  }
  if (config.multi_processor_count <= 0 ||
      static_cast<uint64_t>(config.multi_processor_count) > std::numeric_limits<uint32_t>::max()) {
    result.status = Invalid("XQA multiprocessor count must be positive and fit uint32.");
    return result;
  }
  if (config.kv_type != GQAXqaKvType::None &&
      config.kv_type != GQAXqaKvType::Int8 &&
      config.kv_type != GQAXqaKvType::Fp8) {
    result.status = Invalid("XQA KV type is invalid.");
    return result;
  }
  const bool problem_has_quantized_kv =
      problem.k_quantization != GQAKvQuantizationType::None ||
      problem.v_quantization != GQAKvQuantizationType::None;
  if (problem.kv_cache_bit_width == 4) {
    result.status = Invalid("XQA does not support a four-bit KV cache.");
    return result;
  }
  if (config.kv_type == GQAXqaKvType::None) {
    if (problem_has_quantized_kv || problem.cache_element_size != 2 ||
        problem.kv_cache_bit_width != 0) {
      result.status = Invalid(
          "Non-quantized XQA requires bit width zero and unquantized two-byte K and V storage.");
      return result;
    }
  } else {
    const auto is_supported_quantization = [](GQAKvQuantizationType type) {
      return type == GQAKvQuantizationType::PerTensor ||
             type == GQAKvQuantizationType::PerChannel;
    };
    if (problem.cache_element_size != 1 ||
        problem.kv_cache_bit_width != 8 ||
        !is_supported_quantization(problem.k_quantization) ||
        !is_supported_quantization(problem.v_quantization)) {
      result.status = Invalid(
          "Quantized XQA requires bit width eight and quantized one-byte K and V storage.");
      return result;
    }
    if (problem.use_qk_norm) {
      result.status = Invalid("Quantized XQA does not support QK-Norm.");
      return result;
    }
  }
  if (config.kv_type == GQAXqaKvType::Fp8 &&
      !(config.device_major >= 9 ||
        (config.device_major == 8 && config.device_minor == 9))) {
    result.status = Unavailable("FP8 XQA requires SM89 or device major version 9 or newer.");
    return result;
  }
  if (config.head_sink_storage != GQAXqaHeadSinkStorage::None &&
      config.head_sink_storage != GQAXqaHeadSinkStorage::PrepackedFp32 &&
      config.head_sink_storage != GQAXqaHeadSinkStorage::DynamicConversion) {
    result.status = Invalid("XQA head-sink storage mode is invalid.");
    return result;
  }

  const size_t head_size = static_cast<size_t>(problem.head_size);
  if (head_size != 64 && head_size != 128 && head_size != 256) {
    result.status = Unavailable("XQA supports head sizes 64, 128, and 256.");
    return result;
  }
  const size_t group_size = static_cast<size_t>(problem.num_heads / problem.kv_num_heads);
  if (!IsSupportedXqaGroup(group_size, config.kv_type)) {
    result.status = Unavailable("XQA does not support this query-to-KV head group.");
    return result;
  }

  GQAXqaWorkspaceRecipe recipe;
  result.status = Multiply(
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.kv_num_heads),
      recipe.sequence_count);
  if (!result.status.IsOK()) return result;
  if (recipe.sequence_count > std::numeric_limits<uint32_t>::max()) {
    result.status = Invalid("XQA batch and KV-head sequence count must fit uint32.");
    return result;
  }

  size_t cache_tiles = 0;
  result.status = CheckedCeilDivide(
      static_cast<size_t>(problem.present_kv_cache_capacity), kXqaCtaTile, cache_tiles);
  if (!result.status.IsOK()) return result;
  const size_t resident_sequences =
      static_cast<size_t>(config.multi_processor_count) / recipe.sequence_count;
  recipe.subsequences_per_sequence =
      std::min(std::max<size_t>(1, resident_sequences), cache_tiles);
  result.status = Multiply(
      recipe.sequence_count, recipe.subsequences_per_sequence, recipe.subsequence_count);
  if (!result.status.IsOK()) return result;
  if (recipe.subsequence_count > std::numeric_limits<uint32_t>::max()) {
    result.status = Invalid("XQA subsequence count must fit uint32.");
    return result;
  }

  recipe.m_tile_size = group_size <= 8 ? 8 : (group_size <= 16 ? 16 : 32);
  result.status = Multiply(recipe.sequence_count, sizeof(uint32_t), recipe.semaphore_bytes);
  if (!result.status.IsOK()) return result;
  result.status = CheckedGQAWorkspaceAlign(
      recipe.semaphore_bytes, kXqaAlignment, recipe.semaphore_aligned_bytes);
  if (!result.status.IsOK()) return result;

  size_t scratch_cursor = 0;
  recipe.row_max_offset_bytes = recipe.semaphore_aligned_bytes;
  result.status = Multiply(kXqaAlignment, recipe.subsequence_count, recipe.row_max_bytes);
  if (!result.status.IsOK()) return result;
  result.status = Add(scratch_cursor, recipe.row_max_bytes, scratch_cursor);
  if (!result.status.IsOK()) return result;

  size_t row_sum_scratch_offset = 0;
  result.status = CheckedGQAWorkspaceAlign(
      scratch_cursor, kXqaAlignment, row_sum_scratch_offset);
  if (!result.status.IsOK()) return result;
  result.status = Add(
      recipe.semaphore_aligned_bytes,
      row_sum_scratch_offset,
      recipe.row_sum_offset_bytes);
  if (!result.status.IsOK()) return result;
  recipe.row_sum_bytes = recipe.row_max_bytes;
  result.status = Add(row_sum_scratch_offset, recipe.row_sum_bytes, scratch_cursor);
  if (!result.status.IsOK()) return result;

  size_t vector_bytes = 0;
  result.status = MultiplyMany(head_size, recipe.m_tile_size, 2, 1, vector_bytes);
  if (!result.status.IsOK()) return result;
  size_t output_scratch_offset = 0;
  result.status = CheckedGQAWorkspaceAlign(
      scratch_cursor, vector_bytes, output_scratch_offset);
  if (!result.status.IsOK()) return result;
  result.status = Add(
      recipe.semaphore_aligned_bytes,
      output_scratch_offset,
      recipe.output_accumulator_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = Multiply(
      vector_bytes, recipe.subsequence_count, recipe.output_accumulator_bytes);
  if (!result.status.IsOK()) return result;
  result.status = Add(
      output_scratch_offset, recipe.output_accumulator_bytes, scratch_cursor);
  if (!result.status.IsOK()) return result;
  result.status = Add(
      recipe.semaphore_aligned_bytes, scratch_cursor, recipe.internal_scratch_bytes);
  if (!result.status.IsOK()) return result;
  size_t cursor = recipe.internal_scratch_bytes;

  if (problem.do_rotary) {
    // Retain the runtime's aligned RoPE Q/K bytes for exact allocation parity.
    // GQABufferRequirements may later rebind data.qkv_buffer; changing that
    // runtime allocation is an optimization outside this recipe's scope.
    size_t q_bytes = 0;
    result.status = MultiplyMany(
        static_cast<size_t>(problem.batch_size),
        static_cast<size_t>(problem.num_heads),
        head_size,
        problem.qkv_element_size,
        q_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceAlign(
        q_bytes, kGQAWorkspaceAlignment, recipe.rotary_q_bytes);
    if (!result.status.IsOK()) return result;
    recipe.rotary_q_offset_bytes = cursor;
    result.status = Add(cursor, recipe.rotary_q_bytes, cursor);
    if (!result.status.IsOK()) return result;

    size_t k_bytes = 0;
    result.status = MultiplyMany(
        static_cast<size_t>(problem.batch_size),
        static_cast<size_t>(problem.kv_num_heads),
        head_size,
        problem.qkv_element_size,
        k_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceAlign(
        k_bytes, kGQAWorkspaceAlignment, recipe.rotary_k_bytes);
    if (!result.status.IsOK()) return result;
    recipe.rotary_k_offset_bytes = cursor;
    result.status = Add(cursor, recipe.rotary_k_bytes, cursor);
    if (!result.status.IsOK()) return result;
  }

  if (config.head_sink_storage == GQAXqaHeadSinkStorage::DynamicConversion) {
    size_t head_sink_bytes = 0;
    result.status = Multiply(
        static_cast<size_t>(problem.num_heads), sizeof(float), head_sink_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceAlign(
        head_sink_bytes, kGQAWorkspaceAlignment, recipe.dynamic_head_sink_bytes);
    if (!result.status.IsOK()) return result;
    recipe.dynamic_head_sink_offset_bytes = cursor;
    result.status = Add(cursor, recipe.dynamic_head_sink_bytes, cursor);
    if (!result.status.IsOK()) return result;
  }

  recipe.total_backend_bytes = cursor;
  result.status = ValidateGQAXqaWorkspaceRecipe(recipe);
  if (result.status.IsOK()) result.recipe = recipe;
  return result;
}

GQAFlashWorkspaceResult GetGQAFlashWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAFlashConfig& config) noexcept {
  GQAFlashWorkspaceResult result;
  result.status = ValidateCommonGeometry(problem);
  if (!result.status.IsOK()) return result;
  result.status = ValidatePositiveInt32(
      config.total_sequence_length,
      "Flash total sequence length must be positive and fit int32.");
  if (!result.status.IsOK()) return result;
  if (config.multi_processor_count <= 0) {
    result.status = Invalid("Flash multiprocessor count must be positive.");
    return result;
  }
  if (config.local_window_size != -1 &&
      (config.local_window_size <= 0 ||
       config.local_window_size > std::numeric_limits<int32_t>::max())) {
    result.status = Invalid("Flash local window size must be -1 or a positive int32.");
    return result;
  }
  if (problem.head_size > 256 || problem.head_size % 8 != 0) {
    result.status = Unavailable("Flash requires a head size no greater than 256 and divisible by eight.");
    return result;
  }

  GQAFlashWorkspaceRecipe recipe;
  recipe.split_heuristic_head_count = static_cast<size_t>(
      config.fast_decode ? problem.kv_num_heads : problem.num_heads);
  recipe.split_heuristic_kv_length = static_cast<size_t>(config.total_sequence_length);
  if (config.fast_decode && config.local_window_size > 0) {
    recipe.split_heuristic_kv_length = std::min(
        recipe.split_heuristic_kv_length,
        static_cast<size_t>(config.local_window_size));
  }

  result.status = ComputeFlashSplitCount(
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.sequence_length),
      recipe.split_heuristic_kv_length,
      recipe.split_heuristic_head_count,
      static_cast<size_t>(problem.head_size),
      static_cast<size_t>(config.multi_processor_count),
      recipe.selected_split_count);
  if (!result.status.IsOK()) return result;
  recipe.runtime_num_splits = recipe.selected_split_count > 1
                                  ? recipe.selected_split_count
                                  : 0;

  result.status = CheckedGQAWorkspaceAlign(
      static_cast<size_t>(problem.head_size), 32, recipe.rounded_head_size);
  if (!result.status.IsOK()) return result;

  size_t lse_elements = 0;
  result.status = MultiplyMany(
      static_cast<size_t>(problem.batch_size),
      static_cast<size_t>(problem.num_heads),
      static_cast<size_t>(problem.sequence_length),
      sizeof(float),
      recipe.softmax_lse_bytes);
  if (!result.status.IsOK()) return result;

  if (recipe.selected_split_count > 1) {
    result.status = MultiplyMany(
        recipe.selected_split_count,
        static_cast<size_t>(problem.batch_size),
        static_cast<size_t>(problem.sequence_length),
        static_cast<size_t>(problem.num_heads),
        lse_elements);
    if (!result.status.IsOK()) return result;
    result.status = Multiply(
        lse_elements, sizeof(float), recipe.softmax_lse_accumulator_bytes);
    if (!result.status.IsOK()) return result;

    size_t output_elements = 0;
    result.status = Multiply(lse_elements, recipe.rounded_head_size, output_elements);
    if (!result.status.IsOK()) return result;
    result.status = Multiply(
        output_elements, sizeof(float), recipe.output_accumulator_bytes);
    if (!result.status.IsOK()) return result;
  }

  size_t cursor = 0;
  result.status = AppendAllocation(
      recipe.softmax_lse_bytes, cursor, recipe.softmax_lse_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = AppendAllocation(
      recipe.softmax_lse_accumulator_bytes,
      cursor,
      recipe.softmax_lse_accumulator_offset_bytes);
  if (!result.status.IsOK()) return result;
  result.status = AppendAllocation(
      recipe.output_accumulator_bytes, cursor, recipe.output_accumulator_offset_bytes);
  if (!result.status.IsOK()) return result;
  recipe.total_backend_bytes = cursor;

  result.status = ValidateGQAFlashWorkspaceRecipe(recipe);
  if (result.status.IsOK()) result.recipe = recipe;
  return result;
}

GQAWorkspaceStatus ValidateGQAXqaWorkspaceRecipe(
    const GQAXqaWorkspaceRecipe& recipe) noexcept {
  if (recipe.sequence_count == 0 || recipe.subsequences_per_sequence == 0 ||
      recipe.subsequence_count == 0 ||
      (recipe.m_tile_size != 8 && recipe.m_tile_size != 16 && recipe.m_tile_size != 32) ||
      recipe.internal_scratch_bytes == 0 ||
      recipe.total_backend_bytes < recipe.internal_scratch_bytes) {
    return Invalid("XQA workspace recipe has invalid core geometry.");
  }
  if (recipe.semaphore_offset_bytes != 0 ||
      recipe.semaphore_aligned_bytes < recipe.semaphore_bytes ||
      recipe.semaphore_aligned_bytes % kXqaAlignment != 0 ||
      recipe.row_max_offset_bytes != recipe.semaphore_aligned_bytes ||
      recipe.output_accumulator_bytes == 0) {
    return Invalid("XQA internal scratch layout is inconsistent.");
  }

  size_t expected_subsequence_count = 0;
  auto status = Multiply(
      recipe.sequence_count,
      recipe.subsequences_per_sequence,
      expected_subsequence_count);
  if (!status.IsOK()) return status;
  size_t expected_semaphore_bytes = 0;
  status = Multiply(recipe.sequence_count, sizeof(uint32_t), expected_semaphore_bytes);
  if (!status.IsOK()) return status;
  size_t expected_semaphore_aligned_bytes = 0;
  status = CheckedGQAWorkspaceAlign(
      expected_semaphore_bytes, kXqaAlignment, expected_semaphore_aligned_bytes);
  if (!status.IsOK()) return status;
  size_t expected_row_bytes = 0;
  status = Multiply(kXqaAlignment, recipe.subsequence_count, expected_row_bytes);
  if (!status.IsOK()) return status;
  if (recipe.subsequence_count != expected_subsequence_count ||
      recipe.semaphore_bytes != expected_semaphore_bytes ||
      recipe.semaphore_aligned_bytes != expected_semaphore_aligned_bytes ||
      recipe.row_max_bytes != expected_row_bytes ||
      recipe.row_sum_bytes != expected_row_bytes) {
    return Invalid("XQA internal scratch sizes do not match its sequence geometry.");
  }

  size_t row_max_end = 0;
  status = Add(recipe.row_max_offset_bytes, recipe.row_max_bytes, row_max_end);
  if (!status.IsOK()) return status;
  size_t expected_row_sum_offset = 0;
  status = CheckedGQAWorkspaceAlign(row_max_end, kXqaAlignment, expected_row_sum_offset);
  if (!status.IsOK()) return status;
  if (recipe.row_sum_offset_bytes != expected_row_sum_offset) {
    return Invalid("XQA row-max and row-sum scratch regions are not ordered and contiguous.");
  }

  size_t row_sum_end = 0;
  status = Add(recipe.row_sum_offset_bytes, recipe.row_sum_bytes, row_sum_end);
  if (!status.IsOK()) return status;
  if (row_sum_end < recipe.semaphore_aligned_bytes ||
      recipe.output_accumulator_bytes % recipe.subsequence_count != 0) {
    return Invalid("XQA output accumulator geometry is inconsistent.");
  }

  const size_t vector_bytes =
      recipe.output_accumulator_bytes / recipe.subsequence_count;
  if (vector_bytes == 0) {
    return Invalid("XQA output accumulator vector size must be nonzero.");
  }
  const size_t row_sum_scratch_end = row_sum_end - recipe.semaphore_aligned_bytes;
  size_t expected_output_scratch_offset = 0;
  status = CheckedGQAWorkspaceAlign(
      row_sum_scratch_end, vector_bytes, expected_output_scratch_offset);
  if (!status.IsOK()) return status;
  size_t expected_output_offset = 0;
  status = Add(
      recipe.semaphore_aligned_bytes,
      expected_output_scratch_offset,
      expected_output_offset);
  if (!status.IsOK()) return status;
  if (recipe.output_accumulator_offset_bytes != expected_output_offset) {
    return Invalid(
        "XQA output accumulator is not ordered or aligned to its defining vector size.");
  }

  size_t expected_internal_scratch_bytes = 0;
  status = Add(
      recipe.output_accumulator_offset_bytes,
      recipe.output_accumulator_bytes,
      expected_internal_scratch_bytes);
  if (!status.IsOK()) return status;
  if (recipe.internal_scratch_bytes != expected_internal_scratch_bytes) {
    return Invalid(
        "XQA internal scratch does not end at the output accumulator.");
  }

  if ((recipe.rotary_q_bytes == 0) != (recipe.rotary_k_bytes == 0)) {
    return Invalid("XQA extra scratch alignment or RoPE pairing is inconsistent.");
  }
  size_t expected_extra_offset = recipe.internal_scratch_bytes;
  if (recipe.rotary_q_bytes != 0) {
    if (recipe.rotary_q_offset_bytes != expected_extra_offset ||
        recipe.rotary_q_bytes % kGQAWorkspaceAlignment != 0 ||
        recipe.rotary_k_bytes % kGQAWorkspaceAlignment != 0) {
      return Invalid("XQA RoPE scratch is misaligned or non-contiguous.");
    }
    status = Add(expected_extra_offset, recipe.rotary_q_bytes, expected_extra_offset);
    if (!status.IsOK()) return status;
    if (recipe.rotary_k_offset_bytes != expected_extra_offset) {
      return Invalid("XQA RoPE K scratch does not follow RoPE Q scratch.");
    }
    status = Add(expected_extra_offset, recipe.rotary_k_bytes, expected_extra_offset);
    if (!status.IsOK()) return status;
  } else if (recipe.rotary_q_offset_bytes != 0 || recipe.rotary_k_offset_bytes != 0) {
    return Invalid("An XQA recipe without RoPE scratch exposes RoPE offsets.");
  }

  if (recipe.dynamic_head_sink_bytes != 0) {
    if (recipe.dynamic_head_sink_offset_bytes != expected_extra_offset ||
        recipe.dynamic_head_sink_bytes % kGQAWorkspaceAlignment != 0) {
      return Invalid("XQA dynamic head-sink scratch is misaligned or non-contiguous.");
    }
    status = Add(
        expected_extra_offset, recipe.dynamic_head_sink_bytes, expected_extra_offset);
    if (!status.IsOK()) return status;
  } else if (recipe.dynamic_head_sink_offset_bytes != 0) {
    return Invalid("An XQA recipe without dynamic head-sink scratch exposes an offset.");
  }

  status = ValidateRange(
      recipe.rotary_q_offset_bytes, recipe.rotary_q_bytes,
      recipe.total_backend_bytes, "XQA RoPE Q scratch is out of bounds.");
  if (!status.IsOK()) return status;
  status = ValidateRange(
      recipe.rotary_k_offset_bytes, recipe.rotary_k_bytes,
      recipe.total_backend_bytes, "XQA RoPE K scratch is out of bounds.");
  if (!status.IsOK()) return status;
  status = ValidateRange(
      recipe.dynamic_head_sink_offset_bytes, recipe.dynamic_head_sink_bytes,
      recipe.total_backend_bytes, "XQA dynamic head-sink scratch is out of bounds.");
  if (!status.IsOK()) return status;
  return expected_extra_offset == recipe.total_backend_bytes
             ? Ok()
             : Invalid("XQA backend total does not match its terminal scratch region.");
}

GQAWorkspaceStatus ValidateGQAFlashWorkspaceRecipe(
    const GQAFlashWorkspaceRecipe& recipe) noexcept {
  if (recipe.selected_split_count == 0 || recipe.softmax_lse_bytes == 0 ||
      recipe.rounded_head_size == 0) {
    return Invalid("Flash workspace recipe has invalid geometry.");
  }
  if ((recipe.selected_split_count == 1 &&
       (recipe.runtime_num_splits != 0 ||
        recipe.softmax_lse_accumulator_bytes != 0 ||
        recipe.output_accumulator_bytes != 0)) ||
      (recipe.selected_split_count > 1 &&
       (recipe.runtime_num_splits != recipe.selected_split_count ||
        recipe.softmax_lse_accumulator_bytes == 0 ||
        recipe.output_accumulator_bytes == 0))) {
    return Invalid("Flash split metadata and accumulators are inconsistent.");
  }

  const std::array<std::array<size_t, 2>, 3> ranges{{
      {recipe.softmax_lse_offset_bytes, recipe.softmax_lse_bytes},
      {recipe.softmax_lse_accumulator_offset_bytes, recipe.softmax_lse_accumulator_bytes},
      {recipe.output_accumulator_offset_bytes, recipe.output_accumulator_bytes},
  }};
  size_t previous_end = 0;
  for (const auto& range : ranges) {
    if (range[1] == 0) {
      if (range[0] != 0) return Invalid("An empty Flash region has a nonzero offset.");
      continue;
    }
    if (range[0] % kGQAWorkspaceAlignment != 0 || range[0] < previous_end) {
      return Invalid("Flash allocation regions overlap or are misaligned.");
    }
    auto status = Add(range[0], range[1], previous_end);
    if (!status.IsOK()) return status;
    if (previous_end > recipe.total_backend_bytes) {
      return Invalid("A Flash allocation region exceeds the backend root.");
    }
  }
  return previous_end == recipe.total_backend_bytes
             ? Ok()
             : Invalid("Flash backend total does not match its terminal region.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
