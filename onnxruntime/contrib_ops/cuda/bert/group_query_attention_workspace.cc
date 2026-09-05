// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

#include <array>
#include <limits>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Mirrors the extra bytes in the runtime GQABufferRequirements quantized Flash decode formula.
constexpr size_t kGQAQuantizedFlashDecodeSlackBytes = 256;

constexpr GQAWorkspaceStatus Ok() noexcept {
  return {};
}

constexpr GQAWorkspaceStatus Invalid(const char* message) noexcept {
  return {GQAWorkspaceError::InvalidArgument, message};
}

constexpr GQAWorkspaceStatus Overflow(const char* message) noexcept {
  return {GQAWorkspaceError::Overflow, message};
}

GQAWorkspaceStatus ValidateDimension(int64_t value, const char* message) noexcept {
  if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
    return Invalid(message);
  }

  return Ok();
}

bool IsValidQuantizationType(GQAKvQuantizationType type) noexcept {
  switch (type) {
    case GQAKvQuantizationType::None:
    case GQAKvQuantizationType::PerTensor:
    case GQAKvQuantizationType::PerChannel:
      return true;
    default:
      return false;
  }
}

GQAWorkspaceStatus ValidateProblem(const GQAWorkspaceProblem& problem,
                                   const GQAPreparationRoute& route) noexcept {
  if (problem.qkv_element_size != 2) {
    return Invalid("GQA preparation element size must match FP16 or BF16 storage.");
  }

  if (problem.cache_element_size != 1 && problem.cache_element_size != 2) {
    return Invalid("GQA cache element size must be one or two bytes.");
  }

  auto status = ValidateDimension(problem.batch_size, "GQA batch size must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateDimension(problem.sequence_length, "GQA sequence length must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateDimension(problem.num_heads, "GQA head count must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateDimension(problem.kv_num_heads, "GQA KV head count must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateDimension(problem.head_size, "GQA head size must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateDimension(
      problem.present_kv_cache_capacity,
      "GQA present KV cache capacity must be positive and fit int32.");
  if (!status.IsOK()) {
    return status;
  }

  if (problem.num_heads % problem.kv_num_heads != 0) {
    return Invalid("GQA num_heads must be a multiple of kv_num_heads.");
  }

  if (problem.head_size % 8 != 0) {
    return Invalid("GQA head size must be a multiple of eight.");
  }

  if (problem.kv_cache_bit_width != 0 &&
      problem.kv_cache_bit_width != 4 &&
      problem.kv_cache_bit_width != 8) {
    return Invalid("GQA KV cache bit width must be zero, four, or eight.");
  }

  if (problem.kv_cache_bit_width != 0 && problem.cache_element_size != 1) {
    return Invalid("Packed GQA KV cache storage must use one-byte elements.");
  }

  if (problem.kv_cache_bit_width == 4 && problem.head_size % 2 != 0) {
    return Invalid("INT4 GQA KV cache rows require an even head size.");
  }

  if (problem.is_windowed_kv_cache) {
    if (problem.kv_cache_bit_width == 8 && problem.head_size % 16 != 0) {
      return Invalid("Windowed eight-bit GQA KV cache rows require head size divisible by 16.");
    }

    if (problem.kv_cache_bit_width == 4 && problem.head_size % 32 != 0) {
      return Invalid("Windowed INT4 GQA KV cache rows require head size divisible by 32.");
    }
  }

  if (!IsValidQuantizationType(problem.k_quantization) ||
      !IsValidQuantizationType(problem.v_quantization)) {
    return Invalid("GQA KV quantization type is invalid.");
  }

  switch (route.preprocess_mode) {
    case GQAPreprocessMode::Xqa:
    case GQAPreprocessMode::Flash:
    case GQAPreprocessMode::MemoryEfficient:
    case GQAPreprocessMode::Fallback:
      break;
    default:
      return Invalid("GQA preprocess mode is invalid.");
  }

  if (route.use_flash_attention_fast_decode &&
      route.preprocess_mode != GQAPreprocessMode::Flash) {
    return Invalid("GQA Flash fast decode requires the Flash preprocess mode.");
  }

  if (route.use_flash_attention_fast_decode && problem.is_first_prompt) {
    return Invalid("GQA Flash fast decode is incompatible with the first prompt.");
  }

  if (route.use_flash_attention_fast_decode && problem.is_windowed_kv_cache) {
    return Invalid("GQA Flash fast decode is incompatible with a windowed KV cache.");
  }

  if (route.use_flash_attention_fast_decode &&
      (problem.k_quantization != GQAKvQuantizationType::None ||
       problem.v_quantization != GQAKvQuantizationType::None)) {
    return Invalid("GQA Flash fast decode requires unquantized K and V.");
  }

  if (route.use_flash_attention_fast_decode && problem.use_qk_norm) {
    return Invalid("GQA Flash fast decode is incompatible with QK-Norm.");
  }

  return Ok();
}

GQAWorkspaceStatus CheckedMultiplyMany(
    size_t first, size_t second, size_t third, size_t fourth, size_t& result) noexcept {
  auto status = CheckedGQAWorkspaceMultiply(first, second, result);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedGQAWorkspaceMultiply(result, third, result);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedGQAWorkspaceMultiply(result, fourth, result);
}

GQAWorkspaceStatus CheckedAddBaseAndTwiceAddend(
    size_t base, size_t addend, size_t& result) noexcept {
  size_t twice_addend = 0;
  auto status = CheckedGQAWorkspaceMultiply(addend, 2, twice_addend);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedGQAWorkspaceAdd(base, twice_addend, result);
}

GQAWorkspaceStatus AppendRegion(size_t bytes, size_t& cursor, size_t& offset) noexcept {
  if (bytes == 0) {
    offset = 0;
    return Ok();
  }

  size_t aligned_cursor = 0;
  auto status = CheckedGQAWorkspaceAlign(cursor, kGQAWorkspaceAlignment, aligned_cursor);
  if (!status.IsOK()) {
    return status;
  }

  size_t end = 0;
  status = CheckedGQAWorkspaceAdd(aligned_cursor, bytes, end);
  if (!status.IsOK()) {
    return status;
  }

  offset = aligned_cursor;
  cursor = end;
  return Ok();
}

GQAWorkspaceStatus ComputeCacheRowBytes(
    const GQAWorkspaceProblem& problem, size_t& cache_row_bytes) noexcept {
  const size_t head_size = static_cast<size_t>(problem.head_size);
  if (problem.kv_cache_bit_width == 0) {
    return CheckedGQAWorkspaceMultiply(head_size, problem.cache_element_size, cache_row_bytes);
  }

  size_t row_bits = 0;
  auto status = CheckedGQAWorkspaceMultiply(
      head_size, static_cast<size_t>(problem.kv_cache_bit_width), row_bits);
  if (!status.IsOK()) {
    return status;
  }

  if (row_bits % 8 != 0) {
    return Invalid("Packed GQA KV cache row size must contain a whole number of bytes.");
  }

  cache_row_bytes = row_bits / 8;
  return Ok();
}

GQAWorkspaceStatus ComputeQkvPreprocessBytes(
    const GQAWorkspaceProblem& problem,
    const GQAPreparationRoute& route,
    int64_t effective_capacity,
    size_t& qkv_preprocess_bytes) noexcept {
  if (route.use_flash_attention_fast_decode) {
    qkv_preprocess_bytes = 0;
    return Ok();
  }

  const size_t qkv_element_size = problem.qkv_element_size;
  const size_t batch_size = static_cast<size_t>(problem.batch_size);
  const size_t sequence_length = static_cast<size_t>(problem.sequence_length);
  const size_t num_heads = static_cast<size_t>(problem.num_heads);
  const size_t kv_num_heads = static_cast<size_t>(problem.kv_num_heads);
  const size_t head_size = static_cast<size_t>(problem.head_size);

  size_t q_elements = 0;
  auto status = CheckedMultiplyMany(
      batch_size, sequence_length, num_heads, head_size, q_elements);
  if (!status.IsOK()) {
    return status;
  }

  size_t k_elements = 0;
  status = CheckedMultiplyMany(
      batch_size, sequence_length, kv_num_heads, head_size, k_elements);
  if (!status.IsOK()) {
    return status;
  }

  size_t preprocess_elements = 0;
  switch (route.preprocess_mode) {
    case GQAPreprocessMode::Xqa:
      // Per-channel K scale is folded into Q before XQA. Per-channel V scale is folded
      // into the attention output, so it does not require a QKV preprocess buffer.
      if (problem.do_rotary || problem.is_packed_qkv || problem.use_qk_norm ||
          problem.k_quantization == GQAKvQuantizationType::PerChannel) {
        preprocess_elements = q_elements;
      }
      break;
    case GQAPreprocessMode::Flash: {
      const bool is_quantized =
          problem.k_quantization != GQAKvQuantizationType::None ||
          problem.v_quantization != GQAKvQuantizationType::None;
      if (is_quantized && !problem.is_first_prompt) {
        size_t full_k_elements = 0;
        status = CheckedMultiplyMany(
            batch_size, static_cast<size_t>(effective_capacity), kv_num_heads, head_size,
            full_k_elements);
        if (!status.IsOK()) {
          return status;
        }

        status = CheckedAddBaseAndTwiceAddend(
            q_elements, full_k_elements, preprocess_elements);
        if (!status.IsOK()) {
          return status;
        }

        status = CheckedGQAWorkspaceMultiply(
            qkv_element_size, preprocess_elements, qkv_preprocess_bytes);
        if (!status.IsOK()) {
          return status;
        }

        return CheckedGQAWorkspaceAdd(
            qkv_preprocess_bytes, kGQAQuantizedFlashDecodeSlackBytes, qkv_preprocess_bytes);
      }

      if (is_quantized) {
        status = CheckedAddBaseAndTwiceAddend(
            q_elements, k_elements, preprocess_elements);
        if (!status.IsOK()) {
          return status;
        }
      } else if (problem.do_rotary || problem.is_packed_qkv || problem.use_qk_norm) {
        preprocess_elements = q_elements;
      }
      break;
    }
    case GQAPreprocessMode::MemoryEfficient:
      if (problem.is_packed_qkv) {
        status = CheckedAddBaseAndTwiceAddend(
            q_elements, k_elements, preprocess_elements);
        if (!status.IsOK()) {
          return status;
        }
      } else if (problem.do_rotary) {
        status = CheckedGQAWorkspaceAdd(q_elements, k_elements, preprocess_elements);
        if (!status.IsOK()) {
          return status;
        }
      } else if (problem.use_qk_norm) {
        preprocess_elements = q_elements;
      }
      break;
    case GQAPreprocessMode::Fallback:
      // The unfused fallback materializes Q for rotary, packed input, or QK-Norm.
      if (problem.do_rotary || problem.is_packed_qkv || problem.use_qk_norm) {
        preprocess_elements = q_elements;
      }
      break;
    default:
      return Invalid("GQA preprocess mode is invalid.");
  }

  return CheckedGQAWorkspaceMultiply(
      qkv_element_size, preprocess_elements, qkv_preprocess_bytes);
}

struct Range {
  size_t offset = 0;
  size_t bytes = 0;
};

GQAWorkspaceStatus ValidateTopLevelRanges(
    const std::array<Range, 5>& ranges, size_t total_bytes) noexcept {
  size_t previous_end = 0;
  bool found_region = false;
  for (const auto& range : ranges) {
    if (range.bytes == 0) {
      if (range.offset != 0) {
        return Invalid("An empty GQA preparation region has a nonzero offset.");
      }
      continue;
    }

    if (range.offset % kGQAWorkspaceAlignment != 0) {
      return Invalid("A GQA preparation region offset is not 256-byte aligned.");
    }

    size_t end = 0;
    auto status = CheckedGQAWorkspaceAdd(range.offset, range.bytes, end);
    if (!status.IsOK()) {
      return status;
    }

    if (end > total_bytes) {
      return Invalid("A GQA preparation region exceeds the preparation allocation.");
    }

    if (found_region && range.offset < previous_end) {
      return Invalid("GQA preparation regions overlap.");
    }

    previous_end = end;
    found_region = true;
  }

  if ((!found_region && total_bytes != 0) ||
      (found_region && previous_end != total_bytes)) {
    return Invalid("GQA preparation total does not match its terminal region.");
  }

  return Ok();
}

}  // namespace

GQAWorkspaceStatus CheckedGQAWorkspaceAdd(size_t left, size_t right, size_t& result) noexcept {
  size_t checked_result = 0;
  if (!SafeAdd(left, right, checked_result)) {
    return Overflow("GQA workspace size addition overflowed size_t.");
  }

  result = checked_result;
  return Ok();
}

GQAWorkspaceStatus CheckedGQAWorkspaceMultiply(size_t left, size_t right, size_t& result) noexcept {
  size_t checked_result = 0;
  if (!SafeMultiply(left, right, checked_result)) {
    return Overflow("GQA workspace size multiplication overflowed size_t.");
  }

  result = checked_result;
  return Ok();
}

GQAWorkspaceStatus CheckedGQAWorkspaceAlign(size_t value, size_t alignment, size_t& result) noexcept {
  if (alignment == 0) {
    return Invalid("GQA workspace alignment must be nonzero.");
  }

  size_t numerator = 0;
  auto status = CheckedGQAWorkspaceAdd(value, alignment - 1, numerator);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedGQAWorkspaceMultiply(numerator / alignment, alignment, result);
}

GQAPreparationResult GetGQAPreparationRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAPreparationRoute& route) noexcept {
  GQAPreparationResult result;
  result.status = ValidateProblem(problem, route);
  if (!result.status.IsOK()) {
    return result;
  }

  GQAPreparationRecipe recipe;
  recipe.effective_kv_cache_capacity = problem.present_kv_cache_capacity;
  result.status = ComputeCacheRowBytes(problem, recipe.cache_row_bytes);
  if (!result.status.IsOK()) {
    return result;
  }

  if (problem.is_windowed_kv_cache && recipe.cache_row_bytes % 16 != 0) {
    result.status = Invalid("Windowed GQA KV cache row size must be a multiple of 16 bytes.");
    return result;
  }

  const size_t batch_size = static_cast<size_t>(problem.batch_size);
  const size_t sequence_length = static_cast<size_t>(problem.sequence_length);
  const size_t kv_num_heads = static_cast<size_t>(problem.kv_num_heads);
  const size_t capacity = static_cast<size_t>(problem.present_kv_cache_capacity);

  size_t cursor = 0;
  if (problem.is_windowed_kv_cache && problem.sequence_length > 1) {
    size_t effective_capacity = 0;
    result.status = CheckedGQAWorkspaceAdd(capacity, sequence_length, effective_capacity);
    if (!result.status.IsOK()) {
      return result;
    }

    if (effective_capacity > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      result.status = Invalid("Windowed GQA staged capacity must fit int32.");
      return result;
    }

    recipe.effective_kv_cache_capacity = static_cast<int64_t>(effective_capacity);
    recipe.uses_staging = true;
    result.status = CheckedMultiplyMany(
        batch_size, kv_num_heads, effective_capacity, recipe.cache_row_bytes,
        recipe.staged_key_bytes);
    if (!result.status.IsOK()) {
      return result;
    }
    recipe.staged_value_bytes = recipe.staged_key_bytes;

    result.status = AppendRegion(
        recipe.staged_key_bytes, cursor, recipe.staged_key_offset_bytes);
    if (!result.status.IsOK()) {
      return result;
    }

    result.status = AppendRegion(
        recipe.staged_value_bytes, cursor, recipe.staged_value_offset_bytes);
    if (!result.status.IsOK()) {
      return result;
    }
  } else if (problem.is_windowed_kv_cache) {
    recipe.uses_compaction = true;
    result.status = CheckedMultiplyMany(
        batch_size, kv_num_heads, capacity, recipe.cache_row_bytes,
        recipe.compaction_key_bytes);
    if (!result.status.IsOK()) {
      return result;
    }
    recipe.compaction_value_bytes = recipe.compaction_key_bytes;

    result.status = CheckedGQAWorkspaceMultiply(
        recipe.compaction_key_bytes, 2, recipe.compaction_bytes);
    if (!result.status.IsOK()) {
      return result;
    }

    result.status = AppendRegion(
        recipe.compaction_bytes, cursor, recipe.compaction_offset_bytes);
    if (!result.status.IsOK()) {
      return result;
    }

    recipe.compaction_key_offset_bytes = recipe.compaction_offset_bytes;
    result.status = CheckedGQAWorkspaceAdd(
        recipe.compaction_key_offset_bytes, recipe.compaction_key_bytes,
        recipe.compaction_value_offset_bytes);
    if (!result.status.IsOK()) {
      return result;
    }
  }

  const bool suppress_sequence_vectors =
      route.use_flash_attention_fast_decode && problem.sequence_length == 1;
  if (!suppress_sequence_vectors) {
    recipe.sequence_length_vector_count = problem.is_windowed_kv_cache ? 6 : 3;
    size_t sequence_length_count = 0;
    result.status = CheckedGQAWorkspaceMultiply(
        recipe.sequence_length_vector_count, batch_size, sequence_length_count);
    if (!result.status.IsOK()) {
      return result;
    }

    result.status = CheckedGQAWorkspaceMultiply(
        sequence_length_count, sizeof(int32_t), recipe.sequence_lengths_bytes);
    if (!result.status.IsOK()) {
      return result;
    }

    result.status = AppendRegion(
        recipe.sequence_lengths_bytes, cursor, recipe.sequence_lengths_offset_bytes);
    if (!result.status.IsOK()) {
      return result;
    }
  }

  result.status = ComputeQkvPreprocessBytes(
      problem, route, recipe.effective_kv_cache_capacity, recipe.qkv_preprocess_bytes);
  if (!result.status.IsOK()) {
    return result;
  }

  result.status = AppendRegion(
      recipe.qkv_preprocess_bytes, cursor, recipe.qkv_preprocess_offset_bytes);
  if (!result.status.IsOK()) {
    return result;
  }

  recipe.total_preparation_bytes = cursor;
  result.status = ValidateGQAPreparationRecipe(recipe);
  if (result.status.IsOK()) {
    result.recipe = recipe;
  }
  return result;
}

GQAWorkspaceStatus ValidateGQAPreparationRecipe(const GQAPreparationRecipe& recipe) noexcept {
  if (recipe.effective_kv_cache_capacity <= 0 || recipe.cache_row_bytes == 0) {
    return Invalid("GQA preparation recipe has invalid cache geometry.");
  }

  if (recipe.uses_staging && recipe.uses_compaction) {
    return Invalid("GQA preparation staging and compaction are mutually exclusive.");
  }

  if (recipe.uses_staging) {
    if (recipe.staged_key_bytes == 0 ||
        recipe.staged_key_bytes != recipe.staged_value_bytes) {
      return Invalid("GQA staged key and value regions must be equal and nonempty.");
    }
  } else if (recipe.staged_key_offset_bytes != 0 || recipe.staged_key_bytes != 0 ||
             recipe.staged_value_offset_bytes != 0 || recipe.staged_value_bytes != 0) {
    return Invalid("A GQA recipe without staging exposes staged regions.");
  }

  if (recipe.uses_compaction) {
    size_t expected_value_offset = 0;
    auto status = CheckedGQAWorkspaceAdd(
        recipe.compaction_key_offset_bytes, recipe.compaction_key_bytes,
        expected_value_offset);
    if (!status.IsOK()) {
      return status;
    }

    size_t expected_compaction_bytes = 0;
    status = CheckedGQAWorkspaceMultiply(
        recipe.compaction_key_bytes, 2, expected_compaction_bytes);
    if (!status.IsOK()) {
      return status;
    }

    if (recipe.compaction_key_bytes == 0 ||
        recipe.compaction_key_bytes != recipe.compaction_value_bytes ||
        recipe.compaction_key_offset_bytes != recipe.compaction_offset_bytes ||
        recipe.compaction_value_offset_bytes != expected_value_offset ||
        recipe.compaction_bytes != expected_compaction_bytes) {
      return Invalid("GQA compaction key and value halves are inconsistent.");
    }
  } else if (recipe.compaction_offset_bytes != 0 || recipe.compaction_bytes != 0 ||
             recipe.compaction_key_offset_bytes != 0 || recipe.compaction_key_bytes != 0 ||
             recipe.compaction_value_offset_bytes != 0 || recipe.compaction_value_bytes != 0) {
    return Invalid("A GQA recipe without compaction exposes compaction regions.");
  }

  if (recipe.sequence_length_vector_count == 0) {
    if (recipe.sequence_lengths_offset_bytes != 0 || recipe.sequence_lengths_bytes != 0) {
      return Invalid("A GQA recipe without sequence vectors exposes a sequence region.");
    }
  } else if ((recipe.sequence_length_vector_count != 3 &&
              recipe.sequence_length_vector_count != 6) ||
             recipe.sequence_lengths_bytes == 0) {
    return Invalid("GQA preparation sequence-vector metadata is invalid.");
  }

  return ValidateTopLevelRanges(
      {{{recipe.staged_key_offset_bytes, recipe.staged_key_bytes},
        {recipe.staged_value_offset_bytes, recipe.staged_value_bytes},
        {recipe.compaction_offset_bytes, recipe.compaction_bytes},
        {recipe.sequence_lengths_offset_bytes, recipe.sequence_lengths_bytes},
        {recipe.qkv_preprocess_offset_bytes, recipe.qkv_preprocess_bytes}}},
      recipe.total_preparation_bytes);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
