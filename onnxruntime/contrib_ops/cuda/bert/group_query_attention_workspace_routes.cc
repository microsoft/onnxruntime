// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

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

bool IsDefault(const GQAXqaWorkspaceRecipe& recipe) noexcept {
  return recipe.sequence_count == 0 &&
         recipe.subsequences_per_sequence == 0 &&
         recipe.subsequence_count == 0 &&
         recipe.m_tile_size == 0 &&
         recipe.semaphore_offset_bytes == 0 &&
         recipe.semaphore_bytes == 0 &&
         recipe.semaphore_aligned_bytes == 0 &&
         recipe.row_max_offset_bytes == 0 &&
         recipe.row_max_bytes == 0 &&
         recipe.row_sum_offset_bytes == 0 &&
         recipe.row_sum_bytes == 0 &&
         recipe.output_accumulator_offset_bytes == 0 &&
         recipe.output_accumulator_bytes == 0 &&
         recipe.internal_scratch_bytes == 0 &&
         recipe.rotary_q_offset_bytes == 0 &&
         recipe.rotary_q_bytes == 0 &&
         recipe.rotary_k_offset_bytes == 0 &&
         recipe.rotary_k_bytes == 0 &&
         recipe.dynamic_head_sink_offset_bytes == 0 &&
         recipe.dynamic_head_sink_bytes == 0 &&
         recipe.total_backend_bytes == 0;
}

bool IsDefault(const GQAFlashWorkspaceRecipe& recipe) noexcept {
  return !recipe.fast_decode &&
         recipe.split_heuristic_head_count == 0 &&
         recipe.split_heuristic_kv_length == 0 &&
         recipe.selected_split_count == 1 &&
         recipe.runtime_num_splits == 0 &&
         recipe.rounded_head_size == 0 &&
         recipe.softmax_lse_offset_bytes == 0 &&
         recipe.softmax_lse_bytes == 0 &&
         recipe.softmax_lse_accumulator_offset_bytes == 0 &&
         recipe.softmax_lse_accumulator_bytes == 0 &&
         recipe.output_accumulator_offset_bytes == 0 &&
         recipe.output_accumulator_bytes == 0 &&
         recipe.total_backend_bytes == 0;
}

bool IsDefault(const GQAMemoryEfficientWorkspaceRecipe& recipe) noexcept {
  return recipe.effective_kv_cache_capacity == 0 &&
         recipe.expanded_key_offset_bytes == 0 &&
         recipe.expanded_key_bytes == 0 &&
         recipe.expanded_value_offset_bytes == 0 &&
         recipe.expanded_value_bytes == 0 &&
         recipe.output_accumulator_offset_bytes == 0 &&
         recipe.output_accumulator_bytes == 0 &&
         recipe.total_backend_bytes == 0;
}

bool IsDefault(const GQAUnfusedWorkspaceRecipe& recipe) noexcept {
  return recipe.q_bnsh_offset_bytes == 0 &&
         recipe.q_bnsh_bytes == 0 &&
         recipe.y_bnsh_offset_bytes == 0 &&
         recipe.y_bnsh_bytes == 0 &&
         recipe.qk_offset_bytes == 0 &&
         recipe.qk_bytes == 0 &&
         recipe.softmax_offset_bytes == 0 &&
         recipe.softmax_bytes == 0 &&
         recipe.total_backend_bytes == 0;
}

}  // namespace

GQACompleteWorkspaceResult GetGQACompleteWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAConcreteRoute& route) noexcept {
  GQACompleteWorkspaceResult result;
  if (route.backend == GQABackend::Cudnn) {
    result.status = Unavailable(
        "A graph-free cuDNN GQA workspace recipe is not provided.");
    return result;
  }

  const bool route_matches_preparation =
      (route.backend == GQABackend::Xqa &&
       route.preparation.preprocess_mode == GQAPreprocessMode::Xqa) ||
      (route.backend == GQABackend::Flash &&
       route.preparation.preprocess_mode == GQAPreprocessMode::Flash &&
       route.preparation.use_flash_attention_fast_decode == route.flash.fast_decode) ||
      (route.backend == GQABackend::MemoryEfficient &&
       route.preparation.preprocess_mode == GQAPreprocessMode::MemoryEfficient) ||
      (route.backend == GQABackend::Unfused &&
       route.preparation.preprocess_mode == GQAPreprocessMode::Unfused);
  if (!route_matches_preparation) {
    result.status = Invalid(
        "The selected GQA backend and preparation route are inconsistent.");
    return result;
  }

  const auto preparation = GetGQAPreparationRecipe(problem, route.preparation);
  result.status = preparation.status;
  if (!result.status.IsOK()) return result;

  GQACompleteWorkspaceRecipe recipe;
  recipe.backend = route.backend;
  recipe.preparation = preparation.recipe;
  switch (route.backend) {
    case GQABackend::Xqa: {
      const auto backend = GetGQAXqaWorkspaceRecipe(problem, route.xqa);
      result.status = backend.status;
      if (!result.status.IsOK()) return result;
      recipe.xqa = backend.recipe;
      recipe.backend_bytes = backend.recipe.total_backend_bytes;
      break;
    }
    case GQABackend::Flash: {
      auto flash_config = route.flash;
      flash_config.total_sequence_length = GetGQAEffectiveWorkspaceKvLength(
          route.flash.total_sequence_length,
          preparation.recipe.effective_kv_cache_capacity,
          problem.is_windowed_kv_cache);
      const auto backend = GetGQAFlashWorkspaceRecipe(problem, flash_config);
      result.status = backend.status;
      if (!result.status.IsOK()) return result;
      recipe.flash = backend.recipe;
      recipe.backend_bytes = backend.recipe.total_backend_bytes;
      break;
    }
    case GQABackend::MemoryEfficient: {
      const auto backend = GetGQAMemoryEfficientWorkspaceRecipe(
          problem, preparation.recipe.effective_kv_cache_capacity);
      result.status = backend.status;
      if (!result.status.IsOK()) return result;
      recipe.memory_efficient = backend.recipe;
      recipe.backend_bytes = backend.recipe.total_backend_bytes;
      break;
    }
    case GQABackend::Unfused: {
      const int64_t effective_kv_length = GetGQAEffectiveWorkspaceKvLength(
          route.unfused.total_sequence_length,
          preparation.recipe.effective_kv_cache_capacity,
          problem.is_windowed_kv_cache);
      const auto backend = GetGQAUnfusedWorkspaceRecipe(
          problem, effective_kv_length);
      result.status = backend.status;
      if (!result.status.IsOK()) return result;
      recipe.unfused = backend.recipe;
      recipe.backend_bytes = backend.recipe.total_backend_bytes;
      break;
    }
    case GQABackend::Cudnn:
      result.status = Unavailable("The selected GQA backend has no graph-free recipe.");
      return result;
    default:
      result.status = Invalid("The selected GQA backend is invalid.");
      return result;
  }

  if (recipe.backend_bytes == 0) {
    recipe.total_workspace_bytes = recipe.preparation.total_preparation_bytes;
  } else {
    result.status = CheckedGQAWorkspaceAlign(
        recipe.preparation.total_preparation_bytes,
        kGQAWorkspaceAlignment,
        recipe.backend_offset_bytes);
    if (!result.status.IsOK()) return result;
    result.status = CheckedGQAWorkspaceAdd(
        recipe.backend_offset_bytes, recipe.backend_bytes, recipe.total_workspace_bytes);
    if (!result.status.IsOK()) return result;
  }

  result.status = ValidateGQACompleteWorkspaceRecipe(recipe);
  if (result.status.IsOK()) result.recipe = recipe;
  return result;
}

GQAWorkspaceStatus ValidateGQACompleteWorkspaceRecipe(
    const GQACompleteWorkspaceRecipe& recipe) noexcept {
  auto status = ValidateGQAPreparationRecipe(recipe.preparation);
  if (!status.IsOK()) return status;

  size_t selected_backend_bytes = 0;
  switch (recipe.backend) {
    case GQABackend::Xqa:
      if (!IsDefault(recipe.flash) || !IsDefault(recipe.memory_efficient) || !IsDefault(recipe.unfused)) {
        return Invalid("A complete XQA recipe exposes another backend recipe.");
      }
      if (recipe.preparation.uses_separate_past_buffer) {
        return Invalid("A complete XQA recipe cannot preserve a separate past buffer.");
      }
      status = ValidateGQAXqaWorkspaceRecipe(recipe.xqa);
      selected_backend_bytes = recipe.xqa.total_backend_bytes;
      break;
    case GQABackend::Flash:
      if (!IsDefault(recipe.xqa) || !IsDefault(recipe.memory_efficient) || !IsDefault(recipe.unfused)) {
        return Invalid("A complete Flash recipe exposes another backend recipe.");
      }
      if (recipe.flash.fast_decode &&
          recipe.preparation.uses_separate_past_buffer) {
        return Invalid("A complete Flash fast-decode recipe cannot preserve a separate past buffer.");
      }
      status = ValidateGQAFlashWorkspaceRecipe(recipe.flash);
      selected_backend_bytes = recipe.flash.total_backend_bytes;
      break;
    case GQABackend::MemoryEfficient:
      if (!IsDefault(recipe.xqa) || !IsDefault(recipe.flash) || !IsDefault(recipe.unfused)) {
        return Invalid("A complete MEA recipe exposes another backend recipe.");
      }
      if (recipe.memory_efficient.effective_kv_cache_capacity !=
          recipe.preparation.effective_kv_cache_capacity) {
        return Invalid("The complete MEA recipe capacity does not match preparation.");
      }
      status = ValidateGQAMemoryEfficientWorkspaceRecipe(recipe.memory_efficient);
      selected_backend_bytes = recipe.memory_efficient.total_backend_bytes;
      break;
    case GQABackend::Unfused:
      if (!IsDefault(recipe.xqa) || !IsDefault(recipe.flash) || !IsDefault(recipe.memory_efficient)) {
        return Invalid("A complete unfused recipe exposes another backend recipe.");
      }
      status = ValidateGQAUnfusedWorkspaceRecipe(recipe.unfused);
      selected_backend_bytes = recipe.unfused.total_backend_bytes;
      break;
    case GQABackend::Cudnn:
      return Unavailable("A complete cuDNN GQA workspace recipe is unavailable.");
    default:
      return Invalid("The complete GQA backend is invalid.");
  }
  if (!status.IsOK()) return status;
  if (recipe.preparation.sequence_length_vector_count == 0 &&
      (recipe.backend != GQABackend::Flash || !recipe.flash.fast_decode)) {
    return Invalid("Only Flash fast decode may omit GQA sequence vectors.");
  }
  if (selected_backend_bytes != recipe.backend_bytes) {
    return Invalid("The complete GQA backend byte count does not match its selected recipe.");
  }

  if (recipe.backend_bytes == 0) {
    return recipe.backend_offset_bytes == 0 &&
                   recipe.total_workspace_bytes == recipe.preparation.total_preparation_bytes
               ? Ok()
               : Invalid("An empty GQA backend has inconsistent root offsets.");
  }
  if (recipe.backend_offset_bytes % kGQAWorkspaceAlignment != 0 ||
      recipe.backend_offset_bytes < recipe.preparation.total_preparation_bytes) {
    return Invalid("The complete GQA backend root is misaligned or overlaps preparation.");
  }
  size_t expected_total = 0;
  status = CheckedGQAWorkspaceAdd(
      recipe.backend_offset_bytes, recipe.backend_bytes, expected_total);
  if (!status.IsOK()) return status;
  return expected_total == recipe.total_workspace_bytes
             ? Ok()
             : Invalid("The complete GQA total does not match its terminal backend region.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
