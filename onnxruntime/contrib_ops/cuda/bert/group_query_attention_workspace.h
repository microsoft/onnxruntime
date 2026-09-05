// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kGQAPreparationAlignment = 256;

enum class GQAWorkspaceError {
  None,
  InvalidArgument,
  Overflow,
};

struct GQAWorkspaceStatus {
  GQAWorkspaceError error = GQAWorkspaceError::None;
  const char* message = "";

  constexpr bool IsOK() const noexcept {
    return error == GQAWorkspaceError::None;
  }
};

enum class GQAKvQuantizationType {
  None,
  PerTensor,
  PerChannel,
};

// This mode describes only the QKV preprocessing behavior selected by the runtime route.
// It does not describe or size the backend's internal attention scratch.
enum class GQAPreprocessMode {
  Xqa,
  Flash,
  MemoryEfficient,
  Fallback,
};

// Plain-scalar runtime geometry and feature facts used by preparation allocations.
// present_kv_cache_capacity is the original GroupQueryAttentionParameters::seqlen_present_kv_cache
// before windowed staging mutates the runtime parameters. It is not the window-only
// GroupQueryAttentionParameters::kv_cache_capacity field.
struct GQAWorkspaceProblem {
  size_t qkv_element_size = 0;
  size_t cache_element_size = 0;
  int64_t batch_size = 0;
  int64_t sequence_length = 0;
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  int64_t head_size = 0;
  int64_t present_kv_cache_capacity = 0;
  int64_t kv_cache_bit_width = 0;
  GQAKvQuantizationType k_quantization = GQAKvQuantizationType::None;
  GQAKvQuantizationType v_quantization = GQAKvQuantizationType::None;
  bool is_windowed_kv_cache = false;
  bool is_first_prompt = false;
  bool do_rotary = false;
  bool is_packed_qkv = false;
  bool use_qk_norm = false;
};

struct GQAPreparationRoute {
  GQAPreprocessMode preprocess_mode = GQAPreprocessMode::Fallback;
  bool use_flash_attention_fast_decode = false;
};

// This recipe covers only simultaneously-live preparation/cross-cutting transients.
// Outputs, present KV tensors, constructor zeros_, prepacked xqa_head_sink_, and
// backend-internal scratch are deliberately excluded. In particular, XQA backend-internal
// and extra scratch is composed later; this recipe includes only the current
// GQABufferRequirements QKV preprocess term for an already-selected XQA route. Staged
// caches remain included because they are transient operator-owned preparation storage
// despite their KV shape.
//
// All non-empty top-level regions start at a 256-byte-aligned offset. Staging and
// compaction are mutually exclusive. Compaction is one allocation whose key and
// value halves are represented by absolute subregion offsets.
struct GQAPreparationRecipe {
  // The original present-cache capacity, except windowed multi-token staging uses C + S.
  int64_t effective_kv_cache_capacity = 0;
  size_t cache_row_bytes = 0;

  bool uses_staging = false;
  size_t staged_key_offset_bytes = 0;
  size_t staged_key_bytes = 0;
  size_t staged_value_offset_bytes = 0;
  size_t staged_value_bytes = 0;

  bool uses_compaction = false;
  size_t compaction_offset_bytes = 0;
  size_t compaction_bytes = 0;
  size_t compaction_key_offset_bytes = 0;
  size_t compaction_key_bytes = 0;
  size_t compaction_value_offset_bytes = 0;
  size_t compaction_value_bytes = 0;

  // Three vectors hold past, total, and padded sequence lengths. Windowed preparation
  // adds cache-relative past/total lengths and eviction counts, for six vectors total.
  size_t sequence_length_vector_count = 0;
  size_t sequence_lengths_offset_bytes = 0;
  size_t sequence_lengths_bytes = 0;

  size_t qkv_preprocess_offset_bytes = 0;
  size_t qkv_preprocess_bytes = 0;

  size_t total_preparation_bytes = 0;
};

struct GQAPreparationResult {
  GQAWorkspaceStatus status;
  GQAPreparationRecipe recipe;
};

GQAWorkspaceStatus CheckedGQAWorkspaceAdd(size_t left, size_t right, size_t& result) noexcept;

GQAWorkspaceStatus CheckedGQAWorkspaceMultiply(size_t left, size_t right, size_t& result) noexcept;

GQAWorkspaceStatus CheckedGQAWorkspaceAlign(size_t value, size_t alignment, size_t& result) noexcept;

GQAPreparationResult GetGQAPreparationRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAPreparationRoute& route) noexcept;

GQAWorkspaceStatus ValidateGQAPreparationRecipe(const GQAPreparationRecipe& recipe) noexcept;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
