// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kGQAWorkspaceAlignment = 256;

enum class GQAWorkspaceError {
  None,
  InvalidArgument,
  Overflow,
  Unavailable,
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
  Unfused,
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
  GQAPreprocessMode preprocess_mode = GQAPreprocessMode::Unfused;
  // This is the authoritative selected runtime fact. Complete-route
  // composition validates that the Flash backend config carries the same value.
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

enum class GQAXqaKvType {
  None,
  Int8,
  Fp8,
};

enum class GQAXqaHeadSinkStorage {
  None,
  PrepackedFp32,
  DynamicConversion,
};

struct GQAXqaConfig {
  int64_t device_major = 0;
  int64_t device_minor = 0;
  int64_t multi_processor_count = 0;
  GQAXqaKvType kv_type = GQAXqaKvType::None;
  GQAXqaHeadSinkStorage head_sink_storage = GQAXqaHeadSinkStorage::None;
  bool is_bf16 = false;
};

// XQA owns one runtime allocation. The semaphore and three internal scratch views
// reproduce GetXQAScratchSize. Optional RoPE and dynamic-head-sink views follow it
// in the same allocation. Offsets are relative to the start of that allocation.
struct GQAXqaWorkspaceRecipe {
  size_t sequence_count = 0;
  size_t subsequences_per_sequence = 0;
  size_t subsequence_count = 0;
  size_t m_tile_size = 0;

  size_t semaphore_offset_bytes = 0;
  size_t semaphore_bytes = 0;
  size_t semaphore_aligned_bytes = 0;
  size_t row_max_offset_bytes = 0;
  size_t row_max_bytes = 0;
  size_t row_sum_offset_bytes = 0;
  size_t row_sum_bytes = 0;
  size_t output_accumulator_offset_bytes = 0;
  size_t output_accumulator_bytes = 0;
  size_t internal_scratch_bytes = 0;

  size_t rotary_q_offset_bytes = 0;
  size_t rotary_q_bytes = 0;
  size_t rotary_k_offset_bytes = 0;
  size_t rotary_k_bytes = 0;
  size_t dynamic_head_sink_offset_bytes = 0;
  size_t dynamic_head_sink_bytes = 0;

  size_t total_backend_bytes = 0;
};

struct GQAXqaWorkspaceResult {
  GQAWorkspaceStatus status;
  GQAXqaWorkspaceRecipe recipe;
};

struct GQAFlashConfig {
  int64_t total_sequence_length = 0;
  int64_t local_window_size = -1;
  int64_t multi_processor_count = 0;
  // Standalone Flash recipes need this selected runtime fact. When composed
  // into a complete route, it must equal the authoritative preparation value.
  bool fast_decode = false;
};

// Flash owns up to three separate runtime allocations. Their offsets describe a
// 256-byte-aligned concrete root; byte counts remain the exact allocation requests.
struct GQAFlashWorkspaceRecipe {
  size_t split_heuristic_head_count = 0;
  size_t split_heuristic_kv_length = 0;
  size_t selected_split_count = 1;
  // The Flash API encodes the no-split case as zero.
  size_t runtime_num_splits = 0;
  size_t rounded_head_size = 0;

  size_t softmax_lse_offset_bytes = 0;
  size_t softmax_lse_bytes = 0;
  size_t softmax_lse_accumulator_offset_bytes = 0;
  size_t softmax_lse_accumulator_bytes = 0;
  size_t output_accumulator_offset_bytes = 0;
  size_t output_accumulator_bytes = 0;
  size_t total_backend_bytes = 0;
};

struct GQAFlashWorkspaceResult {
  GQAWorkspaceStatus status;
  GQAFlashWorkspaceRecipe recipe;
};

// MEA owns separate K expansion, V expansion, and optional FP32 output
// accumulator allocations. effective_kv_cache_capacity is the post-staging
// capacity from GQAPreparationRecipe.
struct GQAMemoryEfficientWorkspaceRecipe {
  int64_t effective_kv_cache_capacity = 0;
  size_t expanded_key_offset_bytes = 0;
  size_t expanded_key_bytes = 0;
  size_t expanded_value_offset_bytes = 0;
  size_t expanded_value_bytes = 0;
  size_t output_accumulator_offset_bytes = 0;
  size_t output_accumulator_bytes = 0;
  size_t total_backend_bytes = 0;
};

struct GQAMemoryEfficientWorkspaceResult {
  GQAWorkspaceStatus status;
  GQAMemoryEfficientWorkspaceRecipe recipe;
};

// The unfused backend owns one combined runtime allocation. Q, Y, FP32 QK, and
// the upper-bound softmax view are all represented explicitly.
struct GQAUnfusedWorkspaceRecipe {
  size_t q_bnsh_offset_bytes = 0;
  size_t q_bnsh_bytes = 0;
  size_t y_bnsh_offset_bytes = 0;
  size_t y_bnsh_bytes = 0;
  size_t qk_offset_bytes = 0;
  size_t qk_bytes = 0;
  size_t softmax_offset_bytes = 0;
  size_t softmax_bytes = 0;
  size_t total_backend_bytes = 0;
};

struct GQAUnfusedWorkspaceResult {
  GQAWorkspaceStatus status;
  GQAUnfusedWorkspaceRecipe recipe;
};

struct GQAUnfusedConfig {
  int64_t total_sequence_length = 0;
};

enum class GQABackend {
  Xqa,
  Flash,
  MemoryEfficient,
  Unfused,
  Cudnn,
};

struct GQAConcreteRoute {
  GQABackend backend = GQABackend::Unfused;
  GQAPreparationRoute preparation;
  // Only the config matching backend is populated.
  GQAXqaConfig xqa;
  GQAFlashConfig flash;
  GQAUnfusedConfig unfused;
};

// A complete recipe composes exactly one preparation recipe and one selected
// backend recipe. Backend recipe offsets remain relative to backend_offset_bytes;
// adding that checked, 256-byte-aligned base gives root-relative offsets. Routes
// are concrete and mutually exclusive: this type does not aggregate reachability.
struct GQACompleteWorkspaceRecipe {
  GQABackend backend = GQABackend::Unfused;
  GQAPreparationRecipe preparation;
  size_t backend_offset_bytes = 0;
  size_t backend_bytes = 0;
  // Only the recipe matching backend is populated; all others remain zero.
  GQAXqaWorkspaceRecipe xqa;
  GQAFlashWorkspaceRecipe flash;
  GQAMemoryEfficientWorkspaceRecipe memory_efficient;
  GQAUnfusedWorkspaceRecipe unfused;
  size_t total_workspace_bytes = 0;
};

struct GQACompleteWorkspaceResult {
  GQAWorkspaceStatus status;
  GQACompleteWorkspaceRecipe recipe;
};

GQAWorkspaceStatus CheckedGQAWorkspaceAdd(size_t left, size_t right, size_t& result) noexcept;

GQAWorkspaceStatus CheckedGQAWorkspaceMultiply(size_t left, size_t right, size_t& result) noexcept;

GQAWorkspaceStatus CheckedGQAWorkspaceAlign(size_t value, size_t alignment, size_t& result) noexcept;

GQAPreparationResult GetGQAPreparationRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAPreparationRoute& route) noexcept;

GQAWorkspaceStatus ValidateGQAPreparationRecipe(const GQAPreparationRecipe& recipe) noexcept;

GQAXqaWorkspaceResult GetGQAXqaWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAXqaConfig& config) noexcept;

GQAFlashWorkspaceResult GetGQAFlashWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAFlashConfig& config) noexcept;

GQAMemoryEfficientWorkspaceResult GetGQAMemoryEfficientWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    int64_t effective_kv_cache_capacity) noexcept;

GQAUnfusedWorkspaceResult GetGQAUnfusedWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    int64_t total_sequence_length) noexcept;

GQAWorkspaceStatus ValidateGQAXqaWorkspaceRecipe(
    const GQAXqaWorkspaceRecipe& recipe) noexcept;

GQAWorkspaceStatus ValidateGQAFlashWorkspaceRecipe(
    const GQAFlashWorkspaceRecipe& recipe) noexcept;

GQAWorkspaceStatus ValidateGQAMemoryEfficientWorkspaceRecipe(
    const GQAMemoryEfficientWorkspaceRecipe& recipe) noexcept;

GQAWorkspaceStatus ValidateGQAUnfusedWorkspaceRecipe(
    const GQAUnfusedWorkspaceRecipe& recipe) noexcept;

GQACompleteWorkspaceResult GetGQACompleteWorkspaceRecipe(
    const GQAWorkspaceProblem& problem,
    const GQAConcreteRoute& route) noexcept;

GQAWorkspaceStatus ValidateGQACompleteWorkspaceRecipe(
    const GQACompleteWorkspaceRecipe& recipe) noexcept;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
