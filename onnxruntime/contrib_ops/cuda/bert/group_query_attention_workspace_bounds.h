// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/framework/kernel_dispatch_policy.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

enum class GQAReachableBackend : uint32_t {
  None = 0,
  Xqa = 1u << 0,
  Flash = 1u << 1,
  FlashFastDecode = 1u << 2,
  MemoryEfficient = 1u << 3,
  Unfused = 1u << 4,
  Cudnn = 1u << 5,
};

enum class GQAExecutionPhase : uint32_t {
  Prefill,
  Decode,
  ChunkedPrefill,
};

constexpr const char* GQAExecutionPhaseName(
    GQAExecutionPhase phase) noexcept {
  switch (phase) {
    case GQAExecutionPhase::Prefill:
      return "prefill";
    case GQAExecutionPhase::Decode:
      return "decode";
    case GQAExecutionPhase::ChunkedPrefill:
      return "chunked prefill";
  }
  return "unknown";
}

constexpr GQAExecutionPhase ClassifyGQAExecutionPhase(
    bool is_first_prompt,
    int64_t sequence_length) noexcept {
  return is_first_prompt
             ? GQAExecutionPhase::Prefill
             : (sequence_length == 1
                    ? GQAExecutionPhase::Decode
                    : GQAExecutionPhase::ChunkedPrefill);
}

constexpr GQAReachableBackend operator|(GQAReachableBackend left,
                                        GQAReachableBackend right) noexcept {
  return static_cast<GQAReachableBackend>(
      static_cast<uint32_t>(left) | static_cast<uint32_t>(right));
}

constexpr GQAReachableBackend IntersectGQAReachableBackends(
    GQAReachableBackend left, GQAReachableBackend right) noexcept {
  return static_cast<GQAReachableBackend>(
      static_cast<uint32_t>(left) & static_cast<uint32_t>(right));
}

constexpr bool HasGQAReachableBackend(GQAReachableBackend mask,
                                      GQAReachableBackend backend) noexcept {
  return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(backend)) != 0;
}

constexpr GQAReachableBackend kAllGQAReachableBackends =
    GQAReachableBackend::Xqa |
    GQAReachableBackend::Flash |
    GQAReachableBackend::FlashFastDecode |
    GQAReachableBackend::MemoryEfficient |
    GQAReachableBackend::Unfused |
    GQAReachableBackend::Cudnn;

constexpr GQAReachableBackend kAnalyticallyBoundedGQABackends =
    GQAReachableBackend::Xqa |
    GQAReachableBackend::Flash |
    GQAReachableBackend::FlashFastDecode |
    GQAReachableBackend::MemoryEfficient |
    GQAReachableBackend::Unfused;

constexpr GQAReachableBackend kGQAGeneralPhaseBackends =
    GQAReachableBackend::Flash |
    GQAReachableBackend::MemoryEfficient |
    GQAReachableBackend::Unfused |
    GQAReachableBackend::Cudnn;

constexpr GQAReachableBackend GetGQAReachableBackendsForPhase(
    GQAReachableBackend reachable_backends,
    GQAExecutionPhase phase) noexcept {
  return phase == GQAExecutionPhase::Decode
             ? reachable_backends
             : IntersectGQAReachableBackends(
                   reachable_backends, kGQAGeneralPhaseBackends);
}

struct GQACommittedBackends {
  GQAReachableBackend prefill = GQAReachableBackend::None;
  GQAReachableBackend decode = GQAReachableBackend::None;
  GQAReachableBackend chunked_prefill = GQAReachableBackend::None;

  constexpr GQAReachableBackend Get(
      GQAExecutionPhase phase) const noexcept {
    switch (phase) {
      case GQAExecutionPhase::Prefill:
        return prefill;
      case GQAExecutionPhase::Decode:
        return decode;
      case GQAExecutionPhase::ChunkedPrefill:
        return chunked_prefill;
    }
    return GQAReachableBackend::None;
  }

  void Set(GQAExecutionPhase phase,
           GQAReachableBackend backends) noexcept {
    switch (phase) {
      case GQAExecutionPhase::Prefill:
        prefill = backends;
        return;
      case GQAExecutionPhase::Decode:
        decode = backends;
        return;
      case GQAExecutionPhase::ChunkedPrefill:
        chunked_prefill = backends;
        return;
    }
  }

  constexpr GQAReachableBackend Union() const noexcept {
    return prefill | decode | chunked_prefill;
  }
};

constexpr bool operator==(const GQACommittedBackends& left,
                          const GQACommittedBackends& right) noexcept {
  return left.prefill == right.prefill &&
         left.decode == right.decode &&
         left.chunked_prefill == right.chunked_prefill;
}

constexpr bool operator!=(const GQACommittedBackends& left,
                          const GQACommittedBackends& right) noexcept {
  return !(left == right);
}

// Graph-free upper bounds and immutable feature facts. Positive dimensions are
// componentwise bounds, not a claim that their combination is one runtime shape.
// The aggregate assumes both past/present K/V pairs alias unless
// account_partial_alias_preservation is enabled, in which case it adds the
// worst-case preservation scratch for partial aliasing. For a non-windowed
// cache, present_kv_cache_capacity_bound must bound both the allocated past
// capacity and the active total KV length.
struct GQAWorkspaceBounds {
  size_t qkv_element_size = 0;
  size_t cache_element_size = 0;
  int64_t batch_size_bound = 0;
  int64_t sequence_length_bound = 0;
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  int64_t head_size_bound = 0;
  int64_t present_kv_cache_capacity_bound = 0;
  int64_t kv_cache_bit_width = 0;
  GQAKvQuantizationType k_quantization = GQAKvQuantizationType::None;
  GQAKvQuantizationType v_quantization = GQAKvQuantizationType::None;
  bool is_windowed_kv_cache = false;
  // Non-windowed GQA permits exactly one past/present K/V pair to alias, which
  // makes the runtime copy the full past cache into preservation scratch that
  // coexists with the selected backend workspace. Windowed GQA rejects
  // non-shared buffers, so both pairs always alias and no copy occurs. When
  // true, the aggregate adds the worst-case preservation copy on top of the
  // per-route maximum. See group_query_attention.cc (separate_past_buffer).
  bool account_partial_alias_preservation = false;
  bool do_rotary = false;
  bool is_packed_qkv = false;
  bool use_qk_norm = false;
  bool prefill_reachable = false;
  bool decode_reachable = false;
  bool chunked_prefill_reachable = false;

  GQAReachableBackend reachable_backends = GQAReachableBackend::None;
  int64_t device_major = 0;
  int64_t device_minor = 0;
  int64_t multi_processor_count = 0;
  bool is_bf16 = false;
  GQAXqaKvType xqa_kv_type = GQAXqaKvType::None;
  // Exact state controls per-run XQA conversion scratch and implies prepack
  // lifetime accounting. Possible state extends that charge to Level 1.
  GQAXqaHeadSinkStorage xqa_head_sink_storage = GQAXqaHeadSinkStorage::None;
  bool head_sink_may_be_prepacked = false;
  int64_t local_window_size = -1;
};

constexpr bool IsGQAPhaseAdmitted(
    const GQAWorkspaceBounds& bounds,
    GQAExecutionPhase phase) noexcept {
  switch (phase) {
    case GQAExecutionPhase::Prefill:
      return bounds.prefill_reachable;
    case GQAExecutionPhase::Decode:
      return bounds.decode_reachable;
    case GQAExecutionPhase::ChunkedPrefill:
      return bounds.chunked_prefill_reachable;
  }
  return false;
}

struct GQAWorkspaceAggregate {
  GQAWorkspaceStatus status;
  size_t total_workspace_bytes = 0;
  size_t persistent_prepack_bytes = 0;
  size_t initialization_scratch_bytes = 0;
  GQAReachableBackend sized_backends = GQAReachableBackend::None;
  GQACommittedBackends committed_backends;
};

// Resolves the intent to runtime-enforced phase-specific route sets and computes
// the maximum over admitted phases and their permitted routes. Auto and Latency
// preserve the phase-applicable route cascade. Memory chooses the smallest proven
// route independently for prefill, decode, and chunked prefill; decode also retains
// the selected general-phase routes as bounded fallbacks. Safe commits every
// analytically bounded route applicable to each phase. A committed cuDNN route
// makes the result unavailable because its allocator-based workspace has no
// graph-free oracle.
GQAWorkspaceAggregate GetGQAWorkspaceAggregateForBounds(
    const GQAWorkspaceBounds& bounds,
    KernelDispatchPolicy policy = KernelDispatchPolicy::Auto) noexcept;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
