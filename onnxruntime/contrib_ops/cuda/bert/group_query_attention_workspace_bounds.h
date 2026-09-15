// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

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

constexpr GQAReachableBackend operator|(GQAReachableBackend left,
                                        GQAReachableBackend right) noexcept {
  return static_cast<GQAReachableBackend>(
      static_cast<uint32_t>(left) | static_cast<uint32_t>(right));
}

constexpr bool HasGQAReachableBackend(GQAReachableBackend mask,
                                      GQAReachableBackend backend) noexcept {
  return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(backend)) != 0;
}

// Graph-free upper bounds and immutable feature facts. Positive dimensions are
// componentwise bounds, not a claim that their combination is one runtime shape.
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
  bool do_rotary = false;
  bool is_packed_qkv = false;
  bool use_qk_norm = false;
  bool prompt_reachable = false;
  bool decode_reachable = false;

  GQAReachableBackend reachable_backends = GQAReachableBackend::None;
  int64_t device_major = 0;
  int64_t device_minor = 0;
  int64_t multi_processor_count = 0;
  bool is_bf16 = false;
  GQAXqaKvType xqa_kv_type = GQAXqaKvType::None;
  GQAXqaHeadSinkStorage xqa_head_sink_storage = GQAXqaHeadSinkStorage::None;
  int64_t local_window_size = -1;
};

struct GQAWorkspaceAggregate {
  GQAWorkspaceStatus status;
  size_t total_workspace_bytes = 0;
  GQAReachableBackend sized_backends = GQAReachableBackend::None;
};

// Computes a sound maximum over all complete routes reachable anywhere in the
// bounded domain. cuDNN reachability makes the result unavailable because its
// allocator-based workspace has no graph-free oracle.
GQAWorkspaceAggregate GetGQAWorkspaceAggregateForBounds(
    const GQAWorkspaceBounds& bounds) noexcept;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
