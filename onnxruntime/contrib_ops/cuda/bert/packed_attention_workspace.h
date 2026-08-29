// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kPackedAttentionWorkspaceAlignment = 256;

enum class PackedAttentionBackend {
  Trt,
  Flash,
  MemoryEfficient,
  Unfused,
};

enum class PackedAttentionBackendMask : uint32_t {
  None = 0,
  Trt = 1U << 0,
  Flash = 1U << 1,
  MemoryEfficient = 1U << 2,
  Unfused = 1U << 3,
};

constexpr PackedAttentionBackendMask operator|(PackedAttentionBackendMask left,
                                               PackedAttentionBackendMask right) noexcept {
  return static_cast<PackedAttentionBackendMask>(
      static_cast<uint32_t>(left) | static_cast<uint32_t>(right));
}

constexpr bool HasPackedAttentionBackend(PackedAttentionBackendMask mask,
                                         PackedAttentionBackend backend) noexcept {
  PackedAttentionBackendMask flag = PackedAttentionBackendMask::None;
  switch (backend) {
    case PackedAttentionBackend::Trt:
      flag = PackedAttentionBackendMask::Trt;
      break;
    case PackedAttentionBackend::Flash:
      flag = PackedAttentionBackendMask::Flash;
      break;
    case PackedAttentionBackend::MemoryEfficient:
      flag = PackedAttentionBackendMask::MemoryEfficient;
      break;
    case PackedAttentionBackend::Unfused:
      flag = PackedAttentionBackendMask::Unfused;
      break;
    default:
      return false;
  }

  return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(flag)) != 0;
}

enum class PackedAttentionWorkspaceError {
  None,
  InvalidArgument,
  Overflow,
};

enum class PackedAttentionQkvWorkspaceLayout {
  None,
  Planar,
  InterleavedTn3h,
};

// Width, in scalar elements, of one element indexed by the QKV materialization
// producer. This is a plain host-side description and does not depend on CUDA types.
enum class PackedAttentionQkvMaterializationIndexWidth : int32_t {
  Scalar = 1,
  Vector2 = 2,
  Vector4 = 4,
};

constexpr PackedAttentionQkvMaterializationIndexWidth GetPackedAttentionQkvMaterializationIndexWidth(
    int32_t qk_head_size, int32_t v_head_size) noexcept {
  if (qk_head_size % 4 == 0 && v_head_size % 4 == 0) {
    return PackedAttentionQkvMaterializationIndexWidth::Vector4;
  }

  if (qk_head_size % 2 == 0 && v_head_size % 2 == 0) {
    return PackedAttentionQkvMaterializationIndexWidth::Vector2;
  }

  return PackedAttentionQkvMaterializationIndexWidth::Scalar;
}

struct PackedAttentionWorkspaceStatus {
  PackedAttentionWorkspaceError error = PackedAttentionWorkspaceError::None;
  const char* message = "";

  constexpr bool IsOK() const noexcept {
    return error == PackedAttentionWorkspaceError::None;
  }
};

// Packed operator inputs have rank at most four. A rank greater than four is retained
// in rank and rejected without reading beyond dimensions.
struct PackedAttentionShape {
  std::array<int64_t, 4> dimensions{};
  size_t rank = 0;
};

struct PackedAttentionInputShapes {
  PackedAttentionShape input;
  PackedAttentionShape weights;
  PackedAttentionShape bias;
  PackedAttentionShape token_offset;
  PackedAttentionShape cumulative_sequence_length;
  PackedAttentionShape attention_bias;
  size_t element_size = 0;
  int64_t num_heads = 0;
  size_t qkv_hidden_sizes_count = 0;
  std::array<int64_t, 3> qkv_hidden_sizes{};
  bool has_attention_bias = false;
};

enum class PackedMultiHeadAttentionQkvFormat {
  Packed,
  Separate,
};

struct PackedMultiHeadAttentionInputShapes {
  PackedAttentionShape query;
  PackedAttentionShape key;
  PackedAttentionShape value;
  PackedAttentionShape bias;
  PackedAttentionShape token_offset;
  PackedAttentionShape cumulative_sequence_length;
  PackedAttentionShape attention_bias;
  size_t element_size = 0;
  int64_t num_heads = 0;
  bool has_key = false;
  bool has_value = false;
  bool has_bias = false;
  bool has_attention_bias = false;
};

struct PackedAttentionProblem {
  size_t element_size = 0;
  int32_t token_count = 0;
  int32_t batch_size = 0;
  int32_t sequence_length = 0;
  int32_t num_heads = 0;
  int32_t input_hidden_size = 0;
  int32_t hidden_size = 0;
  int32_t v_hidden_size = 0;
  int32_t qk_head_size = 0;
  int32_t v_head_size = 0;
  bool has_attention_bias = false;
  bool broadcast_attn_bias_dim_0 = false;
  bool broadcast_attn_bias_dim_1 = false;
  PackedAttentionBackend backend = PackedAttentionBackend::Unfused;
  bool trt_runner_available = false;
  PackedAttentionQkvMaterializationIndexWidth qkv_materialization_index_width =
      PackedAttentionQkvMaterializationIndexWidth::Scalar;
};

struct PackedMultiHeadAttentionProblem {
  size_t element_size = 0;
  int32_t token_count = 0;
  int32_t batch_size = 0;
  int32_t sequence_length = 0;
  int32_t num_heads = 0;
  int32_t hidden_size = 0;
  int32_t v_hidden_size = 0;
  int32_t qk_head_size = 0;
  int32_t v_head_size = 0;
  PackedMultiHeadAttentionQkvFormat qkv_format = PackedMultiHeadAttentionQkvFormat::Packed;
  bool has_bias = false;
  bool has_attention_bias = false;
  bool broadcast_attn_bias_dim_0 = false;
  bool broadcast_attn_bias_dim_1 = false;
  PackedAttentionBackend backend = PackedAttentionBackend::Unfused;
  bool trt_runner_available = false;
  PackedAttentionQkvMaterializationIndexWidth qkv_materialization_index_width =
      PackedAttentionQkvMaterializationIndexWidth::Scalar;
};

template <typename T>
struct PackedAttentionProblemResult {
  PackedAttentionWorkspaceStatus status;
  T problem;
};

struct PackedAttentionWorkspaceRecipe {
  // PackedAttention owns the projection allocation. PackedMultiHeadAttention leaves
  // these fields zero because its Q/K/V inputs are already projected.
  size_t projection_bytes = 0;
  size_t attention_workspace_bytes = 0;

  int32_t projection_m = 0;
  int32_t projection_n = 0;
  int32_t projection_k = 0;

  // qkv_capacity_bytes preserves the legacy B*S allocation size. It can be
  // larger than the route's materialized T-token view.
  bool no_qkv_workspace = false;
  size_t qkv_capacity_bytes = 0;

  // Planar fields are valid only when qkv_layout is Planar. Q starts at byte
  // zero and K has the same byte size as Q.
  PackedAttentionQkvWorkspaceLayout qkv_layout = PackedAttentionQkvWorkspaceLayout::None;
  size_t q_offset_bytes = 0;
  size_t q_bytes = 0;
  size_t k_offset_bytes = 0;
  size_t k_bytes = 0;
  size_t v_offset_bytes = 0;
  size_t v_bytes = 0;

  // The TRT materialization producer writes [T, N, 3, H] here. Planar Q/K/V
  // offsets are unavailable for this layout. These fields are zero otherwise.
  size_t interleaved_qkv_offset_bytes = 0;
  size_t interleaved_qkv_bytes = 0;

  // Backend fields are conditional: Flash uses an LSE buffer, MEA may use an
  // FP32 accumulator, and unfused uses two equally-sized aligned scratch
  // regions. has_second_scratch distinguishes a zero-sized unfused scratch
  // region from routes that have no second scratch region.
  size_t backend_workspace_offset_bytes = 0;
  size_t backend_workspace_bytes = 0;
  bool has_second_scratch = false;
  size_t second_scratch_offset_bytes = 0;
};

struct PackedAttentionWorkspaceResult {
  PackedAttentionWorkspaceStatus status;
  PackedAttentionWorkspaceRecipe recipe;
};

// AOT routes are mutually exclusive. The attention component is therefore the
// maximum single-route recipe, while PA's projection is simultaneously live
// with that component.
struct PackedAttentionWorkspaceAggregate {
  PackedAttentionWorkspaceStatus status;
  size_t projection_bytes = 0;
  size_t attention_workspace_bytes = 0;
  size_t total_workspace_bytes = 0;
};

PackedAttentionWorkspaceStatus CheckedPackedAttentionAdd(size_t left, size_t right, size_t& result) noexcept;

PackedAttentionWorkspaceStatus CheckedPackedAttentionMultiply(size_t left, size_t right,
                                                              size_t& result) noexcept;

PackedAttentionWorkspaceStatus CheckedPackedAttentionAlign(size_t value, size_t alignment,
                                                           size_t& result) noexcept;

// These builders validate input shapes and host-visible geometry only. They do
// not inspect or validate token_offset or cumulative_sequence_length values.
PackedAttentionProblemResult<PackedAttentionProblem> BuildPackedAttentionProblem(
    const PackedAttentionInputShapes& inputs) noexcept;

PackedAttentionProblemResult<PackedMultiHeadAttentionProblem> BuildPackedMultiHeadAttentionProblem(
    const PackedMultiHeadAttentionInputShapes& inputs) noexcept;

PackedAttentionWorkspaceResult GetPackedAttentionWorkspaceRecipe(
    const PackedAttentionProblem& problem) noexcept;

PackedAttentionWorkspaceResult GetPackedMultiHeadAttentionWorkspaceRecipe(
    const PackedMultiHeadAttentionProblem& problem) noexcept;

PackedAttentionWorkspaceAggregate GetPackedAttentionWorkspaceAggregate(
    const PackedAttentionProblem& problem,
    PackedAttentionBackendMask feasible_backends) noexcept;

PackedAttentionWorkspaceAggregate GetPackedMultiHeadAttentionWorkspaceAggregate(
    const PackedMultiHeadAttentionProblem& problem,
    PackedAttentionBackendMask feasible_backends) noexcept;

PackedAttentionWorkspaceStatus ValidatePackedAttentionWorkspaceRecipe(
    const PackedAttentionWorkspaceRecipe& recipe) noexcept;

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
