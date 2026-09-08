// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/packed_attention_workspace.h"

#include <limits>

#include "core/common/safeint.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr PackedAttentionWorkspaceStatus Ok() noexcept {
  return {};
}

constexpr PackedAttentionWorkspaceStatus Invalid(const char* message) noexcept {
  return {PackedAttentionWorkspaceError::InvalidArgument, message};
}

constexpr PackedAttentionWorkspaceStatus Overflow(const char* message) noexcept {
  return {PackedAttentionWorkspaceError::Overflow, message};
}

PackedAttentionWorkspaceStatus ValidateDimension(int64_t value) noexcept {
  if (value < 0) {
    return Invalid("Packed attention dimensions must be non-negative.");
  }

  if (value > std::numeric_limits<int32_t>::max()) {
    return Invalid("Packed attention dimensions and attributes must fit the int32 CUDA ABI.");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus CheckedProductFitsInt32(size_t left, size_t right) noexcept {
  size_t product = 0;
  auto status = CheckedPackedAttentionMultiply(left, right, product);
  if (!status.IsOK()) {
    return status;
  }

  if (product > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return Invalid("A packed attention derived product does not fit the int32 CUDA ABI.");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus CheckedProductFitsInt32(size_t first, size_t second, size_t third) noexcept {
  size_t product = 0;
  auto status = CheckedPackedAttentionMultiply(first, second, product);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedProductFitsInt32(product, third);
}

PackedAttentionWorkspaceStatus CheckedProductFitsInt32(size_t first, size_t second, size_t third,
                                                       size_t fourth) noexcept {
  size_t product = 0;
  auto status = CheckedPackedAttentionMultiply(first, second, product);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionMultiply(product, third, product);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedProductFitsInt32(product, fourth);
}

PackedAttentionWorkspaceStatus ValidateElementSize(size_t element_size) noexcept {
  if (element_size != 2 && element_size != 4) {
    return Invalid("Packed attention element size must be 2 or 4 bytes.");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateShape(const PackedAttentionShape& shape, size_t expected_rank) noexcept {
  if (shape.rank != expected_rank || shape.rank > shape.dimensions.size()) {
    return Invalid("A packed attention input has an invalid rank.");
  }

  for (size_t i = 0; i < shape.rank; ++i) {
    auto status = ValidateDimension(shape.dimensions[i]);
    if (!status.IsOK()) {
      return status;
    }
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateAttentionBias(const PackedAttentionShape& shape,
                                                     int64_t batch_size,
                                                     int64_t num_heads,
                                                     int64_t sequence_length) noexcept {
  if (shape.rank != 4 || shape.rank > shape.dimensions.size()) {
    return Invalid("Attention bias must have rank 4.");
  }

  for (size_t i = 0; i < shape.rank; ++i) {
    auto status = ValidateDimension(shape.dimensions[i]);
    if (!status.IsOK()) {
      return status;
    }
  }

  if ((shape.dimensions[0] != 1 && shape.dimensions[0] != batch_size) ||
      (shape.dimensions[1] != 1 && shape.dimensions[1] != num_heads) ||
      shape.dimensions[2] != sequence_length ||
      shape.dimensions[3] != sequence_length) {
    return Invalid("Attention bias must have shape [B or 1, N or 1, S, S].");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateCoreGeometry(int32_t token_count,
                                                    int32_t batch_size,
                                                    int32_t sequence_length,
                                                    int32_t num_heads,
                                                    int32_t hidden_size,
                                                    int32_t v_hidden_size,
                                                    int32_t qk_head_size,
                                                    int32_t v_head_size) noexcept {
  if (token_count < 0 || batch_size < 0 || sequence_length < 0 ||
      num_heads <= 0 || hidden_size < 0 || v_hidden_size < 0 ||
      qk_head_size < 0 || v_head_size < 0) {
    return Invalid("Packed attention problem dimensions are invalid.");
  }

  const size_t t = static_cast<size_t>(token_count);
  const size_t b = static_cast<size_t>(batch_size);
  const size_t s = static_cast<size_t>(sequence_length);
  const size_t n = static_cast<size_t>(num_heads);
  const size_t h = static_cast<size_t>(qk_head_size);
  const size_t hv = static_cast<size_t>(v_head_size);

  size_t q_hidden = 0;
  auto status = CheckedPackedAttentionMultiply(n, h, q_hidden);
  if (!status.IsOK()) {
    return status;
  }

  size_t v_hidden = 0;
  status = CheckedPackedAttentionMultiply(n, hv, v_hidden);
  if (!status.IsOK()) {
    return status;
  }

  if (q_hidden != static_cast<size_t>(hidden_size) ||
      v_hidden != static_cast<size_t>(v_hidden_size)) {
    return Invalid("Packed attention hidden sizes do not match the head geometry.");
  }

  size_t padded_tokens = 0;
  status = CheckedPackedAttentionMultiply(b, s, padded_tokens);
  if (!status.IsOK()) {
    return status;
  }

  if (t > padded_tokens) {
    return Invalid("Packed attention token count T must not exceed B * S.");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateQkvMaterializationGeometry(int32_t token_count,
                                                                  int32_t num_heads,
                                                                  int32_t qk_head_size,
                                                                  int32_t v_head_size,
                                                                  PackedAttentionQkvMaterializationIndexWidth
                                                                      index_width) noexcept {
  const size_t t = static_cast<size_t>(token_count);
  const size_t n = static_cast<size_t>(num_heads);
  const int32_t width = static_cast<int32_t>(index_width);
  if (width != 1 && width != 2 && width != 4) {
    return Invalid("Packed attention QKV materialization index width is invalid.");
  }

  if (qk_head_size % width != 0 || v_head_size % width != 0) {
    return Invalid("Packed attention head geometry is not divisible by the QKV materialization index width.");
  }

  const size_t h = static_cast<size_t>(qk_head_size / width);
  const size_t hv = static_cast<size_t>(v_head_size / width);

  size_t qkv_head_size = 0;
  auto status = CheckedPackedAttentionAdd(h, h, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionAdd(qkv_head_size, hv, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  // The packed transpose producers use int32 offsets in scalar, T2, or T4
  // elements according to the producer selected by the graph adapter.
  status = CheckedProductFitsInt32(n, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedProductFitsInt32(t, n, qkv_head_size);
}

PackedAttentionWorkspaceStatus ValidateUnfusedGeometry(int32_t token_count,
                                                       int32_t batch_size,
                                                       int32_t sequence_length,
                                                       int32_t num_heads,
                                                       int32_t qk_head_size,
                                                       int32_t v_head_size,
                                                       PackedAttentionQkvMaterializationIndexWidth
                                                           index_width) noexcept {
  constexpr int32_t kMaxGridDimY = 65535;
  if (batch_size > kMaxGridDimY) {
    return Invalid("Packed unfused attention batch size exceeds CUDA gridDim.y.");
  }

  const size_t t = static_cast<size_t>(token_count);
  const size_t b = static_cast<size_t>(batch_size);
  const size_t s = static_cast<size_t>(sequence_length);
  const size_t n = static_cast<size_t>(num_heads);
  const size_t h = static_cast<size_t>(qk_head_size);
  const size_t hv = static_cast<size_t>(v_head_size);
  const int32_t width = static_cast<int32_t>(index_width);
  if ((width != 1 && width != 2 && width != 4) ||
      qk_head_size % width != 0 || v_head_size % width != 0) {
    return Invalid("Packed attention QKV materialization index width is invalid.");
  }

  const size_t producer_h = static_cast<size_t>(qk_head_size / width);
  const size_t producer_hv = static_cast<size_t>(v_head_size / width);

  auto status = CheckedProductFitsInt32(b, s);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(s, s);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(b, n);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(b, n, s);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(s, h);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(s, hv);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(b, n, s, producer_h);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(b, n, s, producer_hv);
  if (!status.IsOK()) {
    return status;
  }

  size_t qkv_head_size = 0;
  status = CheckedPackedAttentionAdd(producer_h, producer_h, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionAdd(qkv_head_size, producer_hv, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt32(t, n, qkv_head_size);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedProductFitsInt32(b, s, n, qkv_head_size);
}

PackedAttentionWorkspaceStatus CheckedProductFitsInt64(size_t first, size_t second,
                                                       size_t third) noexcept {
  size_t product = 0;
  auto status = CheckedPackedAttentionMultiply(first, second, product);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionMultiply(product, third, product);
  if (!status.IsOK()) {
    return status;
  }

  if (product > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return Invalid("A packed attention stride or extent does not fit int64.");
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateFusedGrid(int32_t batch_size, int32_t num_heads,
                                                 const char* backend_name) noexcept {
  constexpr int32_t kMaxGridDimYZ = 65535;
  if (batch_size > kMaxGridDimYZ || num_heads > kMaxGridDimYZ) {
    return Invalid(backend_name);
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateRoundedInt32Dimension(int32_t value,
                                                             int32_t alignment,
                                                             const char* message) noexcept {
  if (value > std::numeric_limits<int32_t>::max() - (alignment - 1)) {
    return Invalid(message);
  }

  return Ok();
}

PackedAttentionWorkspaceStatus ValidateMemoryEfficientGeometry(
    size_t element_size,
    int32_t batch_size,
    int32_t sequence_length,
    int32_t num_heads,
    int32_t qk_head_size,
    int32_t v_head_size,
    bool has_attention_bias,
    bool broadcast_attn_bias_dim_0,
    bool broadcast_attn_bias_dim_1) noexcept {
  auto status = ValidateFusedGrid(
      batch_size, num_heads,
      "Packed memory-efficient attention exceeds CUDA gridDim.y or gridDim.z.");
  if (!status.IsOK()) {
    return status;
  }

  // DispatchBlockSize in fmha_launch_template.h selects at most 64
  // AttentionKernel::kQueriesPerBlock, and AttentionKernel::kAlignLSE is 32.
  // Validate both ceil_div round-ups before either CUDA int32 expression is formed.
  constexpr int32_t kMaxQueriesPerBlock = 64;
  constexpr int32_t kAlignLse = 32;
  status = ValidateRoundedInt32Dimension(
      sequence_length, kMaxQueriesPerBlock,
      "Packed memory-efficient attention query-block rounding exceeds int32.");
  if (!status.IsOK()) {
    return status;
  }

  status = ValidateRoundedInt32Dimension(
      sequence_length, kAlignLse,
      "Packed memory-efficient attention LSE rounding exceeds int32.");
  if (!status.IsOK()) {
    return status;
  }

  const size_t b = static_cast<size_t>(batch_size);
  const size_t s = static_cast<size_t>(sequence_length);
  const size_t n = static_cast<size_t>(num_heads);
  const size_t h = static_cast<size_t>(qk_head_size);
  const size_t hv = static_cast<size_t>(v_head_size);

  // CUTLASS stores BSNH batch strides in int64_t.
  status = CheckedProductFitsInt64(n, h, s);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedProductFitsInt64(n, hv, s);
  if (!status.IsOK()) {
    return status;
  }

  if (!has_attention_bias) {
    return Ok();
  }

  size_t bias_matrix_elements = 0;
  status = CheckedPackedAttentionMultiply(s, s, bias_matrix_elements);
  if (!status.IsOK() ||
      bias_matrix_elements > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return status.IsOK() ? Invalid("The MEA attention-bias head stride does not fit int64.") : status;
  }

  const size_t bias_heads = broadcast_attn_bias_dim_1 ? 1 : n;
  size_t bias_batch_stride = 0;
  status = CheckedPackedAttentionMultiply(bias_heads, bias_matrix_elements, bias_batch_stride);
  if (!status.IsOK() ||
      bias_batch_stride > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return status.IsOK() ? Invalid("The MEA attention-bias batch stride does not fit int64.") : status;
  }

  const size_t bias_batches = broadcast_attn_bias_dim_0 ? 1 : b;
  size_t bias_extent = 0;
  status = CheckedPackedAttentionMultiply(bias_batches, bias_batch_stride, bias_extent);
  if (!status.IsOK() ||
      bias_extent > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return status.IsOK() ? Invalid("The MEA attention-bias extent does not fit int64.") : status;
  }

  size_t bias_bytes = 0;
  return CheckedPackedAttentionMultiply(bias_extent, element_size, bias_bytes);
}

PackedAttentionWorkspaceStatus CheckedMultiplyMany(size_t first, size_t second, size_t third,
                                                   size_t fourth, size_t fifth,
                                                   size_t& result) noexcept {
  auto status = CheckedPackedAttentionMultiply(first, second, result);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionMultiply(result, third, result);
  if (!status.IsOK()) {
    return status;
  }

  status = CheckedPackedAttentionMultiply(result, fourth, result);
  if (!status.IsOK()) {
    return status;
  }

  return CheckedPackedAttentionMultiply(result, fifth, result);
}

PackedAttentionWorkspaceStatus ComputeWorkspaceRecipe(size_t element_size,
                                                      int32_t token_count,
                                                      int32_t batch_size,
                                                      int32_t sequence_length,
                                                      int32_t num_heads,
                                                      int32_t qk_head_size,
                                                      int32_t v_head_size,
                                                      bool has_attention_bias,
                                                      bool broadcast_attn_bias_dim_0,
                                                      bool broadcast_attn_bias_dim_1,
                                                      PackedAttentionBackend backend,
                                                      bool no_qkv_workspace,
                                                      PackedAttentionQkvMaterializationIndexWidth
                                                          qkv_materialization_index_width,
                                                      PackedAttentionWorkspaceRecipe& recipe) noexcept {
  const size_t t = static_cast<size_t>(token_count);
  const size_t b = static_cast<size_t>(batch_size);
  const size_t s = static_cast<size_t>(sequence_length);
  const size_t n = static_cast<size_t>(num_heads);
  const size_t h = static_cast<size_t>(qk_head_size);
  const size_t hv = static_cast<size_t>(v_head_size);

  size_t qkv_head_size = 0;
  auto status = Ok();
  if (!no_qkv_workspace) {
    status = ValidateQkvMaterializationGeometry(
        token_count, num_heads, qk_head_size, v_head_size, qkv_materialization_index_width);
    if (!status.IsOK()) {
      return status;
    }

    status = CheckedPackedAttentionAdd(h, h, qkv_head_size);
    if (!status.IsOK()) {
      return status;
    }

    status = CheckedPackedAttentionAdd(qkv_head_size, hv, qkv_head_size);
    if (!status.IsOK()) {
      return status;
    }
  }

  switch (backend) {
    case PackedAttentionBackend::Trt:
      break;
    case PackedAttentionBackend::Flash:
      status = ValidateFusedGrid(
          batch_size, num_heads,
          "Packed Flash Attention exceeds CUDA gridDim.y or gridDim.z.");
      if (status.IsOK()) {
        status = ValidateRoundedInt32Dimension(
            sequence_length, 128,
            "Packed Flash Attention sequence rounding exceeds int32.");
      }
      break;
    case PackedAttentionBackend::MemoryEfficient:
      status = ValidateMemoryEfficientGeometry(
          element_size, batch_size, sequence_length, num_heads, qk_head_size, v_head_size,
          has_attention_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1);
      break;
    case PackedAttentionBackend::Unfused:
      status = ValidateUnfusedGeometry(
          token_count, batch_size, sequence_length, num_heads, qk_head_size, v_head_size,
          qkv_materialization_index_width);
      break;
    default:
      return Invalid("Packed attention backend is invalid.");
  }

  if (!status.IsOK()) {
    return status;
  }

  size_t qkv_capacity_bytes = 0;
  if (!no_qkv_workspace) {
    status = CheckedMultiplyMany(element_size, b, s, n, qkv_head_size, qkv_capacity_bytes);
    if (!status.IsOK()) {
      return status;
    }
  }

  PackedAttentionWorkspaceRecipe result;
  result.no_qkv_workspace = no_qkv_workspace;
  result.qkv_capacity_bytes = qkv_capacity_bytes;

  if (!no_qkv_workspace) {
    if (backend == PackedAttentionBackend::Trt) {
      result.qkv_layout = PackedAttentionQkvWorkspaceLayout::InterleavedTn3h;
      result.interleaved_qkv_offset_bytes = 0;
      status = CheckedMultiplyMany(
          element_size, t, n, qkv_head_size, 1, result.interleaved_qkv_bytes);
      if (!status.IsOK()) {
        return status;
      }
    } else {
      result.qkv_layout = PackedAttentionQkvWorkspaceLayout::Planar;
      result.q_offset_bytes = 0;

      size_t view_tokens = t;
      if (backend == PackedAttentionBackend::Unfused) {
        status = CheckedPackedAttentionMultiply(b, s, view_tokens);
        if (!status.IsOK()) {
          return status;
        }
      }

      status = CheckedMultiplyMany(element_size, view_tokens, n, h, 1, result.q_bytes);
      if (!status.IsOK()) {
        return status;
      }

      // K has the same shape and size as Q.
      result.k_offset_bytes = result.q_bytes;
      result.k_bytes = result.q_bytes;
      status = CheckedPackedAttentionAdd(result.k_offset_bytes, result.k_bytes, result.v_offset_bytes);
      if (!status.IsOK()) {
        return status;
      }

      status = CheckedMultiplyMany(element_size, view_tokens, n, hv, 1, result.v_bytes);
      if (!status.IsOK()) {
        return status;
      }

      status = CheckedPackedAttentionAdd(result.v_offset_bytes, result.v_bytes,
                                         result.backend_workspace_offset_bytes);
      if (!status.IsOK()) {
        return status;
      }
    }
  }

  switch (backend) {
    case PackedAttentionBackend::Trt:
      break;
    case PackedAttentionBackend::Flash:
      // Keep this checked copy in parity with flash::get_softmax_lse_size(S, B, N).
      status = CheckedMultiplyMany(sizeof(float), b, s, n, 1, result.backend_workspace_bytes);
      if (!status.IsOK()) {
        return status;
      }
      break;
    case PackedAttentionBackend::MemoryEfficient:
      // Keep this in parity with MemoryEfficientAttentionParams::need_workspace.
      if (v_head_size > 128 && element_size != sizeof(float)) {
        status = CheckedMultiplyMany(sizeof(float), b, s, n, hv, result.backend_workspace_bytes);
        if (!status.IsOK()) {
          return status;
        }
      }
      break;
    case PackedAttentionBackend::Unfused: {
      // Checked equivalent of the dense GetAttentionScratchSize formula.
      size_t scratch_bytes = 0;
      status = CheckedMultiplyMany(element_size, b, n, s, s, scratch_bytes);
      if (!status.IsOK()) {
        return status;
      }

      status = CheckedPackedAttentionAlign(scratch_bytes, kPackedAttentionWorkspaceAlignment,
                                           result.backend_workspace_bytes);
      if (!status.IsOK()) {
        return status;
      }

      if (result.backend_workspace_offset_bytes != qkv_capacity_bytes) {
        return Invalid("Packed unfused attention QKV views do not end at the legacy QKV capacity.");
      }

      result.has_second_scratch = true;
      status = CheckedPackedAttentionAdd(result.backend_workspace_offset_bytes,
                                         result.backend_workspace_bytes,
                                         result.second_scratch_offset_bytes);
      if (!status.IsOK()) {
        return status;
      }

      size_t both_scratch_bytes = 0;
      status = CheckedPackedAttentionMultiply(result.backend_workspace_bytes, 2, both_scratch_bytes);
      if (!status.IsOK()) {
        return status;
      }

      status = CheckedPackedAttentionAdd(qkv_capacity_bytes, both_scratch_bytes,
                                         result.attention_workspace_bytes);
      if (!status.IsOK()) {
        return status;
      }
      break;
    }
    default:
      return Invalid("Packed attention backend is invalid.");
  }

  if (backend != PackedAttentionBackend::Unfused) {
    status = CheckedPackedAttentionAdd(qkv_capacity_bytes, result.backend_workspace_bytes,
                                       result.attention_workspace_bytes);
    if (!status.IsOK()) {
      return status;
    }
  }

  status = ValidatePackedAttentionWorkspaceRecipe(result);
  if (!status.IsOK()) {
    return status;
  }

  recipe = result;
  return Ok();
}

}  // namespace

PackedAttentionWorkspaceStatus CheckedPackedAttentionAdd(size_t left, size_t right, size_t& result) noexcept {
  // Keep these helpers non-throwing for the provider/plugin boundary. SafeInt<T>
  // construction is intentionally avoided because it reports overflow by throwing.
  size_t checked_result = 0;
  if (!SafeAdd(left, right, checked_result)) {
    return Overflow("Packed attention size addition overflowed size_t.");
  }

  result = checked_result;
  return Ok();
}

PackedAttentionWorkspaceStatus CheckedPackedAttentionMultiply(size_t left, size_t right,
                                                              size_t& result) noexcept {
  size_t checked_result = 0;
  if (!SafeMultiply(left, right, checked_result)) {
    return Overflow("Packed attention size multiplication overflowed size_t.");
  }

  result = checked_result;
  return Ok();
}

PackedAttentionWorkspaceStatus CheckedPackedAttentionAlign(size_t value, size_t alignment,
                                                           size_t& result) noexcept {
  if (alignment == 0) {
    return Invalid("Packed attention alignment must be non-zero.");
  }

  size_t numerator = 0;
  auto status = CheckedPackedAttentionAdd(value, alignment - 1, numerator);
  if (!status.IsOK()) {
    return status;
  }

  const size_t quotient = numerator / alignment;
  return CheckedPackedAttentionMultiply(quotient, alignment, result);
}

PackedAttentionWorkspaceStatus ValidatePackedAttentionWorkspaceRecipe(
    const PackedAttentionWorkspaceRecipe& recipe) noexcept {
  const auto validate_range = [&recipe](size_t offset, size_t bytes) {
    size_t end = 0;
    auto status = CheckedPackedAttentionAdd(offset, bytes, end);
    if (!status.IsOK()) {
      return status;
    }

    return end <= recipe.attention_workspace_bytes
               ? Ok()
               : Invalid("A packed attention workspace view exceeds its allocation.");
  };

  if (recipe.qkv_capacity_bytes > recipe.attention_workspace_bytes) {
    return Invalid("Packed attention QKV capacity exceeds its allocation.");
  }

  PackedAttentionWorkspaceStatus status;
  switch (recipe.qkv_layout) {
    case PackedAttentionQkvWorkspaceLayout::None:
      if (!recipe.no_qkv_workspace ||
          recipe.q_offset_bytes != 0 || recipe.q_bytes != 0 ||
          recipe.k_offset_bytes != 0 || recipe.k_bytes != 0 ||
          recipe.v_offset_bytes != 0 || recipe.v_bytes != 0 ||
          recipe.interleaved_qkv_offset_bytes != 0 || recipe.interleaved_qkv_bytes != 0) {
        return Invalid("A no-QKV-workspace recipe exposes QKV workspace views.");
      }
      break;
    case PackedAttentionQkvWorkspaceLayout::Planar:
      if (recipe.no_qkv_workspace ||
          recipe.interleaved_qkv_offset_bytes != 0 || recipe.interleaved_qkv_bytes != 0) {
        return Invalid("A planar QKV recipe has inconsistent layout fields.");
      }
      status = validate_range(recipe.q_offset_bytes, recipe.q_bytes);
      if (!status.IsOK()) {
        return status;
      }
      status = validate_range(recipe.k_offset_bytes, recipe.k_bytes);
      if (!status.IsOK()) {
        return status;
      }
      status = validate_range(recipe.v_offset_bytes, recipe.v_bytes);
      if (!status.IsOK()) {
        return status;
      }
      break;
    case PackedAttentionQkvWorkspaceLayout::InterleavedTn3h:
      if (recipe.no_qkv_workspace ||
          recipe.q_offset_bytes != 0 || recipe.q_bytes != 0 ||
          recipe.k_offset_bytes != 0 || recipe.k_bytes != 0 ||
          recipe.v_offset_bytes != 0 || recipe.v_bytes != 0) {
        return Invalid("An interleaved QKV recipe exposes planar Q/K/V views.");
      }
      status = validate_range(recipe.interleaved_qkv_offset_bytes, recipe.interleaved_qkv_bytes);
      if (!status.IsOK()) {
        return status;
      }
      break;
    default:
      return Invalid("Packed attention QKV workspace layout is invalid.");
  }

  status = validate_range(recipe.backend_workspace_offset_bytes, recipe.backend_workspace_bytes);
  if (!status.IsOK()) {
    return status;
  }

  if (recipe.has_second_scratch) {
    status = validate_range(recipe.second_scratch_offset_bytes, recipe.backend_workspace_bytes);
    if (!status.IsOK()) {
      return status;
    }
  } else if (recipe.second_scratch_offset_bytes != 0) {
    return Invalid("A packed attention recipe without a second scratch region has a second-scratch offset.");
  }

  return Ok();
}

PackedAttentionProblemResult<PackedAttentionProblem> BuildPackedAttentionProblem(
    const PackedAttentionInputShapes& inputs) noexcept {
  PackedAttentionProblemResult<PackedAttentionProblem> result;

  auto status = ValidateElementSize(inputs.element_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateDimension(inputs.num_heads);
  if (!status.IsOK() || inputs.num_heads == 0) {
    result.status = Invalid("PackedAttention num_heads must be positive and fit int32.");
    return result;
  }

  status = ValidateShape(inputs.input, 2);
  if (!status.IsOK()) {
    result.status = Invalid("PackedAttention input must have rank 2.");
    return result;
  }

  status = ValidateShape(inputs.weights, 2);
  if (!status.IsOK()) {
    result.status = Invalid("PackedAttention weights must have rank 2.");
    return result;
  }

  status = ValidateShape(inputs.bias, 1);
  if (!status.IsOK()) {
    result.status = Invalid("PackedAttention bias must have rank 1.");
    return result;
  }

  status = ValidateShape(inputs.token_offset, 2);
  if (!status.IsOK()) {
    result.status = Invalid("PackedAttention token_offset must have rank 2.");
    return result;
  }

  status = ValidateShape(inputs.cumulative_sequence_length, 1);
  if (!status.IsOK()) {
    result.status = Invalid("PackedAttention cumulative_sequence_length must have rank 1.");
    return result;
  }

  const int64_t token_count = inputs.input.dimensions[0];
  const int64_t input_hidden_size = inputs.input.dimensions[1];
  const int64_t batch_size = inputs.token_offset.dimensions[0];
  const int64_t sequence_length = inputs.token_offset.dimensions[1];
  const int64_t num_heads = inputs.num_heads;

  if (inputs.weights.dimensions[0] != input_hidden_size) {
    result.status = Invalid("PackedAttention weights dimension 0 must equal input hidden size.");
    return result;
  }

  if (inputs.bias.dimensions[0] != inputs.weights.dimensions[1]) {
    result.status = Invalid("PackedAttention bias size must equal weights dimension 1.");
    return result;
  }

  size_t batch_plus_one = 0;
  status = CheckedPackedAttentionAdd(static_cast<size_t>(batch_size), 1, batch_plus_one);
  if (!status.IsOK() ||
      batch_plus_one > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      inputs.cumulative_sequence_length.dimensions[0] != static_cast<int64_t>(batch_plus_one)) {
    result.status = Invalid("Cumulative sequence length must have shape [B + 1] within the int32 ABI.");
    return result;
  }

  int64_t q_hidden_size = 0;
  int64_t k_hidden_size = 0;
  int64_t v_hidden_size = 0;
  if (inputs.qkv_hidden_sizes_count == 0) {
    q_hidden_size = inputs.bias.dimensions[0] / 3;
    k_hidden_size = q_hidden_size;
    v_hidden_size = q_hidden_size;
  } else {
    if (inputs.qkv_hidden_sizes_count != inputs.qkv_hidden_sizes.size()) {
      result.status = Invalid("PackedAttention qkv_hidden_sizes must contain exactly three values.");
      return result;
    }

    q_hidden_size = inputs.qkv_hidden_sizes[0];
    k_hidden_size = inputs.qkv_hidden_sizes[1];
    v_hidden_size = inputs.qkv_hidden_sizes[2];
  }

  status = ValidateDimension(q_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateDimension(k_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateDimension(v_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  if (q_hidden_size != k_hidden_size) {
    result.status = Invalid("PackedAttention Q and K hidden sizes must match.");
    return result;
  }

  if (q_hidden_size % num_heads != 0 || v_hidden_size % num_heads != 0) {
    result.status = Invalid("PackedAttention hidden sizes must be divisible by num_heads.");
    return result;
  }

  size_t qkv_hidden_size = 0;
  status = CheckedPackedAttentionAdd(static_cast<size_t>(q_hidden_size),
                                     static_cast<size_t>(k_hidden_size), qkv_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = CheckedPackedAttentionAdd(qkv_hidden_size, static_cast<size_t>(v_hidden_size),
                                     qkv_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  if (qkv_hidden_size != static_cast<size_t>(inputs.bias.dimensions[0]) ||
      qkv_hidden_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    result.status = Invalid("PackedAttention Q/K/V hidden sizes must sum to the projection output size.");
    return result;
  }

  if (inputs.has_attention_bias) {
    status = ValidateAttentionBias(inputs.attention_bias, batch_size, num_heads, sequence_length);
    if (!status.IsOK()) {
      result.status = status;
      return result;
    }
  }

  PackedAttentionProblem problem;
  problem.element_size = inputs.element_size;
  problem.token_count = static_cast<int32_t>(token_count);
  problem.batch_size = static_cast<int32_t>(batch_size);
  problem.sequence_length = static_cast<int32_t>(sequence_length);
  problem.num_heads = static_cast<int32_t>(num_heads);
  problem.input_hidden_size = static_cast<int32_t>(input_hidden_size);
  problem.hidden_size = static_cast<int32_t>(q_hidden_size);
  problem.v_hidden_size = static_cast<int32_t>(v_hidden_size);
  problem.qk_head_size = static_cast<int32_t>(q_hidden_size / num_heads);
  problem.v_head_size = static_cast<int32_t>(v_hidden_size / num_heads);
  problem.has_attention_bias = inputs.has_attention_bias;
  problem.broadcast_attn_bias_dim_0 =
      inputs.has_attention_bias && inputs.attention_bias.dimensions[0] == 1;
  problem.broadcast_attn_bias_dim_1 =
      inputs.has_attention_bias && inputs.attention_bias.dimensions[1] == 1;
  problem.qkv_materialization_index_width =
      GetPackedAttentionQkvMaterializationIndexWidth(problem.qk_head_size, problem.v_head_size);

  status = ValidateCoreGeometry(problem.token_count, problem.batch_size, problem.sequence_length,
                                problem.num_heads, problem.hidden_size, problem.v_hidden_size,
                                problem.qk_head_size, problem.v_head_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  result.problem = problem;
  result.status = Ok();
  return result;
}

PackedAttentionProblemResult<PackedMultiHeadAttentionProblem> BuildPackedMultiHeadAttentionProblem(
    const PackedMultiHeadAttentionInputShapes& inputs) noexcept {
  PackedAttentionProblemResult<PackedMultiHeadAttentionProblem> result;

  auto status = ValidateElementSize(inputs.element_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateDimension(inputs.num_heads);
  if (!status.IsOK() || inputs.num_heads == 0) {
    result.status = Invalid("PackedMultiHeadAttention num_heads must be positive and fit int32.");
    return result;
  }

  if (inputs.query.rank != 2 && inputs.query.rank != 4) {
    result.status = Invalid("PackedMultiHeadAttention query must have rank 2 or 4.");
    return result;
  }

  status = ValidateShape(inputs.query, inputs.query.rank);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateShape(inputs.token_offset, 2);
  if (!status.IsOK()) {
    result.status = Invalid("PackedMultiHeadAttention token_offset must have rank 2.");
    return result;
  }

  status = ValidateShape(inputs.cumulative_sequence_length, 1);
  if (!status.IsOK()) {
    result.status = Invalid("PackedMultiHeadAttention cumulative_sequence_length must have rank 1.");
    return result;
  }

  const int64_t token_count = inputs.query.dimensions[0];
  const int64_t batch_size = inputs.token_offset.dimensions[0];
  const int64_t sequence_length = inputs.token_offset.dimensions[1];
  const int64_t num_heads = inputs.num_heads;
  int64_t hidden_size = 0;
  int64_t v_hidden_size = 0;
  PackedMultiHeadAttentionQkvFormat qkv_format;

  if (inputs.query.rank == 4) {
    if (inputs.has_key || inputs.has_value) {
      result.status = Invalid("Key and value must be absent when packed QKV is used.");
      return result;
    }

    if (inputs.query.dimensions[1] != num_heads ||
        inputs.query.dimensions[2] != 3) {
      result.status = Invalid("Packed QKV must have shape [T, N, 3, H].");
      return result;
    }

    size_t checked_hidden_size = 0;
    status = CheckedPackedAttentionMultiply(static_cast<size_t>(num_heads),
                                            static_cast<size_t>(inputs.query.dimensions[3]),
                                            checked_hidden_size);
    if (!status.IsOK() ||
        checked_hidden_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      result.status = Invalid("Packed QKV hidden size does not fit the int32 CUDA ABI.");
      return result;
    }

    hidden_size = static_cast<int64_t>(checked_hidden_size);
    v_hidden_size = hidden_size;
    qkv_format = PackedMultiHeadAttentionQkvFormat::Packed;
  } else {
    if (!inputs.has_key || !inputs.has_value) {
      result.status = Invalid("Separate Q, K, and V inputs must all be present.");
      return result;
    }

    status = ValidateShape(inputs.key, 2);
    if (!status.IsOK()) {
      result.status = Invalid("PackedMultiHeadAttention key must have rank 2.");
      return result;
    }

    status = ValidateShape(inputs.value, 2);
    if (!status.IsOK()) {
      result.status = Invalid("PackedMultiHeadAttention value must have rank 2.");
      return result;
    }

    if (inputs.key.dimensions != inputs.query.dimensions) {
      result.status = Invalid("Separate query and key shapes must match.");
      return result;
    }

    if (inputs.value.dimensions[0] != token_count) {
      result.status = Invalid("Separate query, key, and value token dimensions must match.");
      return result;
    }

    hidden_size = inputs.query.dimensions[1];
    v_hidden_size = inputs.value.dimensions[1];
    qkv_format = PackedMultiHeadAttentionQkvFormat::Separate;
  }

  if (hidden_size % num_heads != 0 || v_hidden_size % num_heads != 0) {
    result.status = Invalid("PackedMultiHeadAttention hidden sizes must be divisible by num_heads.");
    return result;
  }

  size_t qkv_hidden_size = 0;
  status = CheckedPackedAttentionAdd(static_cast<size_t>(hidden_size),
                                     static_cast<size_t>(hidden_size), qkv_hidden_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = CheckedPackedAttentionAdd(qkv_hidden_size, static_cast<size_t>(v_hidden_size),
                                     qkv_hidden_size);
  if (!status.IsOK() ||
      qkv_hidden_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    result.status = Invalid("PackedMultiHeadAttention Q/K/V hidden-size sum does not fit int32.");
    return result;
  }

  if (inputs.has_bias) {
    status = ValidateShape(inputs.bias, 1);
    if (!status.IsOK() ||
        inputs.bias.dimensions[0] != static_cast<int64_t>(qkv_hidden_size)) {
      result.status = Invalid("PackedMultiHeadAttention bias must match the Q/K/V hidden-size sum.");
      return result;
    }
  }

  size_t batch_plus_one = 0;
  status = CheckedPackedAttentionAdd(static_cast<size_t>(batch_size), 1, batch_plus_one);
  if (!status.IsOK() ||
      batch_plus_one > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      inputs.cumulative_sequence_length.dimensions[0] != static_cast<int64_t>(batch_plus_one)) {
    result.status = Invalid("Cumulative sequence length must have shape [B + 1] within the int32 ABI.");
    return result;
  }

  if (inputs.has_attention_bias) {
    status = ValidateAttentionBias(inputs.attention_bias, batch_size, num_heads, sequence_length);
    if (!status.IsOK()) {
      result.status = status;
      return result;
    }
  }

  PackedMultiHeadAttentionProblem problem;
  problem.element_size = inputs.element_size;
  problem.token_count = static_cast<int32_t>(token_count);
  problem.batch_size = static_cast<int32_t>(batch_size);
  problem.sequence_length = static_cast<int32_t>(sequence_length);
  problem.num_heads = static_cast<int32_t>(num_heads);
  problem.hidden_size = static_cast<int32_t>(hidden_size);
  problem.v_hidden_size = static_cast<int32_t>(v_hidden_size);
  problem.qk_head_size = static_cast<int32_t>(hidden_size / num_heads);
  problem.v_head_size = static_cast<int32_t>(v_hidden_size / num_heads);
  problem.qkv_format = qkv_format;
  problem.has_bias = inputs.has_bias;
  problem.has_attention_bias = inputs.has_attention_bias;
  problem.broadcast_attn_bias_dim_0 =
      inputs.has_attention_bias && inputs.attention_bias.dimensions[0] == 1;
  problem.broadcast_attn_bias_dim_1 =
      inputs.has_attention_bias && inputs.attention_bias.dimensions[1] == 1;
  problem.qkv_materialization_index_width =
      GetPackedAttentionQkvMaterializationIndexWidth(problem.qk_head_size, problem.v_head_size);

  status = ValidateCoreGeometry(problem.token_count, problem.batch_size, problem.sequence_length,
                                problem.num_heads, problem.hidden_size, problem.v_hidden_size,
                                problem.qk_head_size, problem.v_head_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  result.problem = problem;
  result.status = Ok();
  return result;
}

PackedAttentionWorkspaceResult GetPackedAttentionWorkspaceRecipe(
    const PackedAttentionProblem& problem) noexcept {
  PackedAttentionWorkspaceResult result;

  auto status = ValidateElementSize(problem.element_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = ValidateCoreGeometry(problem.token_count, problem.batch_size, problem.sequence_length,
                                problem.num_heads, problem.hidden_size, problem.v_hidden_size,
                                problem.qk_head_size, problem.v_head_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  if (problem.input_hidden_size < 0) {
    result.status = Invalid("PackedAttention input hidden size must be non-negative.");
    return result;
  }

  if (problem.backend == PackedAttentionBackend::Flash) {
    result.status = Invalid("PackedAttention does not have a Flash Attention route.");
    return result;
  }

  if (problem.backend == PackedAttentionBackend::Trt) {
    if (!problem.trt_runner_available) {
      result.status = Invalid("TRT workspace sizing requires an existing validated runner.");
      return result;
    }

    if (problem.qk_head_size != problem.v_head_size) {
      result.status = Invalid("The packed TRT route requires equal Q/K and V head sizes.");
      return result;
    }
  }

  PackedAttentionWorkspaceRecipe recipe;
  status = ComputeWorkspaceRecipe(problem.element_size, problem.token_count, problem.batch_size,
                                  problem.sequence_length, problem.num_heads, problem.qk_head_size,
                                  problem.v_head_size, problem.has_attention_bias,
                                  problem.broadcast_attn_bias_dim_0, problem.broadcast_attn_bias_dim_1,
                                  problem.backend, false, problem.qkv_materialization_index_width, recipe);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  size_t projection_n = 0;
  status = CheckedPackedAttentionAdd(static_cast<size_t>(problem.hidden_size),
                                     static_cast<size_t>(problem.hidden_size), projection_n);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = CheckedPackedAttentionAdd(projection_n, static_cast<size_t>(problem.v_hidden_size),
                                     projection_n);
  if (!status.IsOK() ||
      projection_n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    result.status = Invalid("PackedAttention projection n does not fit the int32 GEMM ABI.");
    return result;
  }

  size_t projection_elements = 0;
  status = CheckedPackedAttentionMultiply(static_cast<size_t>(problem.token_count), projection_n,
                                          projection_elements);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  status = CheckedPackedAttentionMultiply(projection_elements, problem.element_size,
                                          recipe.projection_bytes);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  recipe.projection_m = problem.token_count;
  recipe.projection_n = static_cast<int32_t>(projection_n);
  recipe.projection_k = problem.input_hidden_size;

  result.recipe = recipe;
  result.status = Ok();
  return result;
}

PackedAttentionWorkspaceResult GetPackedMultiHeadAttentionWorkspaceRecipe(
    const PackedMultiHeadAttentionProblem& problem) noexcept {
  PackedAttentionWorkspaceResult result;

  auto status = ValidateElementSize(problem.element_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  if (problem.qkv_format != PackedMultiHeadAttentionQkvFormat::Packed &&
      problem.qkv_format != PackedMultiHeadAttentionQkvFormat::Separate) {
    result.status = Invalid("PackedMultiHeadAttention QKV format is invalid.");
    return result;
  }

  status = ValidateCoreGeometry(problem.token_count, problem.batch_size, problem.sequence_length,
                                problem.num_heads, problem.hidden_size, problem.v_hidden_size,
                                problem.qk_head_size, problem.v_head_size);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  if (problem.backend == PackedAttentionBackend::Trt) {
    if (!problem.trt_runner_available) {
      result.status = Invalid("TRT workspace sizing requires an existing validated runner.");
      return result;
    }

    if (problem.qk_head_size != problem.v_head_size) {
      result.status = Invalid("The packed TRT route requires equal Q/K and V head sizes.");
      return result;
    }
  }

  if (problem.backend == PackedAttentionBackend::Flash) {
    if (problem.qk_head_size != problem.v_head_size || problem.has_attention_bias) {
      result.status = Invalid("The packed Flash route requires equal head sizes and no attention bias.");
      return result;
    }
  }

  const bool no_qkv_workspace =
      (problem.backend == PackedAttentionBackend::Trt &&
       problem.qkv_format == PackedMultiHeadAttentionQkvFormat::Packed &&
       !problem.has_bias) ||
      ((problem.backend == PackedAttentionBackend::Flash ||
        problem.backend == PackedAttentionBackend::MemoryEfficient) &&
       problem.qkv_format == PackedMultiHeadAttentionQkvFormat::Separate &&
       !problem.has_bias);

  PackedAttentionWorkspaceRecipe recipe;
  status = ComputeWorkspaceRecipe(problem.element_size, problem.token_count, problem.batch_size,
                                  problem.sequence_length, problem.num_heads, problem.qk_head_size,
                                  problem.v_head_size, problem.has_attention_bias,
                                  problem.broadcast_attn_bias_dim_0, problem.broadcast_attn_bias_dim_1,
                                  problem.backend, no_qkv_workspace,
                                  problem.qkv_materialization_index_width, recipe);
  if (!status.IsOK()) {
    result.status = status;
    return result;
  }

  result.recipe = recipe;
  result.status = Ok();
  return result;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
