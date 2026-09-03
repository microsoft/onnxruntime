// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <limits>
#include <vector>

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cpu/llm/rotary_embedding_int32_utils.h"

namespace onnxruntime {
namespace contrib {
namespace mrotary_embedding_helper {

namespace detail {
using onnxruntime::rotary_embedding_int32_utils::CheckedMulToInt32;
using onnxruntime::rotary_embedding_int32_utils::NarrowNonNegativeToInt32;
}  // namespace detail

// M-RoPE layout describing how the 3 (T, H, W) position streams are combined
// into a single half_rotary_embedding_dim-sized cos/sin vector per token.
enum class MRopeLayout {
  kSectioned = 0,   // contiguous chunks: [T]*section0 + [H]*section1 + [W]*section2 (Qwen2-VL, Qwen2.5-VL)
  kInterleaved = 1  // T everywhere, then H/W punch in every 3rd position (Qwen3-VL family)
};

// Parameters deduced from node attributes and inputs/outputs.
struct MRotaryParameters {
  int batch_size;            // Batch size used by input
  int sequence_length;       // Sequence length used by input
  int hidden_size;           // Hidden size used by input
  int head_size;             // Head size
  int rotary_embedding_dim;  // Rotary embedding dimension.
  int num_heads;             // num_heads = hidden_size / head_size
  int max_sequence_length;   // Sequence length used by cos/sin cache
  int head_stride;           // Head stride
  int seq_stride;            // Sequence stride
  int batch_stride;          // Batch stride
  bool transposed;           // Whether the input tensor has been transposed into (batch, num_heads, seq_len, hidden)
  int mrope_section[3];      // section sizes for T, H, W (sum == rotary_embedding_dim / 2)
  MRopeLayout mrope_layout;  // how the 3 sections are combined
};

// Computes, for every index in [0, half_rotary_embedding_dim), which of the
// 3 position streams (0=T, 1=H, 2=W) contributes the cos/sin value used at
// that index. This mapping depends only on attributes (mrope_section /
// mrope_layout), not on the runtime input, so it can be computed once and
// reused across every token in the Compute() call.
inline void ComputeDimAssignments(const int mrope_section[3], MRopeLayout layout,
                                  int half_rotary_embedding_dim, std::vector<int8_t>& dim_assignment) {
  dim_assignment.assign(static_cast<size_t>(half_rotary_embedding_dim), 0);
  if (layout == MRopeLayout::kSectioned) {
    int pos = 0;
    for (int dim = 0; dim < 3; ++dim) {
      for (int i = 0; i < mrope_section[dim] && pos < half_rotary_embedding_dim; ++i, ++pos) {
        dim_assignment[static_cast<size_t>(pos)] = static_cast<int8_t>(dim);
      }
    }
  } else {
    // Interleaved: everything defaults to T (0); H and W punch in every 3rd slot.
    for (int dim = 1; dim < 3; ++dim) {
      const int64_t length = static_cast<int64_t>(mrope_section[dim]) * 3;
      for (int64_t i = dim; i < length && i < half_rotary_embedding_dim; i += 3) {
        dim_assignment[static_cast<size_t>(i)] = static_cast<int8_t>(dim);
      }
    }
  }
}

template <typename T>
Status CheckInputs(const T* input,
                   const T* position_ids,
                   const T* cos_cache,
                   const T* sin_cache,
                   int num_heads,
                   int rotary_embedding_dim,
                   const std::vector<int64_t>& mrope_section_attr,
                   int64_t mrope_layout_attr,
                   void* parameters) {
  //    input        : (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, sequence_length, head_size)
  //    position ids : (3, batch_size, sequence_length)
  //    cos cache    : (max_sequence_length, rotary_embedding_dim / 2)
  //    sin cache    : (max_sequence_length, rotary_embedding_dim / 2)

  // Check input
  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3 && input_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'x' is expected to have 3 or 4 ",
                           "dimensions, got ", input_dims.size());
  }

  // Check position_ids: MRotaryEmbedding always requires a 3D (3, batch_size, sequence_length) tensor.
  const auto& position_ids_dims = position_ids->Shape().GetDims();
  if (position_ids_dims.size() != 3 || position_ids_dims[0] != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MRotaryEmbedding: 'position_ids' is expected to have shape "
                           "(3, batch_size, sequence_length), got ",
                           position_ids_dims.size(), " dimensions",
                           position_ids_dims.empty() ? "" : " with dim[0]=",
                           position_ids_dims.empty() ? 0 : position_ids_dims[0]);
  }

  // Check cos_cache and sin_cache
  const auto& cos_cache_dims = cos_cache->Shape().GetDims();
  if (cos_cache_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'cos_cache' is expected to ",
                           "have 2 dimensions, got ", cos_cache_dims.size());
  }
  const auto& sin_cache_dims = sin_cache->Shape().GetDims();
  if (sin_cache_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'sin_cache' is expected to ",
                           "have 2 dimensions, got ", sin_cache_dims.size());
  }
  if (cos_cache_dims[0] != sin_cache_dims[0] || cos_cache_dims[1] != sin_cache_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Inputs 'cos_cache' and 'sin_cache' ",
                           "are expected to have the same shape");
  }

  if (rotary_embedding_dim > 0 && num_heads == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: num_heads must be provided if ",
                           "rotary_embedding_dim is specified");
  }

  int batch_size = 0;
  int sequence_length = 0;
  int hidden_size = 0;
  int head_size = 0;

  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[0], "batch_size", batch_size));
  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[1], "sequence_length", sequence_length));
  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[2], "hidden_size", hidden_size));

  bool transposed = false;
  if (input_dims.size() == 4) {
    // input is [batch, num_heads, seq, head_size]
    ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[2], "sequence_length", sequence_length));
    ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[1], "num_heads", num_heads));
    ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(input_dims[3], "head_size", head_size));
    ORT_RETURN_IF_ERROR(detail::CheckedMulToInt32(num_heads, head_size, "hidden_size", hidden_size));
    transposed = true;
  } else {
    if (num_heads <= 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "MRotaryEmbedding: num_heads must be greater than 0 for rank-3 input");
    }
    if (hidden_size % num_heads != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "MRotaryEmbedding: hidden_size=", hidden_size,
                             " must be divisible by num_heads=", num_heads, " for rank-3 input");
    }
    head_size = static_cast<int>(hidden_size / num_heads);
  }

  int max_sequence_length = 0;
  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(cos_cache_dims[0], "max_sequence_length", max_sequence_length));

  const int effective_rotary_dim = rotary_embedding_dim > 0 ? rotary_embedding_dim : head_size;
  if (effective_rotary_dim > head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: rotary_embedding_dim must be less ",
                           "than or equal to head_size");
  }
  if (input->Shape().Size() > 0 && (effective_rotary_dim <= 0 || (effective_rotary_dim % 2) != 0)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MRotaryEmbedding: effective rotary_embedding_dim must be positive and even for ",
                           "non-empty inputs, got ", effective_rotary_dim);
  }
  if (static_cast<int64_t>(effective_rotary_dim / 2) != cos_cache_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'cos_cache' dimension 1 should ",
                           "be same as head_size / 2 or rotary_embedding_dim / 2, got ", cos_cache_dims[1]);
  }

  // Check position_ids input shape against batch_size/sequence_length
  int position_ids_batch = 0;
  int position_ids_sequence = 0;
  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(position_ids_dims[1], "position_ids_dim1", position_ids_batch));
  ORT_RETURN_IF_ERROR(detail::NarrowNonNegativeToInt32(position_ids_dims[2], "position_ids_dim2", position_ids_sequence));
  if (batch_size != position_ids_batch) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'position_ids' dimension 1 ",
                           "should be batch_size, got ", position_ids_batch);
  }
  if (sequence_length != position_ids_sequence) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MRotaryEmbedding: Input 'position_ids' dimension 2 ",
                           "should be sequence_length, got ", position_ids_sequence);
  }

  if (mrope_section_attr.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MRotaryEmbedding: 'mrope_section' attribute must have exactly 3 elements, got ",
                           mrope_section_attr.size());
  }

  const int half_rotary_embedding_dim = effective_rotary_dim / 2;
  int64_t section_sum = 0;
  int mrope_section[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    if (mrope_section_attr[static_cast<size_t>(i)] < 0 ||
        mrope_section_attr[static_cast<size_t>(i)] > std::numeric_limits<int>::max()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "MRotaryEmbedding: 'mrope_section' values must be non-negative and fit in int32");
    }
    mrope_section[i] = static_cast<int>(mrope_section_attr[static_cast<size_t>(i)]);
    section_sum += mrope_section_attr[static_cast<size_t>(i)];
  }
  if (section_sum != static_cast<int64_t>(half_rotary_embedding_dim)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MRotaryEmbedding: sum of 'mrope_section' (", section_sum,
                           ") must equal rotary_embedding_dim / 2 (", half_rotary_embedding_dim, ")");
  }

  if (mrope_layout_attr != 0 && mrope_layout_attr != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "MRotaryEmbedding: 'mrope_layout' must be 0 (Sectioned) or 1 (Interleaved), got ",
                           mrope_layout_attr);
  }

  // Calculate stride values
  int head_stride;
  int seq_stride;
  int batch_stride;
  if (transposed) {
    // Transposed input tensor shape is [batch, n_heads, seq_len, head_size]
    seq_stride = head_size;
    ORT_RETURN_IF_ERROR(detail::CheckedMulToInt32(sequence_length, seq_stride, "head_stride", head_stride));
    ORT_RETURN_IF_ERROR(detail::CheckedMulToInt32(num_heads, head_stride, "batch_stride", batch_stride));
  } else {
    // Default input tensor shape is [batch, seq_len, hidden_size]
    head_stride = head_size;
    ORT_RETURN_IF_ERROR(detail::CheckedMulToInt32(num_heads, head_stride, "seq_stride", seq_stride));
    ORT_RETURN_IF_ERROR(detail::CheckedMulToInt32(sequence_length, seq_stride, "batch_stride", batch_stride));
  }

  if (parameters != nullptr) {
    MRotaryParameters* output_parameters = reinterpret_cast<MRotaryParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->hidden_size = hidden_size;
    output_parameters->head_size = head_size;
    output_parameters->num_heads = num_heads;
    output_parameters->max_sequence_length = max_sequence_length;
    output_parameters->head_stride = head_stride;
    output_parameters->seq_stride = seq_stride;
    output_parameters->batch_stride = batch_stride;
    output_parameters->transposed = transposed;
    output_parameters->rotary_embedding_dim = effective_rotary_dim;
    for (int i = 0; i < 3; ++i) {
      output_parameters->mrope_section[i] = mrope_section[i];
    }
    output_parameters->mrope_layout = mrope_layout_attr == 1 ? MRopeLayout::kInterleaved : MRopeLayout::kSectioned;
  }

  return Status::OK();
}

}  // namespace mrotary_embedding_helper
}  // namespace contrib
}  // namespace onnxruntime
