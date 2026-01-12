// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace rotary_embedding_helper {

// Parameters deduced from node attributes and inputs/outputs.
struct RotaryParameters {
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
  int position_ids_format;   // Format of position ids - 0 is (0), 1 is (batch_size, sequence_length)
  bool transposed;           // Whether the input tensor has been transposed into (batch, num_heads, seq_len, hidden)
};

template <typename T>
Status CheckInputs(const T* input,
                   const T* position_ids,
                   const T* cos_cache,
                   const T* sin_cache,
                   int num_heads,
                   int rotary_embedding_dim,
                   void* parameters) {
  //    input        : (batch_size, sequence_length, hidden_size) or (batch_size, num_heads, sequence_length, head_size)
  //    IF position ids : (0)
  //            rotary_embedding_dim == 0:
  //                cos_cache    : (batch_size, sequence_length, head_size / 2)
  //                sin_cache    : (batch_size, sequence_length, head_size / 2)
  //            rotary_embedding_dim > 0:
  //                cos_cache    : (batch_size, sequence_length, rotary_embedding_dim / 2)
  //                sin_cache    : (batch_size, sequence_length, rotary_embedding_dim / 2)
  //    ELSE position ids : (batch_size, sequence_length)
  //            rotary_embedding_dim == 0:
  //                cos_cache    : (max_position_id_plus_1, head_size / 2)
  //                sin_cache    : (max_position_id_plus_1, head_size / 2)
  //            rotary_embedding_dim > 0:
  //                cos_cache    : (max_position_id_plus_1, rotary_embedding_dim / 2)
  //                sin_cache    : (max_position_id_plus_1, rotary_embedding_dim / 2)

  // Check input is either 3d or 4d
  const auto& input_dims = input->Shape().GetDims();

  // Get attributes from inputs
  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length;
  int hidden_size;
  int head_size;

  // If it's 4d, it is expected to have shape [batch, num_heads, seq_len, head_size].
  bool transposed = false;
  if (input_dims.size() == 4) {
    sequence_length = static_cast<int>(input_dims[2]);
    num_heads = static_cast<int>(input_dims[1]);
    head_size = static_cast<int>(input_dims[3]);
    hidden_size = num_heads * head_size;
    transposed = true;
  } else if (input_dims.size() == 3) {
    // If it's 3d, it is expected to have shape [batch, seq_len, hidden_size].
    sequence_length = static_cast<int>(input_dims[1]);
    hidden_size = static_cast<int>(input_dims[2]);
    head_size = hidden_size / num_heads;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'x' is expected to have 3 or 4 dimensions, got ",
                           input_dims.size());
  }

  int position_ids_format = 0;
  int max_sequence_length = 0;
  // if position_ids is not provided, cos_cache and sin_cache are expected to have 3 dimensions
  // else they are expected to have 2 dimensions.
  if (nullptr == position_ids) {
    // Check cos_cache and sin_cache
    const auto& cos_cache_dims = cos_cache->Shape().GetDims();
    if (cos_cache_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' is expected to have 3 dimensions, got ",
                             cos_cache_dims.size());
    }
    const auto& sin_cache_dims = sin_cache->Shape().GetDims();
    if (sin_cache_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' is expected to have 3 dimensions, got ",
                             sin_cache_dims.size());
    }
    if (cos_cache_dims[0] != sin_cache_dims[0] || cos_cache_dims[1] != sin_cache_dims[1] || cos_cache_dims[2] != sin_cache_dims[2]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'cos_cache' and 'sin_cache' are expected to have ",
                             "the same shape");
    }
    // Make sure cos_cache and sin_cache have the same batch size and sequence length as input x
    // when position_ids is not provided.
    if (cos_cache_dims[0] != batch_size || cos_cache_dims[1] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'cos_cache' and 'sin_cache' are expected to have ",
                             "the same shape as input 'x', got ", cos_cache_dims[0], " and ", cos_cache_dims[1]);
    }

    max_sequence_length = static_cast<int>(cos_cache_dims[1]);

    if (rotary_embedding_dim > 0 && rotary_embedding_dim > head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "rotary_embedding_dim must be less than or equal to ",
                             "head_size");
    }
    // Check cos_cache input shapes
    if (cos_cache_dims[2] != (rotary_embedding_dim > 0 ? rotary_embedding_dim : head_size) / 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 2 should be same as ",
                             "head_size / 2 or rotary_embedding_dim / 2, got ", cos_cache_dims[2]);
    }
  } else {
    // Check cos_cache and sin_cache
    const auto& cos_cache_dims = cos_cache->Shape().GetDims();
    if (cos_cache_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' is expected to have 2 dimensions, got ",
                             cos_cache_dims.size());
    }
    const auto& sin_cache_dims = sin_cache->Shape().GetDims();
    if (sin_cache_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' is expected to have 2 dimensions, got ",
                             sin_cache_dims.size());
    }
    if (cos_cache_dims[0] != sin_cache_dims[0] || cos_cache_dims[1] != sin_cache_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'cos_cache' and 'sin_cache' are expected to have ",
                             "the same shape");
    }
    // Check position_ids
    const auto& position_ids_dims = position_ids->Shape().GetDims();
    if (position_ids_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' is expected to have 2 ",
                             "dimensions, got ", position_ids_dims.size());
    }

    max_sequence_length = static_cast<int>(cos_cache_dims[0]);

    if (rotary_embedding_dim > 0 && rotary_embedding_dim > head_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "rotary_embedding_dim must be less than or equal to ",
                             "head_size");
    }

    // Check position_ids input shapes
    if (batch_size != static_cast<int>(position_ids_dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 0 should be of size ",
                             "batch_size, got ", position_ids_dims[0]);
    }
    if (sequence_length != static_cast<int>(position_ids_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 1 should be of size ",
                             "sequence_length, got ", position_ids_dims[1]);
    }
    position_ids_format = 1;

    // Check cos_cache input shapes
    if (cos_cache_dims[1] != (rotary_embedding_dim > 0 ? rotary_embedding_dim : head_size) / 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 1 should be same as ",
                             "head_size / 2 or rotary_embedding_dim / 2, got ", cos_cache_dims[1]);
    }
  }

  if (sequence_length > max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

  num_heads = num_heads > 0 ? num_heads : static_cast<int>(hidden_size / head_size);
  // Calculate stride values
  int head_stride;
  int seq_stride;
  int batch_stride;
  if (transposed) {
    // Transposed input tensor shape is [batch, n_heads, seq_len, head_size]
    seq_stride = head_size;
    head_stride = sequence_length * seq_stride;
    batch_stride = num_heads * head_stride;
  } else {
    // Default input tensor shape is [batch, seq_len, hidden_size]
    head_stride = head_size;
    seq_stride = num_heads * head_stride;
    batch_stride = sequence_length * seq_stride;
  }

  // Set rotary parameters
  if (parameters != nullptr) {
    RotaryParameters* output_parameters = reinterpret_cast<RotaryParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->hidden_size = hidden_size;
    output_parameters->head_size = head_size;
    output_parameters->num_heads = num_heads;
    output_parameters->max_sequence_length = max_sequence_length;
    output_parameters->head_stride = head_stride;
    output_parameters->seq_stride = seq_stride;
    output_parameters->batch_stride = batch_stride;
    output_parameters->position_ids_format = position_ids_format;
    output_parameters->transposed = transposed;
    output_parameters->rotary_embedding_dim = rotary_embedding_dim > 0 ? rotary_embedding_dim : head_size;
  }

  return Status::OK();
}

}  // namespace rotary_embedding_helper
}  // namespace onnxruntime
