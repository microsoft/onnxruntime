// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace rotary_embedding_helper {

// Parameters deduced from node attributes and inputs/outputs.
struct RotaryParameters {
  int batch_size;           // Batch size used by input
  int sequence_length;      // Sequence length used by input
  int hidden_size;          // Hidden size used by input
  int head_size;            // Head size used by cos/sin cache * 2
  int num_heads;            // num_heads = hidden_size / head_size
  int max_sequence_length;  // Sequence length used by cos/sin cache
  int position_ids_format;  // Format of position ids - 0 is (1), 1 is (batch_size, sequence_length)
};

template <typename T>
Status CheckInputs(const T* input,
                   const T* position_ids,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters) {
  //    input        : (batch_size, sequence_length, hidden_size)
  //    position ids : (1) or (batch_size, sequence_length)
  //    cos cache    : (max_sequence_length, head_size / 2)
  //    sin cache    : (max_sequence_length, head_size / 2)

  // Check input
  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'x' is expected to have 3 dimensions, got ",
                           input_dims.size());
  }
  // Check position_ids
  const auto& position_ids_dims = position_ids->Shape().GetDims();
  if (!onnxruntime::IsScalarOr1ElementVector(position_ids) && position_ids_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' is expected to have 0, 1, or 2 dimensions, got ",
                           position_ids_dims.size());
  }
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
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'cos_cache' and 'sin_cache' are expected to have the same shape");
  }

  // Get attributes from inputs
  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);
  int hidden_size = static_cast<int>(input_dims[2]);
  int max_sequence_length = static_cast<int>(cos_cache_dims[0]);
  int head_size = static_cast<int>(cos_cache_dims[1]) * 2;
  int num_heads = hidden_size / head_size;
  int position_ids_format = -1;

  // Check position_ids input shapes
  if (!onnxruntime::IsScalarOr1ElementVector(position_ids)) {
    if (batch_size != static_cast<int>(position_ids_dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 0 should be of size batch_size, got ",
                             position_ids_dims[0]);
    }
    if (sequence_length != static_cast<int>(position_ids_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 1 should be of size sequence_length, got ",
                             position_ids_dims[1]);
    }
    position_ids_format = 1;
  } else {
    position_ids_format = 0;
  }
  // Check cos_cache input shapes
  if (max_sequence_length != static_cast<int>(cos_cache_dims[0])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 0 should be same as max_sequence_length, got ",
                           cos_cache_dims[0]);
  }
  if ((head_size / 2) != static_cast<int>(cos_cache_dims[1])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 1 should be same as head_size / 2, got ",
                           cos_cache_dims[1]);
  }
  // Check sin_cache input shapes
  if (max_sequence_length != static_cast<int>(sin_cache_dims[0])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' dimension 0 should be same as max_sequence_length, got ",
                           sin_cache_dims[0]);
  }
  if ((head_size / 2) != static_cast<int>(sin_cache_dims[1])) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' dimension 1 should be same as head_size / 2, got ",
                           sin_cache_dims[1]);
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
    output_parameters->position_ids_format = position_ids_format;
  }

  return Status::OK();
}

}  // namespace rotary_embedding_helper
}  // namespace contrib
}  // namespace onnxruntime