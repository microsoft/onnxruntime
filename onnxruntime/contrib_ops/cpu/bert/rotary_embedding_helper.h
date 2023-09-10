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
  int batch_size;             // Batch size used by input
  int sequence_length;        // Sequence length used by input
  int hidden_size;            // Hidden size used by input
  int head_size;              // Head size used by cos/sin cache
  int num_heads;              // num_heads = hidden_size / head_size
  int model_format;           // Format of input shapes - 0 is LLaMA Microsoft, 1 is LLaMA Hugging Face
  int max_sequence_length;    // Sequence length used by cos/sin cache
};

template <typename T>
Status CheckInputs(const T* input,
                   const T* position_ids,
                   const T* cos_cache,
                   const T* sin_cache,
                   void* parameters) {

  // When LLaMA Microsoft model:
  //    input        : (batch_size, sequence_length, hidden_size) or (batch_size, sequence_length, num_heads, head_size)
  //    position ids : (1)
  //    cos cache    : (max_sequence_length, head_size / 2)
  //    sin cache    : (max_sequence_length, head_size / 2)
  // When LLaMA Hugging Face model:
  //    input        : (batch_size, num_heads, sequence_length, head_size)
  //    position ids : (batch_size, sequence_length)
  //    cos cache    : (sequence_length, head_size)
  //    sin cache    : (sequence_length, head_size)
  
  // Check input
  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() != 3 && input_dims.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'x' is expected to have 3 or 4 dimensions, got ",
                           input_dims.size());
  }
  // Check position_ids
  const auto& position_ids_dims = position_ids->Shape().GetDims();
  if (position_ids_dims.size() != 1 && position_ids_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' is expected to have 1 or 2 dimensions, got ",
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
  int sequence_length = 0;
  int hidden_size = 0;
  int max_sequence_length = 0;
  int head_size = 0;
  int num_heads = 0;
  int model_format = -1;

  if (position_ids_dims.size() == 1) {
    if (input_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'x' is expected to have 3 dimensions, got ",
                             input_dims.size());
    }
    model_format = 0;
    sequence_length = static_cast<int>(input_dims[1]);
    hidden_size = static_cast<int>(input_dims[2]);    
    max_sequence_length = static_cast<int>(cos_cache_dims[0]);
    head_size = static_cast<int>(cos_cache_dims[1]) * 2;
    num_heads = static_cast<int>(hidden_size / head_size);

  } else {
    if (input_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'x' is expected to have 4 dimensions, got ",
                             input_dims.size());
    }
    model_format = 1;
    num_heads = static_cast<int>(input_dims[1]);
    sequence_length = static_cast<int>(input_dims[2]);
    head_size = static_cast<int>(input_dims[3]);
    hidden_size = static_cast<int>(num_heads * head_size);
    max_sequence_length = sequence_length;

    // Check position_ids input shapes
    if (batch_size != static_cast<int>(position_ids_dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 0 should be same as batch_size, got ",
                            position_ids_dims[0]);
    }
    if (sequence_length != static_cast<int>(position_ids_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'position_ids' dimension 1 should be same as sequence_length, got ",
                            position_ids_dims[1]);
    }
    // Check cos_cache input shapes
    if (sequence_length != static_cast<int>(cos_cache_dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 0 should be same as sequence_length, got ",
                            cos_cache_dims[0]);
    }
    if (head_size != static_cast<int>(cos_cache_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'cos_cache' dimension 1 should be same as head_size, got ",
                            cos_cache_dims[1]);
    }
    // Check sin_cache input shapes
    if (sequence_length != static_cast<int>(sin_cache_dims[0])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' dimension 0 should be same as sequence_length, got ",
                            sin_cache_dims[0]);
    }
    if (head_size != static_cast<int>(sin_cache_dims[1])) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'sin_cache' dimension 1 should be same as head_size, got ",
                            sin_cache_dims[1]);
    }
  }

  // Set rotary parameters
  if (parameters != nullptr) {
    RotaryParameters* output_parameters = reinterpret_cast<RotaryParameters*>(parameters);
    output_parameters->batch_size = batch_size;
    output_parameters->sequence_length = sequence_length;
    output_parameters->hidden_size = hidden_size;
    output_parameters->head_size = head_size;
    output_parameters->num_heads = num_heads;
    output_parameters->model_format = model_format;
    output_parameters->max_sequence_length = max_sequence_length;
  }

  return Status::OK();
}

}  // namespace rotary_embedding_helper
}  // namespace contrib
}  // namespace onnxruntime