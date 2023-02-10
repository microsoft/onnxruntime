// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {

Status LongformerAttentionBase::CheckInputs(const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const TensorShape& attention_mask_shape,
                                            const TensorShape& global_weights_shape,
                                            const TensorShape& global_bias_shape,
                                            const TensorShape& global_mask_shape) const {
  // Input shapes:
  //   input                 : (batch_size, sequence_length, hidden_size)
  //   weights               : (hidden_size, 3 * hidden_size) -- format 1
  //                           (3, hidden_size, hidden_size)  -- format 0
  //   bias                  : (3 * hidden_size)              -- format 1 (bias for Q, K, V)
  //                           (5 * hidden_size)              -- format 0 (bias for Q, K, V, Global_K, Global_V)
  //   attention_mask        : (batch_size, sequence_length)
  //   global_weights        : (hidden_size, 3 * hidden_size) -- format 1
  //                           (3, hidden_size, hidden_size)  -- format 0
  //   global_bias           : (3 * hidden_size)              -- format 1 (bias for Global_Q, Global_K, Global_V)
  //                           (1 * hidden_size)              -- format 0 (bias for Global_Q)
  //   global_attention_mask : (batch_size, sequence_length)

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }

  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  auto hidden_size = dims[2];
  if (sequence_length % (2 * window_) != 0) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Input 'input' dimension 1 should be divisible by 2W, where W is value of the window attribute.");
  }
  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input' dimension 2 should be divisible by value of the num_heads attribute.");
  }

  const auto& weights_dims = weights_shape.GetDims();
  bool use_merged_qkv_weights = (weights_shape.NumDimensions() == 2);
  if (use_merged_qkv_weights) {
    if (weights_dims[0] != hidden_size || weights_dims[1] != 3 * hidden_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'weights' shape should be (hidden_size, 3 * hidden_size) for format 1");
    }
  } else {
    if (weights_dims.size() != 3 ||
        weights_dims[0] != 3 || weights_dims[1] != hidden_size || weights_dims[2] != hidden_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'weights' shape should be (3, hidden_size, hidden_size) for format 0");
    }
  }

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  if (use_merged_qkv_weights) {
    if (bias_dims[0] != 3 * hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'bias' shape should be (3 * hidden_size) for format 1");
    }
  } else {
    if (bias_dims[0] != 5 * hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'bias' shape should be (5 * hidden_size) for format 0");
    }
  }

  const auto& mask_dims = attention_mask_shape.GetDims();
  if (mask_dims.size() == 2) {
    if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[1]) != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'attention_mask' shape shall be (batch_size, sequence_length)");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'attention_mask' is expected to have 2 dimensions, got ", mask_dims.size());
  }

  const auto& global_weights_dims = global_weights_shape.GetDims();
  if (use_merged_qkv_weights) {
    if (global_weights_dims.size() != 2 ||
        global_weights_dims[0] != hidden_size || global_weights_dims[1] != 3 * hidden_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'global_weights' shape should be (hidden_size, 3 * hidden_size) for format 1");
    }
  } else {
    if (global_weights_dims.size() != 3 || global_weights_dims[0] != 3 ||
        global_weights_dims[1] != hidden_size || global_weights_dims[2] != hidden_size) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, INVALID_ARGUMENT,
          "Input 'global_weights' shape should be (3, hidden_size, hidden_size) for format 0");
    }
  }

  const auto& global_bias_dims = global_bias_shape.GetDims();
  if (global_bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'global_bias' is expected to have 1 dimension, got ",
                           global_bias_dims.size());
  }

  if (use_merged_qkv_weights) {
    if (global_bias_dims[0] != 3 * hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'global_bias' shape should be (3 * hidden_size) for format 1");
    }
  } else {
    if (global_bias_dims[0] != hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'global_bias' shape should be (hidden_size) for format 0");
    }
  }

  const auto& global_mask_dims = global_mask_shape.GetDims();
  if (global_mask_dims.size() != 2 ||
      static_cast<int>(global_mask_dims[0]) != batch_size ||
      static_cast<int>(global_mask_dims[1]) != sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'global_attention_mask' shape shall be (batch_size, sequence_length)");
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
