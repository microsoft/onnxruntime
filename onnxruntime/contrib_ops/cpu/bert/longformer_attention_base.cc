// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace onnxruntime {
namespace contrib {

void Link_longformer_attention_base() {}
}  // namespace contrib
}  // namespace onnxruntime

#include "longformer_attention_base.h"

namespace onnxruntime {
namespace contrib {

Status LongformerAttentionBase::CheckInputs(const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const TensorShape& mask_shape,
                                            const TensorShape& global_weights_shape,
                                            const TensorShape& global_bias_shape,
                                            const TensorShape& global_shape) const {
  // Input shapes:
  //   input           : (batch_size, sequence_length, hidden_size)
  //   weights         : (hidden_size, 3 * hidden_size)
  //   bias            : (3 * hidden_size)
  //   mask            : (batch_size, sequence_length)
  //   global_weights  : (hidden_size, 3 * hidden_size)
  //   global_bias     : (3 * hidden_size)
  //   global          : (batch_size, sequence_length)

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }

  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  int hidden_size = static_cast<int>(dims[2]);
  if (sequence_length % (2 * window_) != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input' dimension 1 should be divisiable by 2W, where W is value of the window attribute.");
  }
  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'input' dimension 2 should be divisiable by value of the num_heads attribute.");
  }

  const auto& weights_dims = weights_shape.GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'weights' dimension 0 should have same length as dimension 2 of input 0");
  }
  if (weights_dims[1] != 3 * weights_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' dimension 1 should be 3 times of dimension 0");
  }

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }
  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
  }

  const auto& mask_dims = mask_shape.GetDims();
  if (mask_dims.size() == 2) {
    if (static_cast<int>(mask_dims[0]) != batch_size || static_cast<int>(mask_dims[1]) != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mask' shall have shape batch_size x sequence_length");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'mask' is expected to have 2 dimensions, got ",
                           mask_dims.size());
  }

  const auto& global_weights_dims = global_weights_shape.GetDims();
  if (global_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'global_weights' is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (global_weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'global_weights' dimension 0 should have same length as dimension 2 of input 0");
  }
  if (global_weights_dims[1] != 3 * global_weights_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'global_weights' dimension 1 should be 3 times of dimension 0");
  }

  const auto& global_bias_dims = global_bias_shape.GetDims();
  if (global_bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'global_bias' is expected to have 1 dimension, got ",
                           global_bias_dims.size());
  }
  if (global_bias_dims[0] != global_weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'global_bias' dimension 0 should have same length as dimension 1 of input 'global_weights'");
  }

  const auto& global_dims = global_shape.GetDims();
  if (global_dims.size() == 2) {
    if (static_cast<int>(global_dims[0]) != batch_size || static_cast<int>(global_dims[1]) != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'global' shall have shape batch_size x sequence_length");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'global' is expected to have 2 dimensions, got ",
                           global_dims.size());
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
