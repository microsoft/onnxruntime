// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionBase {
 protected:
  AttentionBase(const OpKernelInfo& info);
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const Tensor* mask_index,
                     const Tensor* past) const;

  Tensor* GetPresent(OpKernelContext* context,
                     const Tensor* past,
                     int batch_size,
                     int head_size,
                     int sequence_length,
                     int& past_sequence_length) const;

  int num_heads_;           // number of attention heads
  bool is_unidirectional_;  // whether every token can only attend to previous tokens.
};

}  // namespace contrib
}  // namespace onnxruntime
