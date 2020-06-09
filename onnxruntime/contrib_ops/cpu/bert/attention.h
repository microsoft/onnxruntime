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
  Status CheckInputs(const Tensor* input,
                     const Tensor* weights,
                     const Tensor* bias,
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

template <typename T>
class Attention : public OpKernel, public AttentionBase {
 public:
  explicit Attention(const OpKernelInfo& info);
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
