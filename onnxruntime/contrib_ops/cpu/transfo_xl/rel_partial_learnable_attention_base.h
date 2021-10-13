// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelPartialLearnableAttentionBase {
 public:
  // This check function is specifically used in cuda
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& input_weights_shape,
                     const TensorShape& pos_emb_shape,
                     const TensorShape& pos_emb_weights_shape,
                     const TensorShape& r_w_bias_shape,
                     const TensorShape& r_r_bias_shape,
                     const TensorShape& output_weights_shape,
                     const Tensor*& attn_mask,
                     const Tensor*& mems,
                     const int max_threads_per_block) const;

 protected:
  RelPartialLearnableAttentionBase(const OpKernelInfo& info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t head_size = 0;
    ORT_ENFORCE(info.GetAttr("head_size", &head_size).IsOK() && head_size > 0);
    head_size_ = static_cast<int>(head_size);

    int64_t d_model = 0;
    ORT_ENFORCE(info.GetAttr("d_model", &d_model).IsOK() && d_model > 0);
    d_model_ = static_cast<int>(d_model);
  }

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& input_weights_shape,
                     const TensorShape& pos_emb_shape,
                     const TensorShape& pos_emb_weights_shape,
                     const TensorShape& r_w_bias_shape,
                     const TensorShape& r_r_bias_shape,
                     const TensorShape& output_weights_shape,
                     const Tensor*& attn_mask,
                     const Tensor*& mems) const;

  int num_heads_;  // number of attention heads
  int head_size_;  // size of attention heads
  int d_model_;    // dimension of hidden states
};

}  // namespace contrib
}  // namespace onnxruntime
