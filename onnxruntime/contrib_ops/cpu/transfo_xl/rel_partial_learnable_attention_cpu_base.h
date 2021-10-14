// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "rel_partial_learnable_attention_base.h"
#include "rel_partial_learnable_attention_helper.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class RelPartialLearnableAttentionCPUBase : public RelPartialLearnableAttentionBase {
 protected:
  RelPartialLearnableAttentionCPUBase(const OpKernelInfo& info) : RelPartialLearnableAttentionBase(info) {}

  template <typename T>
  Status ApplyRelPartialLearnableAttention(const T* Q,               // Q data with size BxNxSxH
                                           const T* K,               // K data with size BxNxSxH
                                           const T* V,               // V value with size BxNxSxH
                                           const Tensor* pos_emb,
                                           const Tensor* pos_emb_weights,
                                           const Tensor* r_w_bias,
                                           const Tensor* r_r_bias,
                                           const Tensor* output_weights,
                                           const Tensor* attn_mask,  // attention mask. nullptr if no mask or its size is SxS
                                           const Tensor* mems,       // memories. nullptr if no memories
                                           Tensor* output,           // output tensor
                                           int batch_size,           // batch size
                                           int sequence_length,      // sequence length
                                           int d_model,              // dimension hidden states
                                           int num_heads,            // number of heads
                                           int head_size,            // head size
                                           OpKernelContext* context) const {
    return Status::OK();
  }
};

}  // namespace contrib
}  // namespace onnxruntime
