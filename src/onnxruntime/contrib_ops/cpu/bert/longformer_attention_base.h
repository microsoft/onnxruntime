// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class LongformerAttentionBase {
 public:
  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const TensorShape& attention_mask_shape,
                     const TensorShape& global_weights_shape,
                     const TensorShape& global_bias_shape,
                     const TensorShape& global_attention_mask_shape) const;

 protected:
  LongformerAttentionBase(const OpKernelInfo& info) {
    int64_t num_heads = 0;
    ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
    num_heads_ = static_cast<int>(num_heads);

    int64_t window = 0;
    ORT_ENFORCE(info.GetAttr("window", &window).IsOK() && window > 0);
    window_ = static_cast<int>(window);
  }

  int num_heads_;  // Number of attention heads
  int window_;     // Attention windows length (W). It is half (one-sided) of total window size.
};

namespace longformer {
// Environment variable to choose compact memory kernel in experiment. Default is true.
constexpr const char* kUseCompactMemory = "ORT_LONGFORMER_COMPACT_MEMORY";

// Environment variable to enable half4 in AddBiasTranspose kernel. Default is true.
constexpr const char* kUseHalf4 = "ORT_LONGFORMER_USE_HALF4";

}  // namespace longformer

}  // namespace contrib
}  // namespace onnxruntime
