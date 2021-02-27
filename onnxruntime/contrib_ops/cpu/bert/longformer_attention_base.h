// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class LongformerAttentionBase {
 protected:
  LongformerAttentionBase(const OpKernelInfo& info);

  Status CheckInputs(const TensorShape& input_shape,
                     const TensorShape& weights_shape,
                     const TensorShape& bias_shape,
                     const TensorShape& mask_shape,
                     const TensorShape& global_weights_shape,
                     const TensorShape& global_bias_shape,
                     const TensorShape& global_shape) const;

  int num_heads_;  // Number of attention heads
  int window_;     // Attention windows length (W). It is half (one-sided) of total window size.
};

namespace longformer {
// Environment variable to give a hint about choosing kernels for less memory or latency.
constexpr const char* kUseCompactMemory = "ORT_LONGFORMER_COMPACT_MEMORY";
}  // namespace longformer

}  // namespace contrib
}  // namespace onnxruntime
