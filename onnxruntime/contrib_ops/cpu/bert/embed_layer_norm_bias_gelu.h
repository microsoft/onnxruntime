// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "embed_layer_norm.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class EmbedLayerNormBiasGelu final : public EmbedLayerNormBase {
 public:
  explicit EmbedLayerNormBiasGelu(const OpKernelInfo& op_kernel_info);

  Status Compute(OpKernelContext* context) const override;

  //
  // TODO(kreeger): Port over transA|transB|alpha attributes from the matmuls!
  //

  //private:
  TensorShape matmul_1_b_shape_;
  TensorShape matmul_2_b_shape_;

  BufferUniquePtr matmul_1_packed_b_;
  BufferUniquePtr matmul_2_packed_b_;
};

}  // namespace contrib
}  // namespace onnxruntime
