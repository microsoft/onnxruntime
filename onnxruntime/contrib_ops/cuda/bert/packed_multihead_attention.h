// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/bert/packed_attention.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
class PackedMultiHeadAttention final : public PackedAttentionBase<T> {
 public:
  PackedMultiHeadAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status CheckInputs(const TensorShape& query_shape,
                     const Tensor* key_shape,
                     const Tensor* value_shape,
                     const TensorShape& token_offset_shape,
                     const TensorShape& cu_seq_len_shape,
                     const Tensor* relative_position_bias,
                     PackedAttentionParameters& parameters) const;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
