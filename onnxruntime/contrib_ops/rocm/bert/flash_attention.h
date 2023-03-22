// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/rocm_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class FlashAttention final : public RocmKernel {
 public:
  FlashAttention(const OpKernelInfo& info);
  Status ComputeInternal(OpKernelContext* context) const override;

  Status CheckInputs(const TensorShape &query_shape,
                     const TensorShape &key_shape,
                     const TensorShape &value_shape,
                     const Tensor* att_mask,
                     const Tensor* att_bias,
		     void *att_param) const;

 private:
  int64_t num_heads_;
  float mask_filter_value_;
  float scale_;
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
