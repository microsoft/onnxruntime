// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class DenseIntersection final : public OpKernel {
 public:
  explicit DenseIntersection(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ValidateInputShape(
      const TensorShape& w_conv_shape,
      const TensorShape& w_char_embedding_shape) const;

};

}  // namespace contrib
}  // namespace onnxruntime
