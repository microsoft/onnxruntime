// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/tensor/slice.h"

namespace onnxruntime {
namespace contrib {

class SliceGrad final : public OpKernel, public SliceBase {
 public:
  SliceGrad(const OpKernelInfo& info) : OpKernel(info), SliceBase(info, true) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* ctx,
                     Tensor& output_grad_tensor,
                     const gsl::span<const int64_t>& output_dims,
                     TensorShapeVector* flattened_output_dims,
                     const gsl::span<const int64_t>& starts,
                     const gsl::span<const int64_t>& steps) const;
};

}  // namespace contrib
}  // namespace onnxruntime
