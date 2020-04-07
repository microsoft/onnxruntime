// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
class Inverse final : public OpKernel {
 public:
  explicit Inverse(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext* ctx) const override;

private:
  template<typename T>
  struct ComputeImpl;
};

ONNX_CPU_OPERATOR_KERNEL(
    Inverse,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllNumericTensorTypes()),
    Inverse);

template <typename T>
struct Inverse::ComputeImpl {
  void operator()(OpKernelContext* ctx) const {

  }
};


Status Inverse::Compute(OpKernelContext* ctx) const {

  const auto& input = ctx->Input<Tensor>(0);
  const auto& input_shape = input->Shape();
  const auto num_dim = input_shape.NumDimensions();

  return Status::OK();
}

}  // namespace onnxruntime
