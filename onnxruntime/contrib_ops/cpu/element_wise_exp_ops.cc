// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "element_wise_exp_ops.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    Affine,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Affine<float>);

template <>
Status Affine<float>::Compute(OpKernelContext* ctx) const {
  auto& X = *ctx->Input<Tensor>(0);
  auto& Y = *ctx->Output(0, X.Shape());
  MakeEigenArrayMap<float>(Y) = alpha_ * MakeEigenArrayMap<float>(X) + beta_;
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
