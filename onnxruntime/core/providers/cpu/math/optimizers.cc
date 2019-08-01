// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/optimizers.h"

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {

template <typename T>
Status SGDOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor& NW = *ctx->Output(0, W.Shape());

  // NW = W - eta * G
  float eta = *ETA.template Data<float>();
  MakeEigenArrayMap<T>(NW) = MakeEigenArrayMap<T>(W) - eta * MakeEigenArrayMap<T>(G);

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    SGDOptimizer,
    9,
    KernelDefBuilder()
      .Alias(1, 0) // Update weights in-place
      .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer<float>);

}  // namespace onnxruntime
