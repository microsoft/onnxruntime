// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optimizers.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status SGDOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->Input<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor* NW = ctx->Output(0, W.Shape());
  Tensor* NG = ctx->Output(1, G.Shape());

  // NW = W - eta * G
  float eta = *ETA.template Data<float>();
  const auto& delta = -eta * MakeEigenArrayMap<T>(G);

  if (NG != nullptr) {
    MakeEigenArrayMap<T>(*NG) = delta;
  }
  if (NW != nullptr) {
    MakeEigenArrayMap<T>(*NW) = MakeEigenArrayMap<T>(W) + delta;
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    SGDOptimizer,
    1,
    float,
    KernelDefBuilder()
        .Alias(1, 0)  // Update weights in-place
        .Alias(2, 1)  // Update gradients in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer<float>);

}
}  // namespace onnxruntime