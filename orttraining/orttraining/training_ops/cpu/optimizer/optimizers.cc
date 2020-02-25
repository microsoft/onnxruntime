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

ONNX_CPU_OPERATOR_KERNEL(
    SGDOptimizer,
    9,
    KernelDefBuilder()
        .Alias(1, 0)  // Update weights in-place
        .Alias(2, 1)  // Update gradients in-place
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    SGDOptimizer<float>);

template <typename T>
Status AdamOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NS = *ctx->Output(0, S.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor* NW = ctx->Output(3, W.Shape());
  Tensor* NG = ctx->Output(4, G.Shape());

  const float eta = *ETA.template Data<float>();
  const int64_t step = *S.template Data<int64_t>();

  // Update exponentially-averaged historical gradient
  MakeEigenArrayMap<T>(NM1) = alpha_ * MakeEigenArrayMap<T>(M1) + ((1 - alpha_) * MakeEigenArrayMap<T>(G));

  // Update exponentially-averaged historical squared gradient
  MakeEigenArrayMap<T>(NM2) = beta_ * MakeEigenArrayMap<T>(M2) + ((1 - beta_) * MakeEigenArrayMap<T>(G) * MakeEigenArrayMap<T>(G));

  // Compute weight update.
  const auto& denom = MakeEigenArrayMap<T>(NM2).sqrt() + epsilon_;
  const auto& update = (MakeEigenArrayMap<T>(NM1) / denom) + (lambda_ * MakeEigenArrayMap<T>(W));
  const auto& delta = -eta * update;

  // Weight, gradient, and step update.
  if (NG != nullptr) {
    MakeEigenArrayMap<T>(*NG) = delta;
  }
  if (NW != nullptr) {
    MakeEigenArrayMap<T>(*NW) = MakeEigenArrayMap<T>(W) + delta;
  }
  *NS.template MutableData<int64_t>() = step + 1;

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    AdamOptimizer,
    9,
    KernelDefBuilder()
        .Alias(1, 0)  // Update step count in-place
        .Alias(2, 3)  // Update weights in-place
        .Alias(3, 4)  // Update gradients in-place
        .Alias(4, 1)  // Update moment-1 in-place
        .Alias(5, 2)  // Update moment-2 in-place
        .Alias(6, 5)  // Update fp16 weights in-place
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<float>()),
    AdamOptimizer<float>);
}  // namespace contrib
}  // namespace onnxruntime