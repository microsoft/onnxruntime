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

template <typename T>
Status AdamOptimizer<T>::Compute(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& S = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NW = *ctx->Output(0, W.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor& NS = *ctx->Output(3, S.Shape());

  const float eta = *ETA.template Data<float>();
  const int64_t step = *S.template Data<int64_t>();

  // Regularize gradient
  auto g_regularized = lambda_ * MakeEigenArrayMap<T>(W) + MakeEigenArrayMap<T>(G);

  // Update exponentially-averaged historical gradient
  MakeEigenArrayMap<T>(NM1) = alpha_ * MakeEigenArrayMap<T>(M1) + ((1 - alpha_) * g_regularized);

  // Update exponentially-averaged historical squared gradient
  MakeEigenArrayMap<T>(NM2) = beta_ * MakeEigenArrayMap<T>(M2) + ((1 - beta_) * g_regularized * g_regularized);

  // Update learning rate - use the updated eta for the final weight update
  const float numerator = std::sqrt(1 - std::pow(beta_, static_cast<float>(step)));
  const float denom = 1 - std::pow(alpha_, static_cast<float>(step));
  const float eta_new = eta * numerator / denom;

  // Weight and step update.
  MakeEigenArrayMap<T>(NW) = MakeEigenArrayMap<T>(W) - ((eta_new * MakeEigenArrayMap<T>(NM1)) / (MakeEigenArrayMap<T>(NM2).sqrt() + epsilon_));
  *NS.template MutableData<int64_t>() = step + 1;

  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    AdamOptimizer,
    9,
    KernelDefBuilder()
      .Alias(1, 3) // Update step count in-place
      .Alias(2, 0) // Update weights in-place
      .Alias(4, 1) // Update moment-1 in-place
      .Alias(5, 2) // Update moment-2 in-place
      .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>())
      .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("T4", DataTypeImpl::GetTensorType<float>())
      .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<float>()),
    AdamOptimizer<float>);

}  // namespace onnxruntime
