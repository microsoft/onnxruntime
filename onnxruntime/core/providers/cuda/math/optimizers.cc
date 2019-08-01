// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "optimizers.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    SGDOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(1, 0) // Update weights in-place
      .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    SGDOptimizer);

Status SGDOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& W = *ctx->MutableInput<Tensor>(1);
  const Tensor& G = *ctx->Input<Tensor>(2);
  Tensor& NW = *ctx->Output(0, W.Shape());

  ORT_ENFORCE(W.Shape() == G.Shape());

  SGDOptimizerImpl(
      ETA.template Data<float>(),
      W.template Data<float>(),
      G.template Data<float>(),
      NW.template MutableData<float>(),
      W.Shape().Size());

  return Status::OK();
}

// TODO: Once Schema is checked in to onnx lets fix this to match that
ONNX_OPERATOR_KERNEL_EX(
    AdamOptimizer,
    kOnnxDomain,
    9,
    kCudaExecutionProvider,
    KernelDefBuilder()
      .Alias(1, 3) // Update step count in-place
      .Alias(2, 0) // Update weights in-place
      .Alias(4, 1) // Update moment-1 in-place
      .Alias(5, 2) // Update moment-2 in-place
      .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    AdamOptimizer);

Status AdamOptimizer::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& ETA = *ctx->Input<Tensor>(0);
  const Tensor& T = *ctx->Input<Tensor>(1);
  const Tensor& W = *ctx->Input<Tensor>(2);
  const Tensor& G = *ctx->Input<Tensor>(3);
  const Tensor& M1 = *ctx->Input<Tensor>(4);
  const Tensor& M2 = *ctx->Input<Tensor>(5);

  Tensor& NW = *ctx->Output(0, W.Shape());
  Tensor& NM1 = *ctx->Output(1, M1.Shape());
  Tensor& NM2 = *ctx->Output(2, M2.Shape());
  Tensor& NT = *ctx->Output(3, T.Shape());

  AdamOptimizerImpl(
      ETA.template Data<float>(),
      T.template Data<int64_t>(),
      W.template Data<float>(),
      G.template Data<float>(),
      M1.template Data<float>(),
      M2.template Data<float>(),
      alpha_,
      beta_,
      lambda_,
      epsilon_,
      NW.template MutableData<float>(),
      NM1.template MutableData<float>(),
      NM2.template MutableData<float>(),
      NT.template MutableData<int64_t>(),
      W.Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
