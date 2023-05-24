// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "gemm_float8.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL()                                                                                           \
  ONNX_OPERATOR_KERNEL_EX(                                                                                          \
      GemmFloat8,                                                                                                   \
      kMSDomain,                                                                                                    \
      1,                                                                                                            \
      kCudaExecutionProvider,                                                                                       \
      (*KernelDefBuilder::Create())                                                                                 \
          .TypeConstraint("TA", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>())  \
          .TypeConstraint("TB", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>())  \
          .TypeConstraint("TC", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>())  \
          .TypeConstraint("TS", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>())  \
          .TypeConstraint("TR", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>()), \
      GemmFloat8);

REGISTER_KERNEL()

/*
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E4M3FN, BFloat16, BFloat16, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E4M3FN, BFloat16, Float8E4M3FN, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E4M3FN, half, half, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E4M3FN, half, Float8E4M3FN, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E4M3FN, float, float, BFloat16)

REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, BFloat16, BFloat16, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, BFloat16, Float8E4M3FN, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, BFloat16, Float8E5M2, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, half, half, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, half, Float8E4M3FN, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, half, Float8E5M2, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E4M3FN, Float8E5M2, float, float, BFloat16)

REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, BFloat16, BFloat16, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, BFloat16, Float8E4M3FN, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, BFloat16, Float8E5M2, BFloat16)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, half, half, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, half, Float8E4M3FN, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, half, Float8E5M2, half)
REGISTER_KERNEL_FIVE_TYPED(Float8E5M2, Float8E4M3FN, float, float, BFloat16)
*/

Status GemmFloat8::ComputeInternal(OpKernelContext* ctx) const {
  // D = alpha*(A*B) + beta*(C)
  const auto* A = ctx->Input<Tensor>(0);  // X
  const auto* B = ctx->Input<Tensor>(1);  // W
  const auto* C = ctx->Input<Tensor>(2);  // B
  const auto* D = ctx->Input<Tensor>(3);  // result type
  const auto* E = ctx->Input<Tensor>(4);  // bias type

  int32_t dtypes[5] = {A->GetElementType(),
                       B->GetElementType(),
                       C->GetElementType(),
                       D->GetElementType(),
                       E->GetElementType()};

  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(),
                    params_.trans_A_,
                    B->Shape(),
                    params_.trans_B_,
                    C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());

  auto* Y = ctx->Output(0, {M, N});

  cudaStream_t stream = Stream(ctx);
  cublasLtHandle_t cublasLt = CublasLtHandle();

  return this->params_.CudaCompute(dtypes, stream, cublasLt, A, B, C, Y, M, N, K);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
