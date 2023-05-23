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

#define REGISTER_KERNEL()                                                 \
  ONNX_OPERATOR_KERNEL_EX(                                                \
      GemmFloatByte,                                                      \
      kMSDomain,                                                          \
      1,                                                                  \
      kCudaExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                       \
          .TypeConstraint("T", BuildKernelDefConstraints<Float8E4M3FN>()) \
          .TypeConstraint("T2", BuildKernelDefConstraints<MLFloat16>()),  \
      GemmFloatByte);

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

Status GemmFloatByte::ComputeInternal(OpKernelContext* ctx) const {
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
  // cublasHandle_t cublas = GetCublasHandle(ctx);
  cublasLtHandle_t cublasLt = CublasLtHandle();

  if (dtypes[0] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN &&
      dtypes[1] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN &&
      dtypes[2] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      dtypes[3] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
      dtypes[4] == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    this->params_.CudaCompute<Float8E4M3FN, Float8E4M3FN, MLFloat16, MLFloat16, MLFloat16>(
      stream, cublasLt, A, B, C, Y, nullptr, M, N, K);
  } else {
    ORT_THROW("Unable to find an implementation for GemmFloatByte and types ",
              dtypes[0], ",", dtypes[1], ",", dtypes[2], ",", dtypes[3], ",", dtypes[4], ".");
  }
  return helper.State();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
