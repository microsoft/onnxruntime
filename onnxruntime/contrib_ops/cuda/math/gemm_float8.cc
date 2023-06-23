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
          .TypeConstraint("TR", BuildKernelDefConstraints<Float8E4M3FN, Float8E5M2, MLFloat16, BFloat16, float>())  \
          .TypeConstraint("TS", BuildKernelDefConstraints<float>())  \
      GemmFloat8);

REGISTER_KERNEL()


GemmFloat8::  GemmFloat8(const OpKernelInfo& info) : CudaKernel(info){
    params_.transA_ = info.GetAttrOrDefault<int64_t>("transA", 0);
    params_.transB_ = info.GetAttrOrDefault<int64_t>("transB", 0);
    params_.fastAccumulationMode_ = info.GetAttrOrDefault<int64_t>("fastAccumulationMode", 1) != 0;
    params_.rowMajor_ = info.GetAttrOrDefault<int64_t>("rowMajor", 1) != 0;
    params_.smCount_ = info.GetAttrOrDefault<int64_t>("smCount", 0);
    params_.alpha_ = info.GetAttrOrDefault<float>("alpha", 1);

    std::string stemp = info.GetAttrOrDefault<std::string>("computeType", "CUBLAS_COMPUTE_32F");
    if (stemp == "CUBLAS_COMPUTE_16F") {
      params_.computeType_ = CUBLAS_COMPUTE_16F;
      params_.scaleType_ = CUDA_R_16F;
    } else if (stemp == "CUBLAS_COMPUTE_32F") {
      params_.computeType_ = CUBLAS_COMPUTE_32F;
      params_.scaleType_ = CUDA_R_32F;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16F") {
      params_.computeType_ = CUBLAS_COMPUTE_32F_FAST_16F;
      params_.scaleType_ = CUDA_R_16F;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_16BF") {
      params_.computeType_ = CUBLAS_COMPUTE_32F_FAST_16BF;
      params_.scaleType_ = CUDA_R_16BF;
    } else if (stemp == "CUBLAS_COMPUTE_32F_FAST_TF32") {
      params_.computeType_ = CUBLAS_COMPUTE_32F_FAST_TF32;
      params_.scaleType_ = CUDA_R_32F;
    } else {
      ORT_THROW("Unexpected value for compute_type: ", stemp, ".");
    }
  }


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
