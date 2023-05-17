// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "gemm_float8.h"
#include "gemm_float8.cuh"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_FIVE_TYPED(T1, T2, T3, T4, T5)              \
  ONNX_OPERATOR_FIVE_TYPED_KERNEL_EX(                               \
      GemmFloat8,                                                   \
      kMSDomain,                                                    \
      1,                                                            \
      T1, T2, T3, T4, T5,                                           \
      kCudaExecutionProvider,                                       \
      (*KernelDefBuilder::Create())                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T3>())  \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<T4>())  \
          .TypeConstraint("T5", DataTypeImpl::GetTensorType<T5>()), \
      GemmFloat8<T1, T2, T3, T4, T5>);

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

template <typename AType, typename BType, typename CType, typename DType, typename BiasType>
Status GemmFloat8<AType, BType, CType, DType, BiasType>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<DType>::MappedType CudaDType;

  // D = alpha*(A*B) + beta*(C)
  const auto* A = ctx->Input<Tensor>(0);  // X
  const auto* B = ctx->Input<Tensor>(1);  // W
  const auto* C = ctx->Input<Tensor>(2);  // B
  // Bias could be missing. Treat as scalar 0 if that is the case.
  GemmHelper helper(A->Shape(), trans_A_, B->Shape(), trans_B_, C != nullptr ? C->Shape() : TensorShape({}));

  if (!helper.State().IsOK())
    return helper.State();

  int M = gsl::narrow_cast<int>(helper.M());
  int N = gsl::narrow_cast<int>(helper.N());
  int K = gsl::narrow_cast<int>(helper.K());
  auto* Y = ctx->Output(0, {M, N});
  CudaDType* out_data = reinterpret_cast<CudaDType*>(Y->MutableData<T>());

  int lda, ldb, ldd;
  if (trans_A_ && !trans_B_) {  // TN
    lda = k;
    ldb = k;
    ldd = m;
  } else if (!trans_A_ && !trans_B_) {  // NN
    lda = m;
    ldb = k;
    ldd = m;
  } else if (!trans_A_ && trans_B_) {  // NT
    lda = m;
    ldb = n;
    ldd = m;
  } else {  // TT
    ORT_THROW("trans_A == true && trans_B == true not allowed.");
  }

  GemmFloat8_Impl<AType, BType, CType, DType, BiasType>::CudaCompute();
}

}  // namespace cuda
}  // namespace onnxruntime
