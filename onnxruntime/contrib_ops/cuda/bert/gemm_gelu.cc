// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gemm_gelu.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      GemmGelu,                                                   \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      GemmGelu<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)

using namespace ONNX_NAMESPACE;

template <typename T>
Status GemmGelu<T>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* left_X = context->Input<Tensor>(0);
  const Tensor* right_X = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);

  MatMulComputeHelper helper;
  // TODO: Handle transpose attributes
  ORT_RETURN_IF_ERROR(helper.Compute(left_X->Shape(),
                                     right_X->Shape(),
                                     false, false, false, false, false));

  // TODO: Fix me
  if (helper.OutputOffsets().size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported");
  }

  Tensor* Y = context->Output(0, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  const int lda = helper.Lda(false);
  const int ldb = helper.Ldb(false);
  const int ldc = helper.Ldc();

  CUBLAS_RETURN_IF_ERROR(cublasLtMatmulHelper(
      CublasLtHandle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(helper.N()),
      static_cast<int>(helper.M()),
      static_cast<int>(helper.K()),
      &alpha,
      reinterpret_cast<const CudaT*>(right_X->Data<T>()),
      ldb,
      reinterpret_cast<const CudaT*>(left_X->Data<T>()),
      lda,
      &zero,
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      ldc,
      bias != nullptr
          ? reinterpret_cast<const CudaT*>(bias->Data<T>())
          : nullptr,
      true,
      NULL, 0,
      Stream()));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
