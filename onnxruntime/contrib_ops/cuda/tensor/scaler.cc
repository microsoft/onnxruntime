// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scaler.h"
#include "scaler_impl.cuh"

#include "core/providers/common.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::cuda;
namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, U)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      Scaler,                                              \
      kMSDomain,                                                   \
      1,                                                           \
      T##_##U,                                                     \
      kCudaExecutionProvider,                                      \
      KernelDefBuilder()                                           \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())  \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<U>()),  \
      Scaler<T, U>);

REGISTER_KERNEL_TYPED(int32_t, MLFloat16)
REGISTER_KERNEL_TYPED(int32_t, float)

template <typename T, typename U>
Status Scaler<T, U>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* scale = context->Input<Tensor>(1);
  ORT_ENFORCE(X != nullptr &&
              scale != nullptr);

  // Todo: Add support of 1D tensor
  ORT_ENFORCE(IsScalarOr1ElementVector(scale), "scale must be a scalar or 1D tensor of size 1.");

  Tensor* Y = context->Output(0, X->Shape());

  typedef typename ToCudaType<U>::MappedType CudaU;
  ScalerImpl<T, CudaU>(
      X->template Data<T>(),
      reinterpret_cast<const CudaU*>(scale->template Data<U>()),
      reinterpret_cast<CudaU*>(Y->template MutableData<U>()),
      X->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
