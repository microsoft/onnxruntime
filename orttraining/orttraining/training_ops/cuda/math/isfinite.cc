// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/isfinite.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_ISFINITE_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      IsFinite,                                                       \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>()), \
      IsFiniteOp<T>);

template <typename TSrc>
Status IsFiniteOp<TSrc>::ComputeInternal(OpKernelContext* context) const {
  typedef typename ToCudaType<TSrc>::MappedType CudaTSrc;
  const Tensor& input = *context->Input<Tensor>(0);
  Tensor& output = *context->Output(0, input.Shape());
  IsFinite(
      Stream(),
      reinterpret_cast<const CudaTSrc*>(input.Data<TSrc>()),
      output.MutableData<bool>(), input.Shape().Size());

  return Status::OK();
}

REGISTER_ISFINITE_KERNEL_TYPED(MLFloat16)
REGISTER_ISFINITE_KERNEL_TYPED(float)
REGISTER_ISFINITE_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
