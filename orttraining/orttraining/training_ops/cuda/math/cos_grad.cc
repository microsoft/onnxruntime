// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/math/cos_grad.h"
#include "orttraining/training_ops/cuda/math/cos_grad_impl.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {
namespace cuda {

#define REGISTER_COSGRAD_KERNEL_TYPED(T)                             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      CosGrad,                                                       \
      kMSDomain,                                                      \
      1,                                                              \
      T,                                                              \
      kCudaExecutionProvider,                                         \
      (*KernelDefBuilder::Create())                                   \
          .TypeConstraint("dy", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("Y", DataTypeImpl::GetTensorType<T>())      \
          .TypeConstraint("output", DataTypeImpl::GetTensorType<T>()), \
      CosGrad<T>);

#define BINARY_ELEMENTWISE_COMPUTE(T)                                                                         \
  template <>                                                                                                    \
  Status CosGrad<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare;                                                                        \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                             \
    CtxCosGrad func_ctx = MakeFuncCtx();                                                                             \
    Impl_CosGrad<typename ToCudaType<T>::MappedType>(                                                                \
        Stream(),                                                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        &func_ctx, prepare.output_tensor->Shape().Size());                                                       \
    return Status::OK();                                                                                         \
  }

// template <typename T>
// Status CosGrad<T>::ComputeInternal(OpKernelContext* context) const {
//   typedef typename ToCudaType<T>::MappedType CudaT;
//   const Tensor& dy = *context->Input<Tensor>(0);
//   const Tensor& Y = *context->Input<Tensor>(1);
//   Tensor& output = *context->Output(0, dy.Shape());
//   CosGradImpl<CudaT>(
//       Stream(),
//       reinterpret_cast<const CudaT*>(dy.Data<T>()),
//       reinterpret_cast<const CudaT*>(Y.Data<T>()),
//       output.MutableData<T>(), dy.Shape().Size());

//   return Status::OK();
// }

REGISTER_COSGRAD_KERNEL_TYPED(MLFloat16)
REGISTER_COSGRAD_KERNEL_TYPED(float)
REGISTER_COSGRAD_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
