// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/activation/activations_grad.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ACTIVATION_GRAD_KERNEL(x, ver, domain, T)       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      domain,                                                    \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define BINARY_ELEMENTWISE_COMPUTE(x, T)                                                                         \
  template <>                                                                                                    \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                                 \
    BinaryElementwisePreparation prepare;                                                                        \
    ORT_RETURN_IF_ERROR(Prepare(context, &prepare));                                                             \
    Ctx##x func_ctx = MakeFuncCtx();                                                                             \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                                \
        Stream(),                                                                                                \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.lhs_tensor->template Data<T>()),     \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(prepare.rhs_tensor->template Data<T>()),     \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(prepare.output_tensor->template MutableData<T>()), \
        &func_ctx, prepare.output_tensor->Shape().Size());                                                       \
    return Status::OK();                                                                                         \
  }

#define ACTIVATION_GRAD_OP_TYPED(name, ver, domain, T)  \
  REGISTER_ACTIVATION_GRAD_KERNEL(name, ver, domain, T) \
  BINARY_ELEMENTWISE_COMPUTE(name, T)

#define ACTIVATION_GRAD_OP_HFD(name, ver, domain)        \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, MLFloat16) \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, float)     \
  ACTIVATION_GRAD_OP_TYPED(name, ver, domain, double)

ACTIVATION_GRAD_OP_HFD(GeluGrad, 1, kMSDomain);
ACTIVATION_GRAD_OP_HFD(FastGeluGrad, 1, kMSDomain);
ACTIVATION_GRAD_OP_HFD(ReluGrad, 1, kMSDomain);

}  //namespace cuda
}  // namespace onnxruntime
