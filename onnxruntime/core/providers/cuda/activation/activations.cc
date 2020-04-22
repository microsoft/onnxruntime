// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activations.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ACTIVATION_KERNEL(x, ver, T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      kOnnxDomain,                                               \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      KernelDefBuilder()                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define REGISTER_ACTIVATION_KERNEL_NONTEMPL(x, ver, ...)                 \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      x,                                                                 \
      kOnnxDomain,                                                       \
      ver,                                                               \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", BuildKernelDefConstraints<__VA_ARGS__>()) \
          .MayInplace(0, 0),                                             \
      x);

#define UNARY_ACTIVATION_COMPUTE(x, T)                                                                     \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    UnaryElementwise::Prepare(context, &p);                                                                \
    Ctx##x func_ctx = MakeFuncCtx();                                                                       \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                          \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        &func_ctx, p.output_tensor->Shape().Size());                                                       \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_ACTIVATION_OP_TYPED(name, ver, T) \
  REGISTER_ACTIVATION_KERNEL(name, ver, T)      \
  UNARY_ACTIVATION_COMPUTE(name, T)

#define UNARY_ACTIVATION_OP_HFD(name, ver)        \
  UNARY_ACTIVATION_OP_TYPED(name, ver, MLFloat16) \
  UNARY_ACTIVATION_OP_TYPED(name, ver, float)     \
  UNARY_ACTIVATION_OP_TYPED(name, ver, double)

UNARY_ACTIVATION_OP_HFD(Elu, 6);
REGISTER_ACTIVATION_KERNEL_NONTEMPL(Celu, 12, float);
UNARY_ACTIVATION_OP_HFD(HardSigmoid, 6);
UNARY_ACTIVATION_OP_HFD(LeakyRelu, 6);
UNARY_ACTIVATION_OP_HFD(Relu, 6);
UNARY_ACTIVATION_OP_HFD(Selu, 6);
UNARY_ACTIVATION_OP_HFD(Sigmoid, 6);
UNARY_ACTIVATION_OP_HFD(Softplus, 1);
UNARY_ACTIVATION_OP_HFD(Softsign, 1);
UNARY_ACTIVATION_OP_HFD(Tanh, 6);
UNARY_ACTIVATION_OP_HFD(ThresholdedRelu, 10);

Status Celu::ComputeInternal(OpKernelContext* context) const {
  UnaryElementwisePreparation p;
  UnaryElementwise::Prepare(context, &p);
  CtxCelu func_ctx = MakeFuncCtx();
  // Float support only for now
  Impl_Celu(
      reinterpret_cast<const typename ToCudaType<float>::MappedType*>(p.input_tensor->template Data<float>()),
      reinterpret_cast<typename ToCudaType<float>::MappedType*>(p.output_tensor->template MutableData<float>()),
      &func_ctx, p.output_tensor->Shape().Size());

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
