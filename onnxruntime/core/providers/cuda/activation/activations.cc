// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "activations.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ACTIVATION_VERSIONED_KERNEL(x, startver, endver, T) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                           \
      x,                                                             \
      kOnnxDomain,                                                   \
      startver,                                                      \
      endver,                                                        \
      T,                                                             \
      kCudaExecutionProvider,                                        \
      (*KernelDefBuilder::Create())                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())     \
          .MayInplace(0, 0),                                         \
      x<T>);

#define REGISTER_ACTIVATION_KERNEL(x, ver, T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      x,                                                         \
      kOnnxDomain,                                               \
      ver,                                                       \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      x<T>);

#define UNARY_ACTIVATION_COMPUTE(x, T)                                                                     \
  template <>                                                                                              \
  Status x<T>::ComputeInternal(OpKernelContext* context) const {                                           \
    UnaryElementwisePreparation p;                                                                         \
    ORT_RETURN_IF_ERROR(UnaryElementwise::Prepare(context, &p));                                           \
    Ctx##x func_ctx = MakeFuncCtx();                                                                       \
    Impl_##x<typename ToCudaType<T>::MappedType>(                                                          \
        Stream(),                                                                                          \
        reinterpret_cast<const typename ToCudaType<T>::MappedType*>(p.input_tensor->template Data<T>()),   \
        reinterpret_cast<typename ToCudaType<T>::MappedType*>(p.output_tensor->template MutableData<T>()), \
        &func_ctx, p.output_tensor->Shape().Size());                                                       \
                                                                                                           \
    return Status::OK();                                                                                   \
  }

#define UNARY_ACTIVATION_OP_VERSIONED_TYPED(name, startver, endver, T) \
  REGISTER_ACTIVATION_VERSIONED_KERNEL(name, startver, endver, T)

#define UNARY_ACTIVATION_OP_VERSIONED_HFD(name, startver, endver)        \
  UNARY_ACTIVATION_OP_VERSIONED_TYPED(name, startver, endver, MLFloat16) \
  UNARY_ACTIVATION_OP_VERSIONED_TYPED(name, startver, endver, float)     \
  UNARY_ACTIVATION_OP_VERSIONED_TYPED(name, startver, endver, double)

#define UNARY_ACTIVATION_OP_TYPED(name, ver, T) \
  REGISTER_ACTIVATION_KERNEL(name, ver, T)      \
  UNARY_ACTIVATION_COMPUTE(name, T)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#define UNARY_ACTIVATION_OP_TYPED_BF16(name, ver) UNARY_ACTIVATION_OP_TYPED(name, ver, BFloat16)
#else
#define UNARY_ACTIVATION_OP_TYPED_BF16(name, ver)
#endif

#define UNARY_ACTIVATION_OP_HFD(name, ver)        \
  UNARY_ACTIVATION_OP_TYPED(name, ver, MLFloat16) \
  UNARY_ACTIVATION_OP_TYPED_BF16(name, ver)       \
  UNARY_ACTIVATION_OP_TYPED(name, ver, float)     \
  UNARY_ACTIVATION_OP_TYPED(name, ver, double)

UNARY_ACTIVATION_OP_HFD(Elu, 6);
UNARY_ACTIVATION_OP_HFD(HardSigmoid, 6);
UNARY_ACTIVATION_OP_HFD(LeakyRelu, 6);
UNARY_ACTIVATION_OP_HFD(Relu, 13);
UNARY_ACTIVATION_OP_VERSIONED_HFD(Relu, 6, 12);
UNARY_ACTIVATION_OP_HFD(Selu, 6);
UNARY_ACTIVATION_OP_HFD(Sigmoid, 13);
UNARY_ACTIVATION_OP_VERSIONED_HFD(Sigmoid, 6, 12);
UNARY_ACTIVATION_OP_HFD(Softplus, 1);
UNARY_ACTIVATION_OP_HFD(Softsign, 1);
UNARY_ACTIVATION_OP_HFD(Tanh, 13);
UNARY_ACTIVATION_OP_VERSIONED_HFD(Tanh, 6, 12);
UNARY_ACTIVATION_OP_HFD(ThresholdedRelu, 10);

}  // namespace cuda
}  // namespace onnxruntime
