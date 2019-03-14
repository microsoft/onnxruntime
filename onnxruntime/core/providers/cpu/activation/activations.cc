// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#if defined(USE_MLAS)
#include "core/mlas/inc/mlas.h"
#endif

namespace onnxruntime {

#define REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(alias, x, sinceVersion)                              \
  ONNX_CPU_OPERATOR_KERNEL(                                                                          \
      alias,                                                                                         \
      sinceVersion,                                                                                  \
      KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), \
      x<float>);

#define REGISTER_UNARY_ELEMENTWISE_KERNEL(x, sinceVersion) \
  REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(x, x, sinceVersion)

REGISTER_UNARY_ELEMENTWISE_KERNEL(Elu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(HardSigmoid, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(LeakyRelu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Relu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ScaledTanh, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Selu, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Sigmoid, 6);
// SoftPlus is the default case for ParametricSoftPlus
REGISTER_UNARY_ELEMENTWISE_KERNEL_ALIAS(Softplus, ParametricSoftplus, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Softsign, 1);
REGISTER_UNARY_ELEMENTWISE_KERNEL(Tanh, 6);
REGISTER_UNARY_ELEMENTWISE_KERNEL(ThresholdedRelu, 1);

template <>
Status Sigmoid<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);
#if defined(USE_MLAS)
  MlasComputeLogistic(X->template Data<float>(), Y->template MutableData<float>(), x_shape.Size());
#else  // make Eigen the default
  ConstEigenVectorArrayMap<float> xm(X->template Data<float>(), X->Shape().Size());
  EigenVectorArrayMap<float> ym(Y->template MutableData<float>(), Y->Shape().Size());
  ym = (xm >= 0).select(1 / (1. + (-xm.abs()).exp()), 1 - 1 / (1. + (-xm.abs()).exp()));
#endif
  return Status::OK();
}

template <>
Status Tanh<float>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const auto& x_shape = X->Shape();
  Tensor* Y = context->Output(0, x_shape);
#if defined(USE_MLAS)
  MlasComputeTanh(X->template Data<float>(), Y->template MutableData<float>(), x_shape.Size());
#else  // make Eigen the default
  EigenVectorArrayMap<float>(Y->template MutableData<float>(), Y->Shape().Size()) =
      ConstEigenVectorArrayMap<float>(X->template Data<float>(), X->Shape().Size()).tanh();
#endif
  return Status::OK();
}

}  // namespace onnxruntime
