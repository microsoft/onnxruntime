// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "batch_norm.h"

namespace onnxruntime {
namespace js {

#define REGISTER_BATCHNORM_KERNEL(OP_TYPE, DOMAIN, TYPE, KERNEL_CLASS)                                    \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                \
      OP_TYPE, DOMAIN, 7, 8, TYPE, kJsExecutionProvider,                                                  \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), KERNEL_CLASS);        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                                \
      OP_TYPE, DOMAIN, 9, 13, TYPE, kJsExecutionProvider,                                                 \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()), KERNEL_CLASS);        \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(OP_TYPE, DOMAIN, 14, 14, TYPE, kJsExecutionProvider,            \
                                          KernelDefBuilder()                                              \
                                              .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())  \
                                              .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()), \
                                          KERNEL_CLASS);                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(OP_TYPE, DOMAIN, 15, TYPE, kJsExecutionProvider,                          \
                                KernelDefBuilder()                                                        \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())            \
                                    .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())           \
                                    .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),          \
                                KERNEL_CLASS);

template <typename T>
T declval();

#define REGISTER_KERNEL_TYPED(T)                                                                                    \
  REGISTER_BATCHNORM_KERNEL(BatchNormalization, kMSInternalNHWCDomain, T, decltype(declval<BatchNorm<T, true>>())); \
  REGISTER_BATCHNORM_KERNEL(BatchNormalization, kOnnxDomain, T, decltype(declval<BatchNorm<T, false>>()));

REGISTER_KERNEL_TYPED(float);

}  // namespace js
}  // namespace onnxruntime
