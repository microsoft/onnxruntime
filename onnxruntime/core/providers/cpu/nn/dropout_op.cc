// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/dropout_op.h"

namespace onnxruntime {

// Dropout
#define REGISTER_KERNEL_VERSIONED_TYPED(OpName, START_VER, END_VER, T1, T2)            \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                                            \
      OpName,                                                                          \
      START_VER,                                                                       \
      END_VER,                                                                         \
      T1##_##T2,                                                                       \
      KernelDefBuilder()                                                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())                      \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())                     \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()),                  \
      Dropout<T1, T2>);

#define REGISTER_KERNEL_TYPED(OpName, VER, T1, T2)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OpName,                                                         \
      kOnnxDomain,                                                    \
      VER,                                                            \
      T1##_##T2,                                                      \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()), \
      Dropout<T1, T2>);

// REVIEW(mzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
// However these types work on GPU implementation.
// REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
// REGISTER_KERNEL_TYPED(MLFloat16, float)
// REGISTER_KERNEL_TYPED(MLFloat16, double)

REGISTER_KERNEL_VERSIONED_TYPED(Dropout, 12, 12, float, float)
REGISTER_KERNEL_VERSIONED_TYPED(Dropout, 12, 12, float, double)
REGISTER_KERNEL_VERSIONED_TYPED(Dropout, 12, 12, double, float)
REGISTER_KERNEL_VERSIONED_TYPED(Dropout, 12, 12, double, double)

REGISTER_KERNEL_TYPED(Dropout, 13, float, float)
REGISTER_KERNEL_TYPED(Dropout, 13, float, double)
REGISTER_KERNEL_TYPED(Dropout, 13, double, float)
REGISTER_KERNEL_TYPED(Dropout, 13, double, double)
}  // namespace onnxruntime
