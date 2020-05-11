// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T1, T2)                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                     \
      Dropout,                                                       \
      kOnnxDomain,                                                   \
      12,                                                            \
      T1##_##T2,                                                     \
      kCudaExecutionProvider,                                        \
      KernelDefBuilder()                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())    \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()) \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                    \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                   \
      Dropout<T1, T2, false>);

REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(MLFloat16, double)
REGISTER_KERNEL_TYPED(float, MLFloat16)
REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(float, double)
REGISTER_KERNEL_TYPED(double, MLFloat16)
REGISTER_KERNEL_TYPED(double, float)
REGISTER_KERNEL_TYPED(double, double)

}  // namespace cuda
}  // namespace onnxruntime
