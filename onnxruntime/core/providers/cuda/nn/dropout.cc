// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/dropout.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      Dropout,                                                           \
      kOnnxDomain,                                                       \
      12,                                                                \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T1", DataTypeImpl::AllIEEEFloatTensorTypes()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())     \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                        \
          .InputMemoryType<OrtMemTypeCPUInput>(2),                       \
      Dropout<T, false>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)

}  // namespace cuda
}  // namespace onnxruntime
