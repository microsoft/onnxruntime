// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/nn/dropout.h"

#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(OpName, Domain, VER, T1, T2, MemIndex, ClassName)  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                 \
      OpName,                                                                    \
      Domain,                                                                    \
      VER,                                                                       \
      T1##_##T2,                                                                 \
      kCudaExecutionProvider,                                                    \
      KernelDefBuilder()                                                         \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())                \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())               \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>())             \
          .InputMemoryType<OrtMemTypeCPUInput>(MemIndex),                        \
      ClassName<T1, T2>);

REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, MLFloat16, MLFloat16, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, MLFloat16, float, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, MLFloat16, double, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, MLFloat16, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, float, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, double, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, MLFloat16, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, float, 1, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, double, 1, Dropout)

REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, MLFloat16, MLFloat16, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, MLFloat16, float, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, MLFloat16, double, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, MLFloat16, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, float, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, double, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, MLFloat16, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, float, 2, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, double, 2, DropoutGrad)

}  // namespace cuda
}  // namespace onnxruntime
