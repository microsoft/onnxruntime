// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/nn/dropout_op.h"
#include <chrono>
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

// TrainableDropout is the same as Dropout V12.
// Registering the operator for the sake of backward compatibility.
// Give notice to the users to use Dropout V12 and then deprecate this kernel.

// TrainableDropout
#define REGISTER_KERNEL_TYPED(OpName, Domain, VER, T1, T2, ClassName) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                      \
      OpName,                                                         \
      Domain,                                                         \
      VER,                                                            \
      T1##_##T2,                                                      \
      kCpuExecutionProvider,                                          \
      KernelDefBuilder()                                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T1>())     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T2>())    \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<bool>()), \
      ClassName<T1, T2>);

// REVIEW(mzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
// However these types work on GPU implementation.
// REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
// REGISTER_KERNEL_TYPED(MLFloat16, float)
// REGISTER_KERNEL_TYPED(MLFloat16, double)

REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, MLFloat16, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, float, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, float, double, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, MLFloat16, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, float, Dropout)
REGISTER_KERNEL_TYPED(TrainableDropout, kOnnxDomain, 9, double, double, Dropout)


// TrainableDropoutGrad
// REVIEW(mzs): ConstEigenVectorArrayMap.cast<MLFLoat16) does not seem to be supported.
// However these types work on GPU implementation.
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, MLFloat16)
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, float)
// REGISTER_GRADIENT_KERNEL_TYPED(MLFloat16, double)

REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, MLFloat16, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, float, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, float, double, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, MLFloat16, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, float, DropoutGrad)
REGISTER_KERNEL_TYPED(TrainableDropoutGrad, kMSDomain, 1, double, double, DropoutGrad)

}  // namespace contrib
}  // namespace onnxruntime
