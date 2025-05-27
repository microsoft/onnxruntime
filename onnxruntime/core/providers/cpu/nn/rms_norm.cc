// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rms_norm.h"

#include "core/providers/common.h"

namespace onnxruntime {
#define REGISTER_ONNX_KERNEL_TYPED(T)                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(RMSNormalization, 23, T,                                        \
                                 KernelDefBuilder()                                              \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
                                     .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()), \
                                 RMSNorm);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(double)
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

}  // namespace onnxruntime
