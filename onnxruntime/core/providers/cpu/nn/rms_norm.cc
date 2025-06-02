// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/rms_norm.h"

#include "core/providers/common.h"

namespace onnxruntime {
// RMSNorm uses LayerNorm kernel, which only supports X and scale both
// being the same data type.
#define REGISTER_ONNX_KERNEL_TYPED(T)                                                        \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(RMSNormalization, 23, T,                                    \
                                 KernelDefBuilder()                                          \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
                                     .TypeConstraint("V", DataTypeImpl::GetTensorType<T>()), \
                                 RMSNorm);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(double)
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

}  // namespace onnxruntime
