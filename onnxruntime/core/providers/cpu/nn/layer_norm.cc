// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.h"

#include "core/providers/common.h"

namespace onnxruntime {
#define REGISTER_ONNX_KERNEL_TYPED(T)                                                            \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(LayerNormalization, 17, T,                                      \
                                 KernelDefBuilder()                                              \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())      \
                                     .TypeConstraint("U", DataTypeImpl::GetTensorType<float>()), \
                                 LayerNorm);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(double)

}  // namespace onnxruntime
