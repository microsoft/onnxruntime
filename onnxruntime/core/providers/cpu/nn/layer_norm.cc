// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "layer_norm.h"

#include "core/providers/common.h"

namespace onnxruntime {
// official onnx operator registration. originally LayerNormalization was a contrib op.
// Only 2 type constraints (values using 'T' and 'U' in the contrib op all use 'T' in the ONNX spec)
#define REGISTER_ONNX_KERNEL_TYPED(T)                                                        \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(LayerNormalization, 17, T,                                  \
                                 KernelDefBuilder()                                          \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())  \
                                     .TypeConstraint("U", DataTypeImpl::GetTensorType<T>()), \
                                 LayerNorm);

// ONNX LayerNorm doesn't support 'double' for Mean/InvStdDev so we can only register a version with float
// with our current implementation which originally handled 'double' and 'float' for the contrib op.
REGISTER_ONNX_KERNEL_TYPED(float)

}  // namespace onnxruntime
