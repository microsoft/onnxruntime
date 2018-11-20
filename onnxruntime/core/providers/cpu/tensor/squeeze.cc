// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Squeeze,
    1,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

}  // namespace onnxruntime
