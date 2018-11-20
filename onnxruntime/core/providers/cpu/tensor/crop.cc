// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/crop.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_KERNEL(
    Crop,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Crop<float>);
}  // namespace onnxruntime
