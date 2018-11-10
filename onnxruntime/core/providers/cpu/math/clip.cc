// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/clip.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Clip,
    6,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip<float>);

}  // namespace onnxruntime
