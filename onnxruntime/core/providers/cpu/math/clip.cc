// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/clip.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Clip,
    6,
    10,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip_6<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Clip,
    11,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Clip<float>);

}  // namespace onnxruntime
