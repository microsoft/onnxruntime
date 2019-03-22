// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops\cpu\resize.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    Resize,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Resize<float>);
}  // namespace contrib
}  // namespace onnxruntime
