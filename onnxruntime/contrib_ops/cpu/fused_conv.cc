// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fused_conv.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    FusedConv,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    FusedConv<float>);
}  // namespace contrib
}  // namespace onnxruntime
