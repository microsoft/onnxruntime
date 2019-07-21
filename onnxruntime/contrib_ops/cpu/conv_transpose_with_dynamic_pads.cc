// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose_with_dynamic_pads.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    ConvTransposeWithDynamicPads,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTransposeWithDynamicPads<float>);
}  // namespace contrib
}  // namespace onnxruntime
