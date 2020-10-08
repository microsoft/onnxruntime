// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "expand_dims.h"

namespace onnxruntime {
namespace contrib {
ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(
    ExpandDims,
    1,
    float,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("axis", DataTypeImpl::GetTensorType<int32_t>()),
    contrib::ExpandDims);
}  // namespace contrib
}  // namespace onnxruntime
