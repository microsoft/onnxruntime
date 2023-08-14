// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "range.h"

namespace onnxruntime {
namespace js {
ONNX_OPERATOR_KERNEL_EX(
    Range,
    kOnnxDomain,
    11,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Range);
}  // namespace js
}  // namespace onnxruntime
