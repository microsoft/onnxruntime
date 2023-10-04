// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "expand.h"

namespace onnxruntime {
namespace js {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Expand,
    kOnnxDomain,
    8,
    12,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Expand);

ONNX_OPERATOR_KERNEL_EX(
    Expand,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Expand);
}  // namespace js
}  // namespace onnxruntime
