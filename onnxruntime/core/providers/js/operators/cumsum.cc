// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11, 13,
    float,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    float,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedFloatTypes())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

}  // namespace js
}  // namespace onnxruntime
