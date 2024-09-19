// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cumsum.h"

#include "core/providers/js/js_kernel.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    11, 13,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<TypeList<float, int32_t, uint32_t>>())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

ONNX_OPERATOR_KERNEL_EX(
    CumSum,
    kOnnxDomain,
    14,
    kJsExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("T2", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    CumSum);

}  // namespace js
}  // namespace onnxruntime
