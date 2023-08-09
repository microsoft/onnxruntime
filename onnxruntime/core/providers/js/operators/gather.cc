// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "gather.h"

namespace onnxruntime {
namespace js {

using AllSupportedSize =
    TypeList<
        float,
        double,
        int64_t,
        uint64_t,
        int32_t,
        uint32_t>;

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1,
    10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<AllSupportedSize>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()),
    Gather);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11,
    12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<AllSupportedSize>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()),
    Gather);

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraintsFromTypeList<AllSupportedSize>())
        .TypeConstraint("Tind", BuildKernelDefConstraintsFromTypeList<TypeList<int32_t, int64_t>>()),
    Gather);

}  // namespace js
}  // namespace onnxruntime
