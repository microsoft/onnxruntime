// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"

#include "gemm.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Gemm);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Gemm);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Gemm);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7, 8,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Gemm);

}  // namespace js
}  // namespace onnxruntime
