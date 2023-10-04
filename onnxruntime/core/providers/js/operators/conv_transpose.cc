// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

#include "conv_transpose.h"
namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    ConvTranspose,
    kMSInternalNHWCDomain,
    11,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    ConvTranspose<true, true>);

ONNX_OPERATOR_KERNEL_EX(
    ConvTranspose,
    kOnnxDomain,
    11,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    ConvTranspose<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ConvTranspose,
    kOnnxDomain,
    1, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    ConvTranspose<false>);

}  // namespace js
}  // namespace onnxruntime
