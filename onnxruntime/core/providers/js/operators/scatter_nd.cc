// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/js/js_data_types.h"
#include "scatter_nd.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    18,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    16,
    17,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    13,
    15,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    ScatterND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    ScatterND,
    kOnnxDomain,
    11,
    12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    ScatterND);

}  // namespace js
}  // namespace onnxruntime
