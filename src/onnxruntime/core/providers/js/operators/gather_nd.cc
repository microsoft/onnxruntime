// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/js/js_data_types.h"
#include "gather_nd.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    GatherND,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    GatherND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherND,
    kOnnxDomain,
    12,
    12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    GatherND);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    GatherND,
    kOnnxDomain,
    11,
    11,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes()),
    GatherND);

}  // namespace js
}  // namespace onnxruntime
