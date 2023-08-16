// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"
#include "core/providers/js/js_data_types.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Squeeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .Alias(0, 0),
    Squeeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    1, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .Alias(0, 0),
    Squeeze);

}  // namespace js
}  // namespace onnxruntime
