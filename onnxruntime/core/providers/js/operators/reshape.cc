// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reshape.h"
#include "core/providers/js/js_data_types.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    14,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    13, 13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    5, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", JsepSupportedDataTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Reshape);

}  // namespace js
}  // namespace onnxruntime
