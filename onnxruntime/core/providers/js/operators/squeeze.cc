// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    14,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Squeeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    13, 13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Squeeze);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    5, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("axes", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPU, 1),
    Squeeze);

}  // namespace js
}  // namespace onnxruntime
