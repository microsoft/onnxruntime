// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatten.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    1, 8,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    9, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Flatten);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Flatten);

ONNX_OPERATOR_KERNEL_EX(
    Flatten,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Flatten);

}  // namespace js
}  // namespace onnxruntime
