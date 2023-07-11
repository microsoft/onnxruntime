// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "concat.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    1, 3,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    4, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat);

ONNX_OPERATOR_KERNEL_EX(
    Concat,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Concat);

}  // namespace js
}  // namespace onnxruntime
