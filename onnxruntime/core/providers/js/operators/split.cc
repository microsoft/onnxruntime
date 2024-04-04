// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "split.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split,
    kOnnxDomain,
    1, 1,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Split_1);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split,
    kOnnxDomain,
    2, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Split_2_10);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Split_11_12);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split,
    kOnnxDomain,
    13, 17,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Split_13_17);

ONNX_OPERATOR_KERNEL_EX(
    Split,
    kOnnxDomain,
    18,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .InputMemoryType(OrtMemTypeCPU, 1),
    Split_18);

}  // namespace js
}  // namespace onnxruntime
