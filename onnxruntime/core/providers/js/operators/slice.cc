// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "slice.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    1, 9,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Slice_1);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    10, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Slice);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Slice,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Slice);

ONNX_OPERATOR_KERNEL_EX(
    Slice,
    kOnnxDomain,
    13,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 1)
        .InputMemoryType(OrtMemTypeCPU, 2)
        .InputMemoryType(OrtMemTypeCPU, 3)
        .InputMemoryType(OrtMemTypeCPU, 4)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Slice);

}  // namespace js
}  // namespace onnxruntime
