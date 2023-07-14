// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "resize.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    10, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    11, 12,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)  // roi
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // scales
        .InputMemoryType(OrtMemTypeCPUInput, 3)  // sizes
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    13,
    17,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Resize,
    kOnnxDomain,
    18,
    18,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    Resize);

ONNX_OPERATOR_KERNEL_EX(
    Resize,
    kOnnxDomain,
    19,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
    Resize);

}  // namespace js
}  // namespace onnxruntime
