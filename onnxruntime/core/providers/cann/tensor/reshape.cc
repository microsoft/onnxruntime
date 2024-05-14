// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/tensor/reshape.h"

namespace onnxruntime {
namespace cann {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    1,
    4,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Reshape_1);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    5, 12,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    13, 13,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

ONNX_OPERATOR_KERNEL_EX(
    Reshape,
    kOnnxDomain,
    14,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>())
        .Alias(0, 0)
        .InputMemoryType(OrtMemTypeCPUInput, 1),
    Reshape);

}  // namespace cann
}  // namespace onnxruntime
