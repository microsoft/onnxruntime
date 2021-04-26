// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/reshape.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    5,
    12,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_KERNEL(
    Reshape,
    13,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_KERNEL(
    Reshape,
    14,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    1,
    4,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Reshape_1);

}  // namespace onnxruntime
