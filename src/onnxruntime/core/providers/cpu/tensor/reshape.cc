// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/reshape.h"
namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    1,
    4,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Reshape_1);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    5,
    12,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    13,
    13,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    14, 18,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    19, 20,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

// Opset 21 added support for int4 and uint4.
// TODO(adrianlizarraga): Implement int4 and uint4 support.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    21,
    22,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Reshape,
    23,
    23,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

// Opset 24
ONNX_CPU_OPERATOR_KERNEL(
    Reshape,
    24,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9())
        .TypeConstraint("shape", DataTypeImpl::GetTensorType<int64_t>()),
    Reshape);

}  // namespace onnxruntime
