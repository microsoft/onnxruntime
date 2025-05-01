// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/shape_op.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    1,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    13, 14,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    15, 18,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    19, 20,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

// Opset 21 added support for int4 and uint4.
// TODO(adrianlizarraga): Implement int4 and uint4 support.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Shape,
    21,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

// Opset 23 added support for float4e2m1.
// TODO(titaiwang): Implement float4e2m1 support.
ONNX_CPU_OPERATOR_KERNEL(
    Shape,
    23,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypesIRv9()).TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace onnxruntime
