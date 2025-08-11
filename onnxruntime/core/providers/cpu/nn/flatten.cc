// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/nn/flatten.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    1,
    8,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    9,
    10,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    11,
    12,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    13,
    20,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

// Opset 21 added support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
// TODO(adrianlizarraga): Add support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    21,
    22,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Flatten,
    23,
    23,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

// Opset 24
ONNX_CPU_OPERATOR_KERNEL(
    Flatten,
    24,
    KernelDefBuilder()
        .Alias(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Flatten);

}  // namespace onnxruntime
