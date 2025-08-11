// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/squeeze.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    1,
    10,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// axes is input instead of attribute
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    13,
    20,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// Opset 21 added support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
// TODO(adrianlizarraga): Implement support for float8e4m3fnuz, float8e5m2, float8e5m2fnuz, int4 and uint4.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    21,
    22,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// Opset 23 added support for float4e2m1.
// TODO: Add support for float4e2m1.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Squeeze,
    23,
    23,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

// Opset 24
ONNX_CPU_OPERATOR_KERNEL(
    Squeeze,
    24,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .Alias(0, 0),
    Squeeze);

}  // namespace onnxruntime
