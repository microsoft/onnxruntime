// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gemm,
    7,
    8,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

// opset 9 added support for additional types (int32, uint32, int64, uint64), however we haven't enabled those yet.
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gemm,
    9,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

// opset 11 made bias input 'C' optional
ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Gemm,
    11,
    12,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

// opset 13 Adds BFloat16 support but we are not supporting it yet
ONNX_CPU_OPERATOR_KERNEL(
    Gemm,
    13,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);
}  // namespace onnxruntime
