// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/logsoftmax.h"

namespace onnxruntime {
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LogSoftmax<float>);
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    LogSoftmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    11,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    LogSoftmax<float>);
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    11,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    LogSoftmax<double>);
}  // namespace onnxruntime
