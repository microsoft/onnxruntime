// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/softmax.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    11,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    11,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    11,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    1,
    10,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 11 starts to support Neg Axis.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    11,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

}  // namespace onnxruntime
