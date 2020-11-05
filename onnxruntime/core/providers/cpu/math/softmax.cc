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
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
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
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Softmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Softmax,
    13,
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
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    float,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Softmax<float>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
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
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    LogSoftmax,
    11,
    12,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);

// Opset 13 changed the semantic meaning of the axis attribute.
ONNX_CPU_OPERATOR_TYPED_KERNEL(
    LogSoftmax,
    13,
    double,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
    Softmax<double>);
}  // namespace onnxruntime
