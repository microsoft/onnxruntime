// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/resize.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    10,
    10,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Resize<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    10,
    10,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Resize<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    10,
    10,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    Resize<uint8_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    11, 12,
    float,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    Resize<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    11, 12,
    int32_t,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Resize<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Resize,
    11, 12,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()),
    Resize<uint8_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Resize,
    13,
    float,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<float>()),
    Resize<float>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Resize,
    13,
    int32_t,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<int32_t>()),
    Resize<int32_t>);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    Resize,
    13,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()),
    Resize<uint8_t>);

}  // namespace onnxruntime
