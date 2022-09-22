// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Huawei. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cann/tensor/identity_op.h"

namespace onnxruntime {
namespace cann {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    7, 9,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .Alias(0, 0),
    IdentityOp<true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    10,
    11,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>())
        .Alias(0, 0),
    IdentityOp<true>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    1, 12,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Identity,
    kOnnxDomain,
    13, 13,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

ONNX_OPERATOR_KERNEL_EX(
    Identity,
    kOnnxDomain,
    14,
    kCannExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);

}  // namespace cann
}  // namespace onnxruntime
