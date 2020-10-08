// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "identity_op.h"

namespace onnxruntime {
namespace rocm {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    7, 9,
    kRocmExecutionProvider,
    KernelDefBuilder()
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
    kRocmExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()})
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<bool>())
        .Alias(0, 0),
    IdentityOp<true>);

ONNX_OPERATOR_KERNEL_EX(
    Identity,
    kOnnxDomain,
    1,
    kRocmExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);
}  // namespace rocm
}  // namespace onnxruntime
