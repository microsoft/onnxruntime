// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "identity_op.h"

namespace onnxruntime {
namespace hip {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    7, 9,
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), 
                              DataTypeImpl::GetTensorType<float>(), 
                              DataTypeImpl::GetTensorType<double>()})
        .Alias(0, 0),
    IdentityOp<true>);

ONNX_OPERATOR_KERNEL_EX(
    Dropout,
    kOnnxDomain,
    10,
    kHipExecutionProvider,
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
    kHipExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .Alias(0, 0),
    IdentityOp<false>);
}  // namespace hip
}  // namespace onnxruntime
