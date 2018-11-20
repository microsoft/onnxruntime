// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/identity_op.h"

namespace onnxruntime {

ONNX_CPU_OPERATOR_KERNEL(
    Dropout,
    7,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<MLFloat16>(), DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>()}),
    IdentityOp<true>);

ONNX_CPU_OPERATOR_KERNEL(
    Identity,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()).Alias(0, 0),
    IdentityOp<false>);

}  // namespace onnxruntime
